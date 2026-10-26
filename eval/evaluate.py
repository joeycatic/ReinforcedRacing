"""One-lap policy evaluation on showcase and held-out pools; no replay export."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.config import load_env
from sim.observation import OBS_SIZE
from sim.pool import TrackPool, make_track_pool
from sim.vec_env import VecRaceEnv
from train.config import PPOConfig
from train.ppo import ActorCritic, RunningMeanStd, choose_device, load_checkpoint


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    stats: RunningMeanStd,
    pool: TrackPool,
    config: PPOConfig,
    *,
    runs: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    indices = np.repeat(np.arange(len(pool.tracks)), runs)
    env = VecRaceEnv(len(indices), pool, config=replace(load_env(), max_laps=1))
    observations, _ = env.reset(seed=seed, options={"track_indices": indices})
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    finished = np.zeros(len(indices), dtype=bool)
    records: list[dict[str, Any]] = []
    try:
        for _ in range(env.config.max_agent_steps):
            normalized = stats.normalize(
                observations, config.observation_clip, config.normalization_epsilon
            )
            mean = (
                model.actor(torch.as_tensor(normalized, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )
            if not deterministic:
                mean += model.log_std.exp().cpu().numpy() * rng.standard_normal(mean.shape)
            mean[finished] = 0.0
            observations, _, terminated, truncated, info = env.step(mean.clip(-1, 1))
            completed = (terminated | truncated) & ~finished
            for row in np.flatnonzero(completed):
                track = pool.tracks[indices[row]]
                success = bool(info["laps"][row] >= 1)
                records.append(
                    {
                        "track_id": track.id,
                        "track_seed": track.seed,
                        "run": int(row % runs),
                        "finished": success,
                        "crashed": bool(terminated[row]),
                        "lap_time": float(info["best_lap_time"][row]) if success else None,
                        "return": float(info["episode_return"][row]),
                        "progress": float(info["progress"][row]),
                        "agent_steps": int(info["agent_steps"][row]),
                    }
                )
            finished |= completed
            if finished.all():
                break
    finally:
        env.close()
    if not finished.all():
        raise RuntimeError("Evaluation ended without a terminal result for every scheduled run")
    records.sort(key=lambda record: (record["track_seed"], record["run"]))
    return {
        "deterministic": deterministic,
        "seed": seed,
        "runs_per_track": runs,
        "summary": summarize(records),
        "tracks": {
            track.id: summarize([r for r in records if r["track_id"] == track.id])
            for track in pool.tracks
        },
        "episodes": records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    times = [record["lap_time"] for record in records if record["finished"]]
    return {
        "episodes": len(records),
        "lap_completion_rate": sum(record["finished"] for record in records) / len(records),
        "crash_rate": sum(record["crashed"] for record in records) / len(records),
        "mean_lap_time": float(np.mean(times)) if times else None,
        "best_lap_time": min(times) if times else None,
        "mean_progress": float(np.mean([record["progress"] for record in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/training/default/latest.pt")
    )
    parser.add_argument(
        "--snapshots", action="store_true", help="Evaluate all sibling scheduled snapshots"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Runs per track; each stops after one lap"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action="store_true", help="Use policy means instead of Gaussian samples"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--showcase-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/m3-results.json"))
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    pools = {"showcase": make_track_pool("showcase")}
    if not args.showcase_only:
        pools["held_out"] = make_track_pool("held_out")
    paths = [args.checkpoint]
    if args.snapshots:
        paths = sorted(
            (args.checkpoint.parent / "snapshots").glob("*.pt"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else float("inf"),
        )
        if not paths:
            parser.error("No snapshots beside this checkpoint")
    results = {}
    print(
        f"{'snapshot':>12} {'pool':>10} {'laps':>10} {'crashes':>10} "
        f"{'mean(s)':>10} {'best(s)':>10}",
        flush=True,
    )
    for path in paths:
        checkpoint = load_checkpoint(path)
        config = PPOConfig(**checkpoint["config"])
        torch.set_num_threads(config.cpu_threads)
        model = ActorCritic(config).to(choose_device(args.device))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        stats = RunningMeanStd((OBS_SIZE,), config.initial_stats_count)
        stats.load_state_dict(checkpoint["obs_stats"])
        per_pool = {}
        for name, pool in pools.items():
            result = evaluate_policy(
                model,
                stats,
                pool,
                config,
                runs=args.runs,
                seed=args.seed,
                deterministic=args.deterministic,
            )
            per_pool[name] = result
            summary = result["summary"]
            mean = "—" if summary["mean_lap_time"] is None else f"{summary['mean_lap_time']:.2f}"
            best = "—" if summary["best_lap_time"] is None else f"{summary['best_lap_time']:.2f}"
            print(
                f"{path.stem:>12} {name:>10} {summary['lap_completion_rate']:>9.1%} "
                f"{summary['crash_rate']:>9.1%} {mean:>10} {best:>10}",
                flush=True,
            )
        results[str(path)] = {
            "global_step": checkpoint["global_step"],
            "stage": checkpoint["stage"],
            "training_seconds": checkpoint["training_seconds"],
            "checkpoint_sha256": checkpoint["source_sha256"],
            "training_seed": config.seed,
            "config": checkpoint["config"],
            "versions": checkpoint["versions"],
            **per_pool,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Evaluate every scheduled snapshot and record showcase replay/heatmap assets."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from eval.config import ExportConfig, load_export
from eval.evaluate import evaluate_policy
from eval.record import record_runs
from sim.config import ROOT, load_observation, load_physics, load_tracks
from sim.pool import TrackPool, make_track_pool
from train.config import PPOConfig
from train.ppo import ActorCritic, RunningMeanStd, load_checkpoint


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulation_config() -> dict:
    return {
        "physics": asdict(load_physics()),
        "tracks": asdict(load_tracks()[0]),
        "observation": asdict(load_observation()),
    }


def snapshot_paths(run_dir: Path) -> list[Path]:
    first = run_dir / "snapshots/0.pt"
    if not first.exists():
        raise FileNotFoundError(f"Missing {first}; train a complete run before export")
    checkpoint = load_checkpoint(first)
    names = [str(step) for step in checkpoint["config"]["snapshot_steps"]] + ["final"]
    paths = [run_dir / "snapshots" / f"{name}.pt" for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing scheduled checkpoints: " + ", ".join(missing))
    return paths


def fingerprint(checkpoint: dict, config: ExportConfig, pools: list[TrackPool]) -> str:
    # Changing recorder/simulation code invalidates cached evaluations, even with v1 schemas.
    sources = [path for folder in ("sim", "eval") for path in sorted((ROOT / folder).glob("*.py"))]
    data = {
        "checkpoint": checkpoint["source_sha256"],
        "config": asdict(config),
        "simulation": simulation_config(),
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "tracks": [[track.to_dict() for track in pool.tracks] for pool in pools],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def evaluate_snapshot(
    path: Path, config: ExportConfig, showcase: TrackPool, held_out: TrackPool
) -> dict:
    checkpoint = load_checkpoint(path)
    if str(checkpoint.get("requested_step")) != path.stem:
        raise ValueError(f"Snapshot identity disagrees with filename: {path}")
    signature = fingerprint(checkpoint, config, [showcase, held_out])
    directory = ROOT / config.evaluation_dir / path.stem
    report_path = directory / "summary.json"
    if report_path.is_file():
        cached = json.loads(report_path.read_text())
        if cached.get("fingerprint") == signature and all(
            (directory / name).is_file() and sha256(directory / name) == digest
            for name, digest in cached["files"].items()
        ):
            return cached
    deterministic, replays, _ = record_runs(checkpoint, showcase, config, deterministic=True)
    stochastic, _, heatmaps = record_runs(checkpoint, showcase, config, deterministic=False)
    ppo = PPOConfig(**checkpoint["config"])
    torch.set_num_threads(ppo.cpu_threads)
    model = ActorCritic(ppo).eval()
    model.load_state_dict(checkpoint["model"])
    stats = RunningMeanStd((22,), ppo.initial_stats_count)
    stats.load_state_dict(checkpoint["obs_stats"])
    held = evaluate_policy(
        model,
        stats,
        held_out,
        ppo,
        runs=config.held_out_runs,
        seed=config.seed,
        deterministic=False,
    )
    files = {}
    for track, replay, heatmap in zip(showcase.tracks, replays, heatmaps, strict=True):
        for kind, value in (("replay", replay), ("heatmap", heatmap)):
            name = f"{track.id}.{kind}.v1.json"
            write_json(directory / name, value)
            files[name] = sha256(directory / name)
    report = {
        "fingerprint": signature,
        "checkpoint_sha256": checkpoint["source_sha256"],
        "snapshot_id": path.stem,
        "global_step": checkpoint["global_step"],
        "versions": checkpoint["versions"],
        "config": asdict(config),
        "files": files,
        "deterministic": deterministic,
        "showcase": stochastic,
        "held_out": held,
    }
    write_json(report_path, report)
    return report


def run_evaluation(config: ExportConfig) -> list[dict]:
    showcase, held_out = make_track_pool("showcase"), make_track_pool("held_out")
    print(
        f"{'snapshot':>10} {'pool':>10} {'laps':>9} {'crashes':>9} {'mean(s)':>9} {'best(s)':>9}",
        flush=True,
    )
    reports = []
    for path in snapshot_paths(ROOT / config.run_dir):
        print(f"Evaluating snapshot {path.stem}…", flush=True)
        report = evaluate_snapshot(path, config, showcase, held_out)
        reports.append(report)
        for name in ("showcase", "held_out"):
            summary = report[name]["summary"]
            mean = "—" if summary["mean_lap_time"] is None else f"{summary['mean_lap_time']:.2f}"
            best = "—" if summary["best_lap_time"] is None else f"{summary['best_lap_time']:.2f}"
            print(
                f"{path.stem:>10} {name:>10} {summary['lap_completion_rate']:>8.1%} "
                f"{summary['crash_rate']:>8.1%} {mean:>9} {best:>9}",
                flush=True,
            )
    write_json(ROOT / config.evaluation_dir / "summary.json", reports)
    return reports


def parse_config() -> ExportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/export.yaml")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    config = load_export(args.config)
    return replace(config, run_dir=str(args.run_dir)) if args.run_dir else config


def main() -> None:
    if "--checkpoint" in sys.argv:
        from eval.evaluate import main as summary_main

        summary_main()
    else:
        run_evaluation(parse_config())


if __name__ == "__main__":
    main()

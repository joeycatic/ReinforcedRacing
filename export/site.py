"""Shared loaders for the committed, source-backed portfolio evidence."""

import json
from pathlib import Path

import yaml

from sim.config import ROOT

SMOOTH = 15


def load_training(run_dir: Path) -> tuple[list[tuple[int, float]], int]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    switch = next(row["step"] for row in rows if row["stage"] != rows[0]["stage"])
    points = []
    for i, row in enumerate(rows):
        window = rows[max(0, i - SMOOTH + 1) : i + 1]
        points.append(
            (row["step"], sum(r["rollout/episodic_return"] for r in window) / len(window))
        )
    return points, switch


def load_snapshots(summary: Path) -> list[tuple[int, float, float]]:
    return [
        (
            s["global_step"],
            100 * s["showcase"]["summary"]["lap_completion_rate"],
            100 * s["held_out"]["summary"]["lap_completion_rate"],
        )
        for s in json.loads(summary.read_text())
    ]


def evaluation(seed: int, suffix: str) -> dict:
    data = json.loads((ROOT / f"experiments/evaluation/seed{seed}-6m-{suffix}.json").read_text())
    return next(iter(data.values()))


def build_site_data() -> dict:
    snapshots = load_snapshots(ROOT / "artifacts/evaluation/m4/summary.json")
    validation = json.loads((ROOT / "docs/m4-validation.json").read_text())
    results = json.loads((ROOT / "experiments/m3-results.json").read_text())
    tracks = yaml.safe_load((ROOT / "configs/tracks.yaml").read_text())["pools"]
    ppo = yaml.safe_load((ROOT / "configs/ppo.yaml").read_text())
    seeds = []
    for seed in (1, 2):
        showcase = evaluation(seed, "showcase50")["showcase"]["summary"]
        held = evaluation(seed, "stochastic")["held_out"]["summary"]
        deterministic = evaluation(seed, "deterministic")["showcase"]["tracks"]
        seeds.append(
            {
                "seed": seed,
                "showcase_pct": 100 * showcase["lap_completion_rate"],
                "held_out_pct": 100 * held["lap_completion_rate"],
                "deterministic_showcase_finishes": sum(
                    t["lap_completion_rate"] == 1 for t in deterministic.values()
                ),
            }
        )
    return {
        "stats": {
            "showcase_completion": snapshots[-1][1],
            "held_out_completion": snapshots[-1][2],
            "parity_max_error": validation["parity"]["max_measured_absolute_error"],
            "training_steps": snapshots[-1][0],
            "training_minutes": results["runs"]["2"]["final_training_seconds"] / 60,
            "training_tracks": tracks["training_count"],
            "held_out_tracks": tracks["held_out_count"],
            "parallel_cars": ppo["num_envs"],
            "rl_libraries": 0,
        },
        "curves": {
            str(seed): load_training(ROOT / f"artifacts/training/m3-seed{seed}")[0]
            for seed in (1, 2)
        },
        "snapshots": snapshots,
        "seeds": seeds,
    }


def build_parity_demo(fixture: Path) -> dict:
    source = json.loads(fixture.read_text())
    return {
        "track_id": source["track"]["id"],
        "initial_state": source["initial_state"],
        "actions": [s["action"] for s in source["steps"]],
        "python": [[s["state"][k] for k in ("x", "y", "psi")] for s in source["steps"]],
    }

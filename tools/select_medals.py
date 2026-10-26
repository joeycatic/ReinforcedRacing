"""Evaluate retained seed-2 checkpoints and write reproducible medal evidence."""

from dataclasses import asdict

import yaml

from eval.artifacts import sha256, write_json
from eval.config import load_export
from eval.medals import select_medals
from eval.record import record_runs
from sim.config import ROOT
from sim.pool import make_track_pool
from train.ppo import load_checkpoint


def main() -> None:
    config = load_export()
    run_dir = ROOT / config.run_dir
    pool = make_track_pool("showcase")
    candidates = {}
    evidence = {}
    paths = sorted((run_dir / "checkpoints").glob("*.pt"), key=lambda p: int(p.stem))
    for path in [run_dir / "snapshots/final.pt", *paths]:
        checkpoint = load_checkpoint(path)
        checkpoint.setdefault("requested_step", path.stem)
        summary, _, _ = record_runs(checkpoint, pool, config, deterministic=True)
        laps = {t: summary["tracks"][t]["mean_lap_time"] for t in summary["tracks"]}
        print(path.stem, laps, flush=True)
        if path.stem == "final":
            author = laps
            author_sha = sha256(path)
        else:
            step = int(path.stem)
            candidates[step] = laps
            evidence[str(step)] = {"checkpoint_sha256": sha256(path), "lap_times": laps}
    if any(v is None for v in author.values()):
        raise ValueError("Author must finish every showcase track")
    choices = select_medals(candidates, author)
    document = {
        "run_dir": config.run_dir,
        "gap": 0.03,
        "medals": [
            {"medal": c.medal, "step": c.step, "checkpoint": f"checkpoints/{c.step}.pt"}
            for c in choices
        ],
    }
    (ROOT / "configs/medals.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    write_json(
        ROOT / "experiments/medal-selection.json",
        {
            "run_dir": config.run_dir,
            "gap": 0.03,
            "author": {"checkpoint_sha256": author_sha, "lap_times": author},
            "candidates": evidence,
            "choices": [asdict(c) for c in choices],
        },
    )
    for choice in choices:
        print(choice.medal.upper(), choice.step, choice.lap_times, flush=True)


if __name__ == "__main__":
    main()

"""Evaluate existing seed-2 policies and select progressive showcase attempts."""

from dataclasses import asdict

import yaml

from eval.artifacts import sha256, write_json
from eval.attempts import AttemptChoice, Run, select_attempts
from eval.config import load_export
from eval.record import record_runs
from sim.config import ROOT
from sim.pool import make_track_pool
from train.ppo import load_checkpoint


def main() -> None:
    config = load_export()
    run_dir = ROOT / config.run_dir
    pool = make_track_pool("showcase")
    candidates = []
    replays_by_candidate = {}
    paths = list((run_dir / "snapshots").glob("*.pt"))
    paths += list((run_dir / "checkpoints").glob("*.pt"))
    policies = []
    for path in paths:
        checkpoint = load_checkpoint(path)
        source = path.stem if path.parent.name == "snapshots" else "checkpoint"
        step = int(checkpoint["global_step"])
        checkpoint.setdefault("requested_step", f"attempt-{step}")
        policies.append((step, source, path, checkpoint))
    policies.sort(key=lambda p: (p[1] == "final", p[0], p[1] == "checkpoint"))
    for step, source, path, checkpoint in policies:
        summary, replays, _ = record_runs(checkpoint, pool, config, deterministic=True)
        runs = {}
        for episode in summary["episodes"]:
            outcome = (
                "finished"
                if episode["finished"]
                else ("stalled" if episode["truncated"] else "crashed")
            )
            run = Run(step, source, outcome, episode["progress"], episode["lap_time"])
            runs[episode["track_id"]] = asdict(run)
        candidates.append({"checkpoint_sha256": sha256(path), "runs": runs})
        replays_by_candidate[(step, source)] = replays
        print(step, source, runs, flush=True)
    choices = {}
    scratch = ROOT / ".superpowers/sdd/2026-10-26-m8-progressive-watch/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for track in pool.tracks:
        runs = [Run(**candidate["runs"][track.id]) for candidate in candidates]
        selected = select_attempts(runs)
        first_finish = next((r for r in selected if r.outcome == "finished"), None)
        choices[track.id] = []
        print(f"\n{track.id}: step / outcome / progress / lap", flush=True)
        for run in selected:
            headline = (
                "Untrained policy"
                if run.step == 0
                else (
                    "Final policy"
                    if run.source == "final"
                    else ("First full lap" if run == first_finish else None)
                )
            )
            choices[track.id].append(asdict(AttemptChoice(run.step, run.source, headline)))
            print(run.step, run.outcome, f"{run.progress:.2f}", run.lap_time, flush=True)
            replay = next(
                r
                for r in replays_by_candidate[(run.step, run.source)]
                if r["metadata"]["track_id"] == track.id
            )
            write_json(scratch / f"{track.id}-{run.step}-{run.source}.json", replay)
    (ROOT / "configs/attempts.yaml").write_text(
        yaml.safe_dump({"run_dir": config.run_dir, "tracks": choices}, sort_keys=False)
    )
    write_json(
        ROOT / "experiments/attempt-selection.json",
        {"run_dir": config.run_dir, "candidates": candidates, "choices": choices},
    )


if __name__ == "__main__":
    main()

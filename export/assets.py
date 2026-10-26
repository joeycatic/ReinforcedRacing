"""Assemble and verify a self-contained, versioned static asset directory."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from eval.artifacts import (
    fingerprint,
    parse_config,
    run_evaluation,
    sha256,
    simulation_config,
    snapshot_paths,
    write_json,
)
from eval.attempts import load_attempts
from eval.config import ExportConfig
from eval.formats import HEATMAP_VERSION, REPLAY_VERSION
from eval.medals import load_medals
from eval.record import record_runs
from export.onnx import export_policy
from export.site import build_parity_demo, build_site_data
from sim import OBS_VERSION, PHYSICS_VERSION, TRACK_VERSION
from sim.config import ROOT
from sim.pool import TrackPool, make_track_pool
from train.ppo import load_checkpoint

VERSIONS = {
    "physics": PHYSICS_VERSION,
    "obs": OBS_VERSION,
    "track": TRACK_VERSION,
    "replay": REPLAY_VERSION,
    "heatmap": HEATMAP_VERSION,
}


def descriptor(root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def deterministic_replay_cache(
    checkpoint: dict, pool: TrackPool, config: ExportConfig, directory: Path
) -> dict:
    """Load verified deterministic replays, regenerating an incomplete or stale cache."""
    signature = fingerprint(checkpoint, config, [pool])
    cache = directory / "summary.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    names = {f"{track.id}.replay.v1.json" for track in pool.tracks}
    if (
        cached.get("fingerprint") != signature
        or set(cached.get("files", {})) != names
        or not all(
            (directory / name).is_file() and sha256(directory / name) == digest
            for name, digest in cached.get("files", {}).items()
        )
    ):
        summary, replays, _ = record_runs(checkpoint, pool, config, deterministic=True)
        files = {}
        for track, replay in zip(pool.tracks, replays, strict=True):
            name = f"{track.id}.replay.v1.json"
            write_json(directory / name, replay)
            files[name] = sha256(directory / name)
        cached = {"fingerprint": signature, "files": files, "summary": summary}
        write_json(cache, cached)
    return cached


def export_attempts(stage: Path, manifest: dict, config: ExportConfig) -> dict:
    evidence = json.loads((ROOT / "experiments/attempt-selection.json").read_text())
    candidates = {
        (track, run["step"], run["source"]): candidate["checkpoint_sha256"]
        for candidate in evidence["candidates"]
        for track, run in candidate["runs"].items()
    }
    snapshots = {snapshot["id"]: snapshot for snapshot in manifest["snapshots"]}
    showcase = make_track_pool("showcase")
    result = {}
    for track_id, choices in load_attempts().items():
        result[track_id] = []
        for choice in choices:
            expected_hash = candidates[track_id, choice.step, choice.source]
            if choice.source != "checkpoint":
                snapshot = snapshots[choice.source]
                if (
                    snapshot["checkpoint_sha256"] != expected_hash
                    or snapshot["step"] != choice.step
                ):
                    raise ValueError("Attempt snapshot changed since selection")
                asset = snapshot["replays"][track_id]
            else:
                path = ROOT / config.run_dir / "checkpoints" / f"{choice.step}.pt"
                checkpoint = load_checkpoint(path)
                if checkpoint["source_sha256"] != expected_hash:
                    raise ValueError(f"Attempt checkpoint changed since selection: {path}")
                checkpoint["requested_step"] = f"attempt-{choice.step}"
                directory = ROOT / config.evaluation_dir / "attempts" / str(choice.step)
                cached = deterministic_replay_cache(checkpoint, showcase, config, directory)
                name = f"{track_id}.replay.v1.json"
                source = directory / name
                if sha256(source) != cached["files"][name]:
                    raise ValueError("Attempt replay changed during export")
                destination = stage / "replays" / f"attempt-{choice.step}" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                asset = descriptor(stage, destination)
            metadata = json.loads((stage / asset["path"]).read_text())["metadata"]
            if (
                metadata["checkpoint_sha256"] != expected_hash
                or metadata["snapshot_step"] != choice.step
            ):
                raise ValueError("Attempt replay does not match selection")
            outcome = (
                "finished"
                if metadata["finished"]
                else "crashed"
                if metadata["crashed"]
                else "stalled"
            )
            result[track_id].append(
                {
                    "step": choice.step,
                    "source": choice.source,
                    "headline": choice.headline,
                    "outcome": outcome,
                    "progress": metadata["progress"],
                    "lap_time": metadata["lap_time"],
                    "replay": asset,
                }
            )
    return result


def build_assets(config: ExportConfig) -> dict:
    reports = run_evaluation(config)
    target = ROOT / config.assets_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    # Verify everything in a staging directory before replacing a working bundle.
    with tempfile.TemporaryDirectory(prefix=".assets-", dir=target.parent) as temporary:
        stage = Path(temporary)
        config_path = stage / "simulation.json"
        write_json(config_path, {"versions": VERSIONS, **simulation_config()})
        manifest = {
            "version": "manifest.v3",
            "versions": VERSIONS,
            "simulation": descriptor(stage, config_path),
            "tracks": [],
            "snapshots": [],
        }
        for track in make_track_pool("showcase").tracks:
            path = stage / "tracks" / f"{track.id}.track.v1.json"
            track.save(path)
            manifest["tracks"].append(
                {"id": track.id, "name": track.name, **descriptor(stage, path)}
            )
        rng = np.random.default_rng(config.seed)
        samples = rng.uniform(-1, 1, (config.onnx_validation_samples, 22))
        samples[:3] = [np.zeros(22), np.ones(22), -np.ones(22)]
        for path, report in zip(snapshot_paths(ROOT / config.run_dir), reports, strict=True):
            checkpoint = load_checkpoint(path)
            if checkpoint["source_sha256"] != report["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after evaluation: {path}")
            model_path = stage / "models" / f"{path.stem}.onnx"
            verification = export_policy(checkpoint, model_path, config, samples)
            snapshot = {
                "id": path.stem,
                "step": checkpoint["global_step"],
                "checkpoint_sha256": checkpoint["source_sha256"],
                "model": descriptor(stage, model_path),
                "onnx_validation": verification,
                "replays": {},
                "heatmaps": {},
            }
            for track in manifest["tracks"]:
                for kind in ("replay", "heatmap"):
                    name = f"{track['id']}.{kind}.v1.json"
                    source = ROOT / config.evaluation_dir / path.stem / name
                    if sha256(source) != report["files"][name]:
                        raise ValueError(f"Evaluation asset changed during export: {source}")
                    destination = stage / f"{kind}s" / path.stem / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    snapshot[f"{kind}s"][track["id"]] = descriptor(stage, destination)
            manifest["snapshots"].append(snapshot)
            print(
                f"ONNX {path.stem}: action {verification['max_action_error']:.3g}, "
                f"value {verification['max_value_error']:.3g}",
                flush=True,
            )
        manifest["medals"] = []
        evidence = json.loads((ROOT / "experiments/medal-selection.json").read_text())
        showcase = make_track_pool("showcase")
        for choice in load_medals():
            path = ROOT / config.run_dir / "checkpoints" / f"{choice.step}.pt"
            checkpoint = load_checkpoint(path)
            if (
                checkpoint["source_sha256"]
                != evidence["candidates"][str(choice.step)]["checkpoint_sha256"]
            ):
                raise ValueError(f"Medal checkpoint changed since selection: {path}")
            checkpoint["requested_step"] = f"medal-{choice.medal}"
            directory = ROOT / config.evaluation_dir / "medals" / str(choice.step)
            cached = deterministic_replay_cache(checkpoint, showcase, config, directory)
            laps = {
                t.id: cached["summary"]["tracks"][t.id]["mean_lap_time"] for t in showcase.tracks
            }
            if laps != choice.lap_times:
                raise ValueError("Medal lap times changed since selection")
            model = stage / "models" / f"medal-{choice.medal}.onnx"
            verification = export_policy(checkpoint, model, config, samples)
            medal = {
                "medal": choice.medal,
                "step": choice.step,
                "checkpoint_sha256": checkpoint["source_sha256"],
                "model": descriptor(stage, model),
                "lap_times": laps,
                "replays": {},
                "onnx_validation": verification,
            }
            for track in showcase.tracks:
                name = f"{track.id}.replay.v1.json"
                source = directory / name
                if sha256(source) != cached["files"][name]:
                    raise ValueError("Medal replay changed during export")
                destination = stage / "replays" / f"medal-{choice.medal}" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                medal["replays"][track.id] = descriptor(stage, destination)
            manifest["medals"].append(medal)
        manifest["attempts"] = export_attempts(stage, manifest, config)
        final = manifest["snapshots"][-1]
        author_laps = {
            t["id"]: reports[-1]["deterministic"]["tracks"][t["id"]]["mean_lap_time"]
            for t in manifest["tracks"]
        }
        manifest["medals"].append(
            {
                "medal": "author",
                "step": final["step"],
                "checkpoint_sha256": final["checkpoint_sha256"],
                "model": final["model"],
                "lap_times": author_laps,
                "replays": final["replays"],
            }
        )
        for key, name, value in [
            ("site", "site.json", build_site_data()),
            (
                "parity_demo",
                "parity-demo.json",
                build_parity_demo(ROOT / "tests/fixtures/parity/sweep.json"),
            ),
        ]:
            write_json(stage / name, value)
            manifest[key] = descriptor(stage, stage / name)
        write_json(stage / "manifest.json", manifest)
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except OSError:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    print(f"Exported {target}", flush=True)
    return manifest


def main() -> None:
    build_assets(parse_config())


if __name__ == "__main__":
    main()

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

import eval.artifacts as artifacts
import eval.evaluate as evaluate
import eval.record as recording
import export.assets as assets
from eval.config import load_export
from eval.formats import Heatmap, decode_replay, encode_replay
from export.onnx import export_policy
from sim.config import ROOT, EnvConfig
from sim.pool import TrackPool
from sim.track import Track
from tools.generate_parity import fixtures
from train.config import load_ppo
from train.ppo import VERSIONS, ActorCritic, RunningMeanStd


def policy() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = replace(load_ppo(), hidden_size=32, snapshot_steps=(0,))
    model = ActorCritic(config)
    stats = RunningMeanStd((22,), config.initial_stats_count)
    stats.update(np.random.default_rng(2).normal(0.3, 0.2, (100, 22)))
    return {
        "format": "ppo.v1",
        "config": asdict(config),
        "model": model.state_dict(),
        "obs_stats": stats.state_dict(),
        "versions": VERSIONS,
        "global_step": 0,
        "requested_step": 0,
        "source_sha256": "a" * 64,
    }


def test_replay_quantizes_absolute_samples_without_accumulating_error() -> None:
    config = load_export()
    frames = np.random.default_rng(5).normal(size=(12001, 6))
    encoded = encode_replay(frames, {"track_id": "test"}, 1 / 60, config)
    decoded = decode_replay(json.loads(json.dumps(encoded)))
    assert np.all(np.abs(frames - decoded) <= 0.5 / np.array(encoded["scales"]) + 1e-14)
    assert np.allclose(decoded[0], np.asarray(encoded["deltas"][0]) / encoded["scales"])
    with pytest.raises(ValueError, match="Unsupported"):
        decode_replay({**encoded, "version": "replay.v2"})


def test_heatmap_sparse_counts_and_crash_positions(circle: Track) -> None:
    heatmap = Heatmap(circle, load_export())
    positions = np.array([circle.centerline[0], circle.centerline[0], circle.centerline[1]])
    heatmap.add(positions, np.array([False, True, True]))
    heatmap.add(np.empty((0, 2)), np.array([], dtype=bool))
    data = heatmap.to_dict({})
    assert data["grid"]["cell_size"] == 0.5
    assert sum(count for _, count in data["visits"]) == data["visit_count"] == 3
    assert sum(count for _, count in data["crash_bins"]) == data["crash_count"] == 2
    assert data["crash_positions"] == positions[1:].tolist()
    with pytest.raises(ValueError, match="outside"):
        heatmap.add(np.array([[1e6, 1e6]]), np.array([True]))


def test_recording_keeps_final_tick_and_freezes_finished_rows(pool: TrackPool) -> None:
    config = replace(load_export(), stochastic_runs=2)
    env = recording.RecordingEnv(pool, config, 2, replay=True)
    env.reset(seed=42, options={"track_indices": env.assignments})
    # One row crashes on the first tick, the other continues. Autoreset must never
    # contaminate its old track's replay or add visits from a subsequent episode.
    env.states[0, :2] = pool.points[0] + [8, 0]
    env.step(np.zeros((2, 2)))
    assert env.frames is not None
    assert len(env.frames[0]) == 2
    env.recording[0] = False
    env.step(np.zeros((2, 2)))
    assert len(env.frames[0]) == 2
    assert len(env.frames[1]) == 5
    assert env.maps[0].to_dict({})["visit_count"] == 5
    assert env.maps[0].to_dict({})["crash_count"] == 1
    env.close()


def test_onnx_embeds_frozen_normalization_and_matches_dynamic_batches(tmp_path: Path) -> None:
    checkpoint = policy()
    observations = np.random.default_rng(9).normal(0, 4, (64, 22))
    path = tmp_path / "model.onnx"
    report = export_policy(checkpoint, path, load_export(), observations)
    assert max(report["max_action_error"], report["max_value_error"]) <= 1e-5
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model = ActorCritic(replace(load_ppo(), hidden_size=32))
    model.load_state_dict(checkpoint["model"])
    stats = RunningMeanStd((22,), 0.0001)
    stats.load_state_dict(checkpoint["obs_stats"])
    for size in (1, 2, 17):
        raw = observations[:size]
        normalized = torch.from_numpy(stats.normalize(raw, 10, 1e-8)).float()
        action, value = session.run(None, {"observations": raw})
        with torch.no_grad():
            np.testing.assert_allclose(action, model.actor(normalized).numpy(), rtol=0, atol=1e-5)
            np.testing.assert_allclose(value, model.value(normalized).numpy(), rtol=0, atol=1e-5)


def test_complete_export_integrity_and_cache_repair(
    tmp_path: Path, pool: TrackPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_export(),
        run_dir=str(tmp_path / "run"),
        evaluation_dir=str(tmp_path / "eval"),
        assets_dir=str(tmp_path / "assets"),
        stochastic_runs=2,
        held_out_runs=1,
        onnx_validation_samples=16,
    )
    directory = Path(config.run_dir) / "snapshots"
    directory.mkdir(parents=True)
    for name in (0, "final"):
        torch.save({**policy(), "requested_step": name}, directory / f"{name}.pt")
    monkeypatch.setattr(artifacts, "make_track_pool", lambda split: pool)
    monkeypatch.setattr(assets, "make_track_pool", lambda split: pool)
    monkeypatch.setattr(recording, "load_env", lambda: EnvConfig(1, 3))
    monkeypatch.setattr(evaluate, "load_env", lambda: EnvConfig(1, 3))
    manifest = assets.build_assets(config)
    root = Path(config.assets_dir)
    assert manifest["versions"] == assets.VERSIONS
    assert len(manifest["snapshots"]) == 2
    for snapshot in manifest["snapshots"]:
        for asset in [
            snapshot["model"],
            *snapshot["replays"].values(),
            *snapshot["heatmaps"].values(),
        ]:
            path = root / asset["path"]
            assert artifacts.sha256(path) == asset["sha256"]
            assert path.stat().st_size == asset["bytes"]
        replay = json.loads((root / snapshot["replays"]["custom"]["path"]).read_text())
        assert len(decode_replay(replay)) == replay["metadata"]["physics_ticks"] + 1 == 7
        heatmap = json.loads((root / snapshot["heatmaps"]["custom"]["path"]).read_text())
        assert heatmap["visit_count"] == 12
    reports = artifacts.run_evaluation(config)
    damaged = Path(config.evaluation_dir) / "0/custom.replay.v1.json"
    damaged.write_text("corrupted")
    repaired = artifacts.evaluate_snapshot(directory / "0.pt", config, pool, pool)
    assert repaired == reports[0]
    assert artifacts.sha256(damaged) == repaired["files"][damaged.name]
    # A failed verification must leave the previously published bundle intact.
    before = (root / "manifest.json").read_bytes()

    def fail(*args: object, **kwargs: object) -> dict:
        raise ValueError("verification failed")

    monkeypatch.setattr(assets, "export_policy", fail)
    with pytest.raises(ValueError, match="verification failed"):
        assets.build_assets(config)
    assert (root / "manifest.json").read_bytes() == before


def test_generated_fixtures_are_current() -> None:
    for name, fixture in fixtures().items():
        assert json.loads((ROOT / "tests/fixtures/parity" / name).read_text()) == json.loads(
            json.dumps(fixture)
        )

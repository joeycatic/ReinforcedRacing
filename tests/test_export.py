import json
import math
from copy import deepcopy
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
    from eval.medals import MedalChoice

    medal_step = 2785280
    medal_hash = json.loads((ROOT / "experiments/medal-selection.json").read_text())["candidates"][
        str(medal_step)
    ]["checkpoint_sha256"]
    original_load = assets.load_checkpoint

    def load_for_export(path: Path) -> dict:
        if path.parent.name == "checkpoints":
            return {**policy(), "global_step": medal_step, "source_sha256": medal_hash}
        return original_load(path)

    monkeypatch.setattr(assets, "load_checkpoint", load_for_export)
    monkeypatch.setattr(
        assets, "load_medals", lambda: [MedalChoice("gold", medal_step, {"custom": None})]
    )
    monkeypatch.setattr(assets, "load_attempts", lambda: {})
    monkeypatch.setattr(
        assets,
        "build_site_data",
        lambda: json.loads((ROOT / "web/public/assets/site.json").read_text()),
    )
    manifest = assets.build_assets(config)
    root = Path(config.assets_dir)
    assert manifest["version"] == "manifest.v3"
    assert manifest["medals"][-1]["medal"] == "author"
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
    medal_cache = Path(config.evaluation_dir) / "medals" / str(medal_step) / "summary.json"
    cached = json.loads(medal_cache.read_text())
    cached["files"] = {}
    medal_cache.write_text(json.dumps(cached))
    assert assets.build_assets(config) == manifest
    assert json.loads(medal_cache.read_text())["files"]
    for asset in manifest["medals"][0]["replays"].values():
        assert artifacts.sha256(root / asset["path"]) == asset["sha256"]
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


def assert_fixture_close(stored: object, fresh: object, path: str = "") -> None:
    # Floats may differ in the last bits across CPUs/NumPy builds; everything else is exact.
    if isinstance(stored, dict) and isinstance(fresh, dict):
        assert stored.keys() == fresh.keys(), path
        for key in stored:
            assert_fixture_close(stored[key], fresh[key], f"{path}.{key}")
    elif isinstance(stored, list) and isinstance(fresh, list):
        assert len(stored) == len(fresh), path
        for i, (a, b) in enumerate(zip(stored, fresh, strict=True)):
            assert_fixture_close(a, b, f"{path}[{i}]")
    elif isinstance(stored, float) or isinstance(fresh, float):
        assert fresh == pytest.approx(stored, rel=1e-9, abs=1e-12), path
    else:
        assert stored == fresh, path


def canonical_probe_events(fixture: dict) -> dict:
    """Defer only roundoff-ambiguous, scheduled exact-lap probe events.

    This is regeneration portability, not Python/TypeScript parity tolerance.
    Validate the complete event state before canonicalizing; the following probe
    is still compared normally, including its interpolated lap timing.
    """
    fixture = deepcopy(fixture)
    probes = fixture["progress_probes"]
    length = fixture["track"]["arc_length"][-1]
    checkpoints = len(fixture["track"]["checkpoints"])
    dt = fixture["config"]["physics"]["dt"]
    fields = ("laps", "lap_times", "lap_started_at", "last_checkpoint", "next_checkpoint")
    for i in range(239, len(probes) - 1, 240):
        previous, current, following = [probes[j]["progress"] for j in (i - 1, i, i + 1)]
        lap = (i + 1) // 240
        boundary = lap * length
        tolerance = 64 * math.ulp(boundary)
        assert abs(current["unwrapped"] - boundary) <= tolerance
        assert min(current["projection"]["arc"], length - current["projection"]["arc"]) <= tolerance
        assert current["delta"] > 0 and not current["wrong_way"]
        assert previous["laps"] == lap - 1
        assert previous["next_checkpoint"] == lap * checkpoints
        expected = {key: previous[key] for key in fields}
        if current["laps"] == lap:
            crossed_at = (i + 1) * dt
            expected.update(
                laps=lap,
                lap_times=previous["lap_times"] + [crossed_at - previous["lap_started_at"]],
                lap_started_at=crossed_at,
                last_checkpoint=boundary,
                next_checkpoint=lap * checkpoints + 1,
            )
        assert_fixture_close(expected, {key: current[key] for key in fields}, f"probe[{i}]")
        assert following["laps"] == lap
        assert following["next_checkpoint"] == lap * checkpoints + 1
        for key in fields:
            current[key] = deepcopy(previous[key])
    return fixture


def seam_fixture_pair() -> tuple[dict, dict]:
    delayed = json.loads((ROOT / "tests/fixtures/parity/ribbon.json").read_text())
    awarded = deepcopy(delayed)
    progress = awarded["progress_probes"][239]["progress"]
    progress.update(
        laps=1,
        next_checkpoint=33,
        last_checkpoint=awarded["track"]["arc_length"][-1],
        lap_times=[4.0],
        lap_started_at=4.0,
    )
    progress["unwrapped"] = math.nextafter(progress["last_checkpoint"], math.inf)
    return delayed, awarded


def test_exact_seam_regeneration_accepts_only_equivalent_event_state() -> None:
    delayed, awarded = seam_fixture_pair()
    assert_fixture_close(canonical_probe_events(delayed), canonical_probe_events(awarded))


@pytest.mark.parametrize(
    "probe,field,value",
    [
        (238, "laps", 1),  # A genuine early event, away from the seam.
        (239, "next_checkpoint", 34),
        (239, "lap_times", [3.9]),
        (239, "lap_started_at", 3.9),
        (239, "last_checkpoint", 775.0),
        (239, "unwrapped", 789.0),
        (240, "laps", 0),  # An event that never converges on the next probe.
        (240, "lap_times", [4.01]),  # A genuinely later crossing.
    ],
)
def test_exact_seam_regeneration_rejects_event_drift(probe: int, field: str, value: object) -> None:
    delayed, awarded = seam_fixture_pair()
    awarded["progress_probes"][probe]["progress"][field] = value
    with pytest.raises(AssertionError):
        assert_fixture_close(canonical_probe_events(delayed), canonical_probe_events(awarded))


def test_exact_seam_regeneration_keeps_unrelated_integers_strict() -> None:
    delayed, awarded = seam_fixture_pair()
    awarded["progress_probes"][239]["progress"]["projection"]["index"] += 1
    with pytest.raises(AssertionError):
        assert_fixture_close(canonical_probe_events(delayed), canonical_probe_events(awarded))


def test_generated_fixtures_are_current() -> None:
    for name, fixture in fixtures().items():
        assert_fixture_close(
            canonical_probe_events(json.loads((ROOT / "tests/fixtures/parity" / name).read_text())),
            canonical_probe_events(json.loads(json.dumps(fixture))),
            name,
        )


def test_site_data_matches_sources() -> None:
    from export.site import build_site_data

    if not (ROOT / "artifacts/training/m3-seed1/metrics.jsonl").exists():
        pytest.skip("Retained training logs are needed to regenerate site data")
    data = build_site_data()
    assert data["stats"]["showcase_completion"] == pytest.approx(98.666, abs=0.01)
    assert data["stats"]["held_out_completion"] == pytest.approx(85.2)
    assert data["stats"]["parity_max_error"] < 1e-6
    assert set(data["curves"]) == {"1", "2"}
    assert data["snapshots"][-1][0] == 6012928


def test_parity_demo_round_trips_fixture() -> None:
    from export.site import build_parity_demo

    fixture = ROOT / "tests/fixtures/parity/sweep.json"
    source = json.loads(fixture.read_text())
    demo = build_parity_demo(fixture)
    assert len(demo["actions"]) == len(demo["python"]) == 600
    assert demo["actions"] == [s["action"] for s in source["steps"]]
    assert demo["python"] == [[s["state"][k] for k in ("x", "y", "psi")] for s in source["steps"]]


def test_committed_medal_bundle_integrity() -> None:
    root = ROOT / "web/public/assets"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["version"] == "manifest.v3"
    medals = manifest["medals"]
    assert medals[-1]["medal"] == "author"
    for medal in medals:
        for asset in [medal["model"], *medal["replays"].values()]:
            path = root / asset["path"]
            assert artifacts.sha256(path) == asset["sha256"]
            assert path.stat().st_size == asset["bytes"]
    for slower, faster in zip(medals, medals[1:], strict=False):
        for track in manifest["tracks"]:
            a, b = slower["lap_times"][track["id"]], faster["lap_times"][track["id"]]
            if a is not None and b is not None:
                assert a >= b * 1.03
    for key in ("site", "parity_demo"):
        assert artifacts.sha256(root / manifest[key]["path"]) == manifest[key]["sha256"]


def test_committed_attempt_bundle_integrity() -> None:
    root = ROOT / "web/public/assets"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["version"] == "manifest.v3"
    snapshots = {s["id"]: s for s in manifest["snapshots"]}
    assert set(manifest["attempts"]) == {t["id"] for t in manifest["tracks"]}
    for track, attempts in manifest["attempts"].items():
        assert 7 <= len(attempts) <= 12
        assert attempts[0]["step"] == 0
        assert attempts[-1]["source"] == "final"
        assert [a["step"] for a in attempts] == sorted(a["step"] for a in attempts)
        for attempt in attempts:
            asset = attempt["replay"]
            path = root / asset["path"]
            assert artifacts.sha256(path) == asset["sha256"]
            assert path.stat().st_size == asset["bytes"]
            meta = json.loads(path.read_text())["metadata"]
            outcome = (
                "finished" if meta["finished"] else "crashed" if meta["crashed"] else "stalled"
            )
            assert meta["finished"] or meta["crashed"] or meta["truncated"]
            assert attempt["outcome"] == outcome
            assert attempt["progress"] == meta["progress"]
            assert attempt["lap_time"] == meta["lap_time"]
            assert attempt["step"] == meta["snapshot_step"]
            if attempt["source"] != "checkpoint":
                assert asset == snapshots[attempt["source"]]["replays"][track]
            else:
                assert meta["snapshot_id"] == f"attempt-{attempt['step']}"


def test_attempt_checkpoint_sha_matches_selection() -> None:
    root = ROOT / "web/public/assets"
    manifest = json.loads((root / "manifest.json").read_text())
    evidence = json.loads((ROOT / "experiments/attempt-selection.json").read_text())
    hashes = {
        (track, run["step"], run["source"]): candidate["checkpoint_sha256"]
        for candidate in evidence["candidates"]
        for track, run in candidate["runs"].items()
    }
    for track, attempts in manifest["attempts"].items():
        for attempt in attempts:
            meta = json.loads((root / attempt["replay"]["path"]).read_text())["metadata"]
            assert meta["checkpoint_sha256"] == hashes[track, attempt["step"], attempt["source"]]


def test_attempt_cache_repair_and_selection_hash_guard(
    tmp_path: Path, pool: TrackPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval.attempts import AttemptChoice

    config = replace(
        load_export(), run_dir=str(tmp_path / "run"), evaluation_dir=str(tmp_path / "eval")
    )
    checkpoint = policy()
    checkpoint["global_step"] = 123
    evidence = tmp_path / "experiments/attempt-selection.json"
    evidence.parent.mkdir()
    evidence.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "checkpoint_sha256": "a" * 64,
                        "runs": {"custom": {"step": 123, "source": "checkpoint"}},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "load_checkpoint", lambda path: checkpoint.copy())
    monkeypatch.setattr(assets, "make_track_pool", lambda split: pool)
    monkeypatch.setattr(
        assets, "load_attempts", lambda: {"custom": [AttemptChoice(123, "checkpoint", None)]}
    )
    monkeypatch.setattr(recording, "load_env", lambda: EnvConfig(1, 3))
    stage = tmp_path / "assets"
    first = assets.export_attempts(stage, {"snapshots": []}, config)
    source = Path(config.evaluation_dir) / "attempts/123/custom.replay.v1.json"
    source.write_text("corrupted")
    assert assets.export_attempts(stage, {"snapshots": []}, config) == first
    summary = source.with_name("summary.json")
    cached = json.loads(summary.read_text())
    cached["files"] = {}
    summary.write_text(json.dumps(cached))
    assert assets.export_attempts(stage, {"snapshots": []}, config) == first
    assert json.loads(summary.read_text())["files"]

    def unexpected_record(*args: object, **kwargs: object) -> None:
        pytest.fail("Valid caches should not record again")

    monkeypatch.setattr(assets, "record_runs", unexpected_record)
    assert assets.export_attempts(stage, {"snapshots": []}, config) == first
    checkpoint["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="changed since selection"):
        assets.export_attempts(stage, {"snapshots": []}, config)


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "b" * 64), ("step", 124)])
def test_attempt_snapshot_selection_guard(
    tmp_path: Path, pool: TrackPool, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    from eval.attempts import AttemptChoice

    evidence = tmp_path / "experiments/attempt-selection.json"
    evidence.parent.mkdir()
    evidence.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "checkpoint_sha256": "a" * 64,
                        "runs": {"custom": {"step": 123, "source": "final"}},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "make_track_pool", lambda split: pool)
    monkeypatch.setattr(
        assets, "load_attempts", lambda: {"custom": [AttemptChoice(123, "final", None)]}
    )
    snapshot = {"id": "final", "checkpoint_sha256": "a" * 64, "step": 123, field: value}
    with pytest.raises(ValueError, match="snapshot changed since selection"):
        assets.export_attempts(tmp_path / "assets", {"snapshots": [snapshot]}, load_export())


def test_generated_editor_fixtures_are_current() -> None:
    from tools.generate_parity import editor_fixtures

    for name, fixture in editor_fixtures().items():
        assert_fixture_close(
            json.loads((ROOT / "tests/fixtures/editor" / name).read_text()),
            json.loads(json.dumps(fixture)),
            name,
        )


def test_generated_crossing_fixture_is_current() -> None:
    from tools.generate_parity import crossing_fixture

    assert_fixture_close(
        json.loads((ROOT / "tests/fixtures/crossing/figure8.json").read_text()),
        json.loads(json.dumps(crossing_fixture())),
        "figure8",
    )

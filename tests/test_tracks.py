from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.config import PhysicsConfig, TrackConfig, load_tracks
from sim.geometry import polylines_intersect
from sim.track import Track, build_track, generate_track, showcase_track, validity_errors


@pytest.mark.parametrize("seed", range(12))
def test_seeded_tracks_are_valid(seed: int, physics: PhysicsConfig, tracks: TrackConfig) -> None:
    track = generate_track(seed, tracks, physics)
    assert not validity_errors(track, tracks.max_curvature)
    assert 10 <= track.width <= 14
    assert track.centerline.dtype == np.float64
    lengths = np.diff(track.arc_length)
    assert np.all(lengths > 0)
    np.testing.assert_allclose(lengths[:-1], 1, atol=0.01)
    assert lengths[-1] <= 1.01
    np.testing.assert_allclose(np.linalg.norm(track.tangents, axis=1), 1)
    np.testing.assert_allclose(np.linalg.norm(track.left - track.right, axis=1), track.width)
    np.testing.assert_allclose(np.diff(track.checkpoints), 25)


def test_tracks_reproduce_exactly_and_differ_by_seed(
    physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    a = generate_track(42, tracks, physics)
    b = generate_track(42, tracks, physics)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != generate_track(43, tracks, physics).to_dict()


def test_showcases(physics: PhysicsConfig, tracks: TrackConfig) -> None:
    showcase = load_tracks()[1]
    assert len(showcase) == 3
    generated = [showcase_track(item, tracks, physics) for item in showcase]
    assert len({item.seed for item in generated}) == 3
    for track in generated:
        assert not validity_errors(track, tracks.max_curvature)


def test_json_roundtrip_preserves_exact_geometry(circle: Track, tmp_path: Path) -> None:
    path = tmp_path / "circle.track.v1.json"
    circle.save(path)
    assert Track.load(path).to_dict() == circle.to_dict()
    for arc in (-1, 0, 7.5, circle.length, circle.length + 7.5):
        point, heading = circle.sample(arc)
        assert np.isfinite(point).all() and np.isfinite(heading)
    np.testing.assert_array_equal(circle.sample(0)[0], circle.sample(circle.length)[0])


@pytest.mark.parametrize("field", ["version", "arc_length", "left", "checkpoints", "centerline"])
def test_json_rejects_invalid_or_inconsistent_geometry(circle: Track, field: str) -> None:
    data = circle.to_dict()
    if field == "version":
        data[field] = "track.v99"
    elif field == "arc_length":
        data[field][1] += 1
    elif field == "checkpoints":
        data[field][0] = 25
    else:
        data[field][0][0] = float("nan")
    with pytest.raises(ValueError):
        Track.from_dict(data)


def test_rejects_crossing_and_overlapping_geometry() -> None:
    bowtie = np.array([[0, 0], [40, 40], [0, 40], [40, 0]], dtype=np.float64)
    assert "centerline self-intersects" in validity_errors(build_track(bowtie, 12, 25))
    # A narrow U-shaped channel has offset boundaries crossing each other.
    points = np.array([[0, 0], [50, 0], [50, 30], [30, 30], [30, 5], [20, 5], [20, 30], [0, 30]])
    assert validity_errors(build_track(points, 16, 25))
    assert polylines_intersect(np.array([[0, 0], [3, 0], [1, 0], [1, 2], [0, 2]]))


def test_rejects_excessive_curvature_and_generation_exhaustion(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    assert "curvature exceeds driveable limit" in validity_errors(circle, 0.001)
    impossible = replace(tracks, max_curvature=0.000001, max_attempts=2)
    with pytest.raises(ValueError, match="Unable to generate"):
        generate_track(42, impossible, physics)
    with pytest.raises(ValueError, match="zero-length"):
        build_track(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]), 12, 25)

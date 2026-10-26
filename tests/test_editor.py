"""Editor geometry and crossing interchange contracts."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from sim.config import ROOT, load_editor
from sim.editor import build_editor_track, editor_issues, tight_sections
from sim.geometry import polyline_intersections
from sim.pool import make_track_pool
from sim.track import Track
from tools.editor_shapes import shapes


def test_shapes_report_expected_issues() -> None:
    cfg = load_editor()
    for name, (control, expected) in shapes().items():
        track = build_editor_track(control, cfg)
        assert editor_issues(track, cfg) == expected, name
        if name == "tight":
            assert tight_sections(track, cfg.max_curvature)
            assert np.max(np.abs(track.curvature)) < 1 / 6


def test_figure8_has_one_right_angle_crossing() -> None:
    track = build_editor_track(shapes()["figure8"][0], load_editor())
    assert len(track.crossings) == 1
    c = track.crossings[0]
    assert c.angle == pytest.approx(90, abs=2)
    assert min(c.arcs[1] - c.arcs[0], track.length - c.arcs[1] + c.arcs[0]) >= 80
    assert c.over == 1


@pytest.mark.parametrize("name", ["figure8", "pretzel"])
def test_hidden_zones_cover_all_crossing_boundary_hits(name: str) -> None:
    track = build_editor_track(shapes()[name][0], load_editor())
    n = len(track.centerline)
    for a, b, offset_a, offset_b in [
        (track.left, None, 0, 0),
        (track.right, None, n, n),
        (track.left, track.right, 0, n),
    ]:
        for i, j, _ in polyline_intersections(a, b):
            assert any(
                (i + offset_a in c.hidden[0] and j + offset_b in c.hidden[1])
                or (i + offset_a in c.hidden[1] and j + offset_b in c.hidden[0])
                for c in track.crossings
            )
    assert len(track.crossings) == (1 if name == "figure8" else 2)


@pytest.mark.parametrize("name", ["loop", "figure8"])
def test_track_length_and_width_in_training_range(name: str) -> None:
    track = build_editor_track(shapes()[name][0], load_editor())
    assert abs(track.length - 700) <= 35
    assert track.width == 12


def test_v1_round_trip_unchanged() -> None:
    for track in make_track_pool("showcase").tracks:
        assert track.to_dict()["version"] == "track.v1"
        assert "crossings" not in track.to_dict()
        assert Track.from_dict(track.to_dict()).to_dict() == track.to_dict()


def test_v2_round_trip() -> None:
    track = build_editor_track(shapes()["figure8"][0], load_editor())
    assert Track.from_dict(track.to_dict()).crossings == track.crossings
    data = track.to_dict()
    data["crossings"][0]["hidden"][0].pop()
    with pytest.raises(ValueError):
        Track.from_dict(data)


def test_editor_yaml_matches_web_config() -> None:
    data = json.loads((ROOT / "web/src/editor/editor-config.json").read_text())
    assert {k: data[k] for k in asdict(load_editor())} == asdict(load_editor())


def crossing_observation(track: Track, arc: float) -> np.ndarray:
    from sim.config import load_observation
    from sim.observation import ObservationBuilder
    from sim.pool import TrackPool

    position, heading = track.sample(arc)
    index = int(np.searchsorted(track.arc_length, arc, side="right")) - 1
    state = np.array([[*position, heading, 20, 0, 0, 0]])
    return ObservationBuilder(TrackPool([track]), load_observation()).build(
        state, np.array([0]), np.array([index]), np.array([arc]), np.zeros(1), np.zeros((1, 2))
    )[0]


def test_crossing_rays_see_no_phantom_wall() -> None:
    from sim.config import load_observation

    track = build_editor_track(shapes()["figure8"][0], load_editor())
    for arc in track.crossings[0].arcs:
        rays = crossing_observation(track, arc)[:9] * load_observation().ray_max_distance
        assert rays[4] >= 25
        assert rays[[1, 7]] == pytest.approx([6, 6], abs=0.3)


def test_crossing_progress_stays_on_branch() -> None:
    from sim.config import load_tracks
    from sim.progress import ProgressTracker, project

    track = build_editor_track(shapes()["figure8"][0], load_editor())
    for center in track.crossings[0].arcs:
        tracker = ProgressTracker(track, load_tracks()[0])
        start = center - 40
        tracker.projection = project(track, *track.sample(start)[0], None, 24)
        previous = start - 1
        for tick, arc in enumerate(np.arange(start, center + 41, 1)):
            tracker.update(*track.sample(arc)[0], tick / 60, 1 / 60)
            assert tracker.projection.distance < 0.01
            assert tracker.projection.arc > previous
            previous = tracker.projection.arc


def test_over_does_not_change_observations() -> None:
    tracks = [build_editor_track(shapes()["figure8"][0], load_editor(), (v,)) for v in (0, 1)]
    for arc in tracks[0].crossings[0].arcs:
        np.testing.assert_array_equal(
            crossing_observation(tracks[0], arc), crossing_observation(tracks[1], arc)
        )


def test_pool_without_crossings_hides_nothing() -> None:
    pool = make_track_pool("showcase")
    assert pool.hidden_tags(np.array([0]), np.array([0.0])) is None
    assert pool.grid.tag_count == 0


def test_fold_does_not_suppress_remote_overlap() -> None:
    from dataclasses import replace

    track = build_editor_track(shapes()["overlap"][0], load_editor())
    curvature = track.curvature.copy()
    curvature[0] = 0.2
    assert editor_issues(replace(track, curvature=curvature), load_editor()) == ["fold", "overlap"]


def test_intersections_match_touching_collinear_and_adjacency_rules() -> None:
    from sim.geometry import polylines_intersect

    a = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert polyline_intersections(a) == []
    for b in (a + [2, 0], a + [4, 4], a + [5, 0]):
        assert bool(polyline_intersections(a, b)) == polylines_intersect(a, b)
    hit = next(p for i, j, p in polyline_intersections(a, a + [2, 0]) if i == j == 0)
    np.testing.assert_array_equal(hit, [3, 0])


def test_mixed_pool_tags_match_single_track_observations() -> None:
    from sim.config import load_observation
    from sim.observation import ObservationBuilder
    from sim.pool import TrackPool

    track = build_editor_track(shapes()["figure8"][0], load_editor())
    pool = TrackPool([make_track_pool("showcase").tracks[0], track, track])
    assert pool.grid.tag_count == 4
    for track_id in (1, 2):
        for arc in track.crossings[0].arcs:
            position, heading = track.sample(arc)
            index = int(np.searchsorted(track.arc_length, arc, side="right")) - 1
            observation = ObservationBuilder(pool, load_observation()).build(
                np.array([[*position, heading, 20, 0, 0, 0]]),
                np.array([track_id]),
                np.array([index]),
                np.array([arc]),
                np.zeros(1),
                np.zeros((1, 2)),
            )[0]
            np.testing.assert_array_equal(observation, crossing_observation(track, arc))

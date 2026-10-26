import math

import pytest

from sim.config import TrackConfig
from sim.progress import ProgressTracker, project
from sim.track import Track


def test_two_laps_and_interpolated_lap_timing(circle: Track, tracks: TrackConfig) -> None:
    tracker = ProgressTracker(circle, tracks)
    dt = 1 / 60
    speed = 30.0
    for tick in range(1, math.ceil(2 * circle.length / (speed * dt)) + 1):
        point, _ = circle.sample(tick * speed * dt)
        tracker.update(*point, tick * dt, dt)
    assert tracker.laps == 2
    assert tracker.lap_times == pytest.approx([circle.length / speed] * 2, abs=1e-9)
    assert tracker.unwrapped == pytest.approx(tick * speed * dt)
    assert tracker.last_checkpoint == pytest.approx(2 * circle.length)


def test_reverse_crossing_start_and_jitter_do_not_award_laps(
    circle: Track, tracks: TrackConfig
) -> None:
    tracker = ProgressTracker(circle, tracks)
    for tick, arc in enumerate([-1.0, -2.0, -0.1, 0.1, -0.1, 0.1], start=1):
        point, _ = circle.sample(arc)
        tracker.update(*point, tick / 60, 1 / 60)
        if tick <= 2:
            assert tracker.wrong_way
    assert tracker.laps == 0
    assert tracker.lap_times == []
    assert tracker.unwrapped == pytest.approx(0.1)
    assert tracker.last_checkpoint == 0


def test_respawn_to_last_checkpoint_preserves_timer_and_laps(
    circle: Track, tracks: TrackConfig
) -> None:
    tracker = ProgressTracker(circle, tracks)
    for tick in range(1, 64):
        point, _ = circle.sample(tick)
        tracker.update(*point, tick / 60, 1 / 60)
    assert tracker.last_checkpoint == 50
    x, y, _ = tracker.respawn()
    assert tracker.unwrapped == 50
    assert tracker.delta == 0
    assert tracker.lap_started_at == 0
    tracker.update(x, y, 2, 1 / 60)
    assert tracker.delta == pytest.approx(0)
    assert tracker.laps == 0


def test_projection_signed_offset_and_bounded_search(circle: Track, tracks: TrackConfig) -> None:
    outside = project(circle, 63, 0, 0, tracks.projection_window)
    inside = project(circle, 57, 0, 0, tracks.projection_window)
    assert outside.offset == pytest.approx(-3, abs=0.01)
    assert inside.offset == pytest.approx(3, abs=0.01)
    assert outside.distance == pytest.approx(3)
    # The nearest global point is halfway around the track, outside this search window.
    local = project(circle, -60, 0, 0, 2)
    assert local.index in {398, 399, 0, 1, 2}
    global_projection = project(circle, -60, 0, None, 2)
    assert global_projection.distance == pytest.approx(0, abs=1e-10)

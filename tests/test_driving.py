"""End-to-end M1 checks: drive real generated geometry and render a pygame frame."""

import math
from pathlib import Path

import pytest

from sim.config import PhysicsConfig, TrackConfig, load_tracks
from sim.physics import Action
from sim.session import DriveSession
from sim.track import showcase_track
from tools.debug_drive import run


@pytest.mark.parametrize("showcase_index", range(3))
def test_showcase_is_driveable_for_a_full_lap(
    physics: PhysicsConfig, tracks: TrackConfig, showcase_index: int
) -> None:
    # A conservative pure-pursuit test driver, not a learning agent or delivered game feature.
    item = load_tracks()[1][showcase_index]
    track = showcase_track(item, tracks, physics)
    session = DriveSession(track, physics, tracks, mode="training")
    for _ in range(6000):
        if session.ticks % physics.action_repeat == 0:
            state = session.state
            target, _ = track.sample(session.progress.projection.arc + 10)
            error = math.atan2(target[1] - state.y, target[0] - state.x) - state.psi
            angle = math.atan2(2 * (physics.front_axle + physics.rear_axle) * math.sin(error), 10)
            steer = angle * (1 + state.speed / physics.steer_speed_falloff) / physics.max_steer
            action = Action(steer, max(-1, min(1, (12 - state.vx) * 0.5)))
        session.tick(action)
        assert not session.terminated
        if session.progress.laps:
            break
    assert session.progress.laps == 1
    assert session.progress.lap_times[0] == pytest.approx(session.time, abs=physics.dt)


def test_headless_pygame_smoke(
    physics: PhysicsConfig, tracks: TrackConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    item = load_tracks()[1][0]
    track = showcase_track(item, tracks, physics)
    screenshot = tmp_path / "drive.png"
    session = run(track, physics, headless=True, steps=600, screenshot=screenshot)
    assert session.ticks == 600
    assert session.crashes > 0  # Scripted steering also exercises the crash/respawn display.
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

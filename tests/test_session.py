from dataclasses import replace
from math import ceil

import pytest

from sim.config import PhysicsConfig, TrackConfig
from sim.physics import Action
from sim.session import DriveSession
from sim.track import Track


def crash(session: DriveSession) -> None:
    session.state = replace(session.state, x=session.state.x + session.track.width)
    session.tick(Action())
    assert session.crashed


def test_play_respawns_after_half_second(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    session = DriveSession(circle, physics, tracks)
    start = session.state
    crash(session)
    delay = ceil(0.5 / physics.dt)
    crashed = session.state
    for _ in range(delay - 1):
        session.tick(Action(1, 1))
        assert session.state == crashed
        assert session.crashed
    session.tick(Action(1, 1))
    assert session.state == start
    assert not session.crashed
    assert session.crashes == 1
    assert session.lap_time == pytest.approx((delay + 1) * physics.dt)


def test_instant_respawn_and_restart(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    session = DriveSession(circle, physics, tracks)
    crash(session)
    session.respawn()
    assert session.respawn_ticks == 0
    assert session.state.speed == session.state.r == 0
    assert session.time > 0
    session.restart()
    assert session.time == session.lap_time == session.crashes == 0


def test_training_crash_terminates_without_autoreset(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    session = DriveSession(circle, physics, tracks, mode="training")
    crash(session)
    assert session.terminated
    state, ticks = session.state, session.ticks
    for _ in range(60):
        session.tick(Action(0, 1))
    assert session.state == state and session.ticks == ticks


def test_identical_actions_reproduce_session(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    a = DriveSession(circle, physics, tracks)
    b = DriveSession(circle, physics, tracks)
    for tick in range(600):
        action = Action(0.2 if tick < 200 else -0.3, 1)
        a.tick(action)
        b.tick(action)
        assert a.state == b.state
        assert a.progress.unwrapped == b.progress.unwrapped
        assert a.time == b.time

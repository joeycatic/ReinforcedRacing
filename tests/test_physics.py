from dataclasses import astuple, replace
from math import hypot, isfinite

import pytest

from sim.config import PhysicsConfig
from sim.physics import Action, CarState, step, tire_forces


def test_acceleration_and_top_speed_match_spec(physics: PhysicsConfig) -> None:
    state = CarState()
    acceleration_time = None
    for tick in range(60 * 60):
        state = step(state, Action(0, 1), physics)
        if acceleration_time is None and state.vx >= 100 / 3.6:
            acceleration_time = (tick + 1) * physics.dt
    assert acceleration_time is not None
    assert 2.5 <= acceleration_time <= 3.5
    assert 60 <= state.vx <= 80
    assert state.vy == state.r == state.psi == 0


def test_steering_at_rest_does_not_create_motion(physics: PhysicsConfig) -> None:
    state = CarState()
    for _ in range(600):
        state = step(state, Action(1, 0), physics)
    assert state.x == state.y == state.psi == state.vx == state.vy == state.r == 0


def test_braking_then_limited_reverse(physics: PhysicsConfig) -> None:
    state = CarState(vx=100 / 3.6)
    for _ in range(60):
        state = step(state, Action(0, -1), physics)
    assert 0 < state.vx < 15
    for _ in range(600):
        state = step(state, Action(0, -1), physics)
    assert state.vx == -physics.max_reverse_speed
    assert state.vy == state.r == 0
    for _ in range(120):
        state = step(state, Action(0, 1), physics)
    assert state.vx > 0


def test_steering_rate_limit_and_action_clipping(physics: PhysicsConfig) -> None:
    first = step(CarState(vx=30), Action(1, 1), physics)
    assert first.steering == pytest.approx(physics.steer_rate * physics.dt)
    assert step(CarState(vx=30), Action(500, 500), physics) == first
    state = first
    for _ in range(120):
        state = step(state, Action(1, 0), physics)
    assert state.steering == 1
    assert step(state, Action(-1, 0), physics).steering == pytest.approx(
        1 - physics.steer_rate * physics.dt
    )


def test_linear_tires_saturate_and_rear_has_less_grip(physics: PhysicsConfig) -> None:
    linear = tire_forces(CarState(vx=30, vy=0.01), 0, physics)
    drift = tire_forces(CarState(vx=30, vy=15), 0, physics)
    assert abs(linear.front) < linear.front_limit
    assert abs(linear.rear) < linear.rear_limit
    assert drift.front == -drift.front_limit
    assert drift.rear == -drift.rear_limit
    assert physics.rear_mu < physics.front_mu


def test_drift_loses_speed_and_recovers(physics: PhysicsConfig) -> None:
    # Equal entry speed: lateral velocity must not give the drift extra energy.
    clean = CarState(vx=hypot(30, 10))
    drifting = CarState(vx=30, vy=10, r=0.4)
    for _ in range(180):
        clean = step(clean, Action(0, 0), physics)
        drifting = step(drifting, Action(0, 0), physics)
    assert abs(drifting.vy) < 0.05
    assert abs(drifting.r) < 0.05
    assert drifting.vx < clean.vx


def test_corner_induced_drift_is_recoverable_with_countersteer(physics: PhysicsConfig) -> None:
    state = CarState(vx=35)
    for _ in range(30):
        state = step(state, Action(0.3, 0), physics)
    assert state.vy < -3  # Excess cornering demand breaks rear traction.
    for _ in range(180):
        countersteer = 3 * state.vy / max(abs(state.vx), 5) - 0.3 * state.r
        state = step(state, Action(countersteer, 0), physics)
    assert state.vx > 20
    assert abs(state.vy) < 0.1 and abs(state.r) < 0.1


@pytest.mark.parametrize("direction", [-1, 1])
def test_turning_and_reverse_steering(physics: PhysicsConfig, direction: int) -> None:
    state = CarState(vx=direction * 4.0)
    for _ in range(20):
        state = step(state, Action(0.5, 0), physics)
    assert direction * state.r > 0


def test_deterministic_stress_and_symmetry(physics: PhysicsConfig) -> None:
    left, right = CarState(vx=50), CarState(vx=50)
    for tick in range(3600):
        steer = 1.0 if tick % 240 < 120 else -1.0
        throttle = 1.0 if tick % 600 < 400 else -1.0
        left = step(left, Action(steer, throttle), physics)
        right = step(right, Action(-steer, throttle), physics)
        assert all(isfinite(value) for value in astuple(left))
        assert abs(left.vx) < 100 and abs(left.vy) < 100 and abs(left.r) < 20
    assert left.x == right.x
    assert left.y == -right.y
    assert left.psi == -right.psi
    assert left.vx == right.vx


def test_coasting_stops_without_oscillating(physics: PhysicsConfig) -> None:
    state = CarState(vx=0.01)
    for _ in range(60):
        state = step(state, Action(), physics)
    assert state.vx == 0


def test_invalid_configuration_and_actions(physics: PhysicsConfig) -> None:
    with pytest.raises(ValueError):
        replace(physics, mass=0)
    with pytest.raises(ValueError):
        replace(physics, dt=float("nan"))
    with pytest.raises(ValueError):
        replace(physics, action_repeat=2.5)
    with pytest.raises(ValueError):
        Action(float("nan"), 0)

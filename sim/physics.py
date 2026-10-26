"""Single-track dynamic bicycle model, float64 and semi-implicit Euler.

Body x is forward; body y and positive steering are left. Tire lateral forces
are linear in slip then clamped independently at each axle's friction limit.
At very low speed, regularized wheel-frame slip gives zero force at rest.
State includes the rate-limited steering actuator, which must be serialized
alongside the six rigid-body coordinates for deterministic continuation.
"""

from dataclasses import dataclass
from math import atan2, copysign, cos, hypot, isfinite, sin

from sim.config import PhysicsConfig


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


@dataclass(frozen=True)
class Action:
    steer: float = 0.0
    throttle: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.steer) or not isfinite(self.throttle):
            raise ValueError("Actions must be finite")


@dataclass(frozen=True)
class CarState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    steering: float = 0.0

    @property
    def speed(self) -> float:
        return hypot(self.vx, self.vy)


@dataclass(frozen=True)
class TireForces:
    front: float
    rear: float
    front_limit: float
    rear_limit: float


def tire_forces(state: CarState, steering: float, config: PhysicsConfig) -> TireForces:
    c = config
    wheelbase = c.front_axle + c.rear_axle
    front_lateral = state.vy + c.front_axle * state.r
    front_wheel_x = state.vx * cos(steering) + front_lateral * sin(steering)
    front_wheel_y = front_lateral * cos(steering) - state.vx * sin(steering)
    front_slip = atan2(front_wheel_y, max(abs(front_wheel_x), c.slip_speed_floor))
    rear_slip = atan2(state.vy - c.rear_axle * state.r, max(abs(state.vx), c.slip_speed_floor))
    front_limit = c.front_mu * c.mass * c.gravity * c.rear_axle / wheelbase
    rear_limit = c.rear_mu * c.mass * c.gravity * c.front_axle / wheelbase
    return TireForces(
        clamp(-c.front_stiffness * front_slip, -front_limit, front_limit),
        clamp(-c.rear_stiffness * rear_slip, -rear_limit, rear_limit),
        front_limit,
        rear_limit,
    )


def step(state: CarState, action: Action, config: PhysicsConfig) -> CarState:
    """Advance exactly one fixed physics tick; never reads a clock or RNG."""
    c = config
    requested = clamp(action.steer, -1.0, 1.0)
    steering = state.steering + clamp(
        requested - state.steering, -c.steer_rate * c.dt, c.steer_rate * c.dt
    )
    angle = steering * c.max_steer / (1.0 + state.speed / c.steer_speed_falloff)
    tire = tire_forces(state, angle, c)
    throttle = clamp(action.throttle, -1.0, 1.0)
    braking = throttle < 0.0 and state.vx > c.reverse_threshold
    if throttle >= 0.0:
        drive = throttle * c.engine_force / (1.0 + max(state.vx, 0.0) / c.engine_speed_falloff)
    elif braking:
        drive = throttle * c.brake_force
    else:
        drive = throttle * c.reverse_force
    resistance = c.drag * state.speed * state.vx
    if state.vx != 0.0:
        resistance += copysign(c.rolling_resistance, state.vx)
    force_x = drive - resistance - tire.front * sin(angle)
    force_y = tire.front * cos(angle) + tire.rear
    vx = state.vx + (force_x / c.mass + state.r * state.vy) * c.dt
    vy = state.vy + (force_y / c.mass - state.r * state.vx) * c.dt
    r = (
        state.r
        + (c.front_axle * tire.front * cos(angle) - c.rear_axle * tire.rear) / c.inertia * c.dt
    )
    # Passive resistance/braking cannot flip the longitudinal velocity in a tick.
    if state.vx > 0.0 and vx < 0.0 and (braking or throttle == 0.0):
        vx = 0.0
    elif state.vx < 0.0 and vx > 0.0 and throttle == 0.0:
        vx = 0.0
    vx = max(vx, -c.max_reverse_speed)
    psi = atan2(sin(state.psi + r * c.dt), cos(state.psi + r * c.dt))
    return CarState(
        state.x + (vx * cos(psi) - vy * sin(psi)) * c.dt,
        state.y + (vx * sin(psi) + vy * cos(psi)) * c.dt,
        psi,
        vx,
        vy,
        r,
        steering,
    )

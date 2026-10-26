"""Typed, validated YAML configuration; no physics tuning in simulation logic."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float
    action_repeat: int
    mass: float
    inertia: float
    front_axle: float
    rear_axle: float
    car_length: float
    car_width: float
    gravity: float
    front_stiffness: float
    rear_stiffness: float
    front_mu: float
    rear_mu: float
    max_steer: float
    steer_speed_falloff: float
    steer_rate: float
    slip_speed_floor: float
    engine_force: float
    engine_speed_falloff: float
    brake_force: float
    drag: float
    rolling_resistance: float
    reverse_force: float
    reverse_threshold: float
    max_reverse_speed: float
    respawn_delay: float

    def __post_init__(self) -> None:
        _positive_fields(self)
        if not 0 < self.max_steer < 1.57:
            raise ValueError("max_steer must be in (0, pi/2)")
        if self.reverse_threshold >= self.max_reverse_speed:
            raise ValueError("reverse_threshold must be below max_reverse_speed")


@dataclass(frozen=True)
class TrackConfig:
    min_control_points: int
    max_control_points: int
    radius: float
    radius_noise: float
    angle_jitter: float
    aspect_ratio: float
    min_width: float
    max_width: float
    spline_samples_per_span: int
    sample_spacing: float
    checkpoint_spacing: float
    max_curvature: float
    max_attempts: int
    projection_window: int
    wrong_way_min_delta: float

    def __post_init__(self) -> None:
        _positive_fields(self)
        if not 4 <= self.min_control_points <= self.max_control_points:
            raise ValueError("control point range must be ordered and start at >= 4")
        if self.min_width > self.max_width:
            raise ValueError("width range is reversed")
        if self.radius_noise >= 1 or self.angle_jitter >= 0.5:
            raise ValueError("radius_noise < 1 and angle_jitter < 0.5 required")
        if self.spline_samples_per_span < 4:
            raise ValueError("spline_samples_per_span must be >= 4")


@dataclass(frozen=True)
class ShowcaseConfig:
    id: str
    name: str
    seed: int
    radius: float
    radius_noise: float
    aspect_ratio: float


def _positive_fields(config: PhysicsConfig | TrackConfig) -> None:
    nonnegative = {"radius_noise", "angle_jitter", "drag", "rolling_resistance"}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
            raise ValueError(f"{field.name} must be a finite number")
        if field.type in (int, "int") and not isinstance(value, int):
            raise ValueError(f"{field.name} must be an integer")
        if value < 0 or (value == 0 and field.name not in nonnegative):
            raise ValueError(f"{field.name} must be positive (or a supported zero-valued tuning)")


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_physics(path: Path = ROOT / "configs/physics.yaml") -> PhysicsConfig:
    return PhysicsConfig(**_read_yaml(path))


def load_tracks(
    path: Path = ROOT / "configs/tracks.yaml",
) -> tuple[TrackConfig, tuple[ShowcaseConfig, ...]]:
    data = _read_yaml(path)
    config = TrackConfig(**data["generation"])
    showcase = tuple(ShowcaseConfig(**item) for item in data["showcase"])
    if len({item.id for item in showcase}) != len(showcase):
        raise ValueError("Showcase track ids must be unique")
    return config, showcase

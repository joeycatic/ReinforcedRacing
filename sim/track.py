"""Seeded closed tracks and strict, versioned JSON interchange."""

import json
from dataclasses import dataclass, replace
from math import atan2, isfinite, tan
from pathlib import Path
from typing import Any

import numpy as np

from sim import TRACK_V2, TRACK_VERSION
from sim.config import PhysicsConfig, ShowcaseConfig, TrackConfig
from sim.geometry import FloatArray, catmull_rom, cross, polylines_intersect, resample_closed


@dataclass(frozen=True)
class Crossing:
    arcs: tuple[float, float]
    point: tuple[float, float]
    angle: float
    over: int
    hidden: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Track:
    id: str
    name: str
    seed: int
    width: float
    centerline: FloatArray
    arc_length: FloatArray  # N+1 entries, including the closing segment's end.
    tangents: FloatArray
    curvature: FloatArray
    left: FloatArray
    right: FloatArray
    checkpoints: FloatArray  # Arc lengths, starting with start/finish at zero.

    crossings: tuple[Crossing, ...] = ()

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def sample(self, arc: float) -> tuple[FloatArray, float]:
        """Interpolate a position and segment heading at a wrapped arc length."""
        s = arc % self.length
        index = min(int(np.searchsorted(self.arc_length, s, side="right")) - 1, len(self.left) - 1)
        end = (index + 1) % len(self.centerline)
        fraction = (s - self.arc_length[index]) / (
            self.arc_length[index + 1] - self.arc_length[index]
        )
        direction = self.centerline[end] - self.centerline[index]
        return self.centerline[index] + fraction * direction, atan2(direction[1], direction[0])

    def to_dict(self) -> dict[str, Any]:
        result = {
            "version": TRACK_V2 if self.crossings else TRACK_VERSION,
            "id": self.id,
            "name": self.name,
            "seed": self.seed,
            "width": self.width,
            "centerline": self.centerline.tolist(),
            "arc_length": self.arc_length.tolist(),
            "tangents": self.tangents.tolist(),
            "curvature": self.curvature.tolist(),
            "left": self.left.tolist(),
            "right": self.right.tolist(),
            "checkpoints": self.checkpoints.tolist(),
        }

        if self.crossings:
            result["crossings"] = [
                {
                    "arcs": list(c.arcs),
                    "point": list(c.point),
                    "angle": c.angle,
                    "over": c.over,
                    "hidden": [list(ids) for ids in c.hidden],
                }
                for c in self.crossings
            ]
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Track":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        if data.get("version") not in (TRACK_VERSION, TRACK_V2):
            raise ValueError(f"Unsupported track version: {data.get('version')}")
        if not isinstance(data.get("seed"), int) or not isinstance(data.get("id"), str):
            raise ValueError("Track requires an integer seed and string id")
        checkpoints = np.asarray(data["checkpoints"], dtype=np.float64)
        if checkpoints.ndim != 1 or len(checkpoints) < 2 or checkpoints[0] != 0:
            raise ValueError("Invalid checkpoint arcs")
        track = build_track(
            np.asarray(data["centerline"], dtype=np.float64),
            float(data["width"]),
            float(checkpoints[1]),
            seed=data["seed"],
            track_id=data["id"],
            name=data["name"],
        )
        for key in ("arc_length", "tangents", "curvature", "left", "right", "checkpoints"):
            supplied = np.asarray(data[key], dtype=np.float64)
            expected = getattr(track, key)
            if supplied.shape != expected.shape or not np.allclose(
                supplied, expected, rtol=0, atol=1e-9
            ):
                raise ValueError(f"Inconsistent track geometry: {key}")
            # Preserve the exact exported numbers, rather than rounded/recomputed geometry.
            object.__setattr__(track, key, supplied)
        if data["version"] == TRACK_V2:
            from sim.config import load_editor
            from sim.crossings import find_crossings
            from sim.editor import editor_issues

            if (
                track.id != "custom"
                or not isinstance(data.get("crossings"), list)
                or not data["crossings"]
            ):
                raise ValueError("track.v2 requires custom track crossings")
            config = load_editor()
            try:
                crossings = find_crossings(track, config, [c["over"] for c in data["crossings"]])
                object.__setattr__(track, "crossings", crossings)
                expected_crossings = track.to_dict()["crossings"]
                if len(expected_crossings) != len(data["crossings"]):
                    raise ValueError("Inconsistent crossings")
                for supplied, expected in zip(data["crossings"], expected_crossings, strict=True):
                    if set(supplied) != set(expected):
                        raise ValueError("Inconsistent crossing fields")
                    for key in ("arcs", "point", "angle"):
                        a, b = np.asarray(supplied[key]), np.asarray(expected[key])
                        if a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=0):
                            raise ValueError("Inconsistent crossing geometry")
                    if supplied["hidden"] != expected["hidden"]:
                        raise ValueError("Inconsistent crossing hidden segments")
            except (KeyError, TypeError, IndexError) as error:
                raise ValueError("Invalid crossing data") from error
            errors = editor_issues(track, config)
        else:
            if "crossings" in data:
                raise ValueError("track.v1 cannot contain crossings")
            errors = validity_errors(track)
        if errors:
            raise ValueError("Invalid track: " + "; ".join(errors))
        return track


def build_track(
    centerline: FloatArray,
    width: float,
    checkpoint_spacing: float,
    *,
    seed: int = 0,
    track_id: str = "custom",
    name: str = "Custom track",
) -> Track:
    points = np.array(centerline, dtype=np.float64, copy=True)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("Centerline must contain at least four 2D points")
    if not np.isfinite(points).all() or not isfinite(width) or width <= 0:
        raise ValueError("Track geometry must be finite with positive width")
    if not isfinite(checkpoint_spacing) or checkpoint_spacing <= 0:
        raise ValueError("checkpoint_spacing must be finite and positive")
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Track contains a zero-length segment (omit repeated closing point)")
    segments = vectors / lengths[:, None]
    previous = np.roll(segments, 1, axis=0)
    tangents = previous + segments
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_lengths < 1e-8):
        raise ValueError("Track has a reversing cusp")
    tangents /= tangent_lengths[:, None]
    turns = np.arctan2(cross(previous, segments), np.sum(previous * segments, axis=1))
    curvature = turns / (0.5 * (lengths + np.roll(lengths, 1)))
    normal = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    arc = np.r_[0.0, np.cumsum(lengths)]
    return Track(
        track_id,
        name,
        seed,
        width,
        points,
        arc,
        tangents,
        curvature,
        points + normal * (width / 2.0),
        points - normal * (width / 2.0),
        np.arange(0.0, arc[-1], checkpoint_spacing, dtype=np.float64),
    )


def validity_errors(track: Track, max_curvature: float | None = None) -> list[str]:
    errors = []
    if polylines_intersect(track.centerline):
        errors.append("centerline self-intersects")
    if polylines_intersect(track.left) or polylines_intersect(track.right):
        errors.append("boundary self-intersects")
    if polylines_intersect(track.left, track.right):
        errors.append("boundaries overlap")
    curvature = float(np.max(np.abs(track.curvature)))
    if curvature * track.width / 2.0 >= 1.0:
        errors.append("road folds inside a corner")
    if max_curvature is not None and curvature > max_curvature:
        errors.append("curvature exceeds driveable limit")
    return errors


def generate_track(
    seed: int,
    config: TrackConfig,
    physics: PhysicsConfig,
    *,
    track_id: str | None = None,
    name: str | None = None,
) -> Track:
    rng = np.random.default_rng(seed)
    curvature_limit = min(
        config.max_curvature, tan(physics.max_steer) / (physics.front_axle + physics.rear_axle)
    )
    for _ in range(config.max_attempts):
        count = int(rng.integers(config.min_control_points, config.max_control_points + 1))
        spacing = 2.0 * np.pi / count
        angles = spacing * (
            np.arange(count) + rng.uniform(-config.angle_jitter, config.angle_jitter, count)
        )
        radii = config.radius * (
            1.0 + rng.uniform(-config.radius_noise, config.radius_noise, count)
        )
        control = np.column_stack(
            (radii * np.cos(angles) * config.aspect_ratio, radii * np.sin(angles))
        )
        centerline = resample_closed(
            catmull_rom(control, config.spline_samples_per_span), config.sample_spacing
        )
        width = float(rng.uniform(config.min_width, config.max_width))
        track = build_track(
            centerline,
            width,
            config.checkpoint_spacing,
            seed=seed,
            track_id=track_id or f"seed-{seed}",
            name=name or f"Seed {seed}",
        )
        if track.length > 2.0 * config.checkpoint_spacing and not validity_errors(
            track, curvature_limit
        ):
            return track
    raise ValueError(
        f"Unable to generate a valid track for seed {seed} in {config.max_attempts} tries"
    )


def showcase_track(item: ShowcaseConfig, config: TrackConfig, physics: PhysicsConfig) -> Track:
    return generate_track(
        item.seed,
        replace(
            config,
            radius=item.radius,
            radius_noise=item.radius_noise,
            aspect_ratio=item.aspect_ratio,
        ),
        physics,
        track_id=item.id,
        name=item.name,
    )

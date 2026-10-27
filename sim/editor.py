"""Deterministic editor geometry; Python is the source of truth for the web builder."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from sim.config import EditorConfig
from sim.crossings import circular_distance, find_crossings
from sim.geometry import FloatArray, catmull_rom, polyline_intersections, resample_closed
from sim.track import Track, build_track


def build_editor_track(
    control: FloatArray, config: EditorConfig, over: Sequence[int] | None = None
) -> Track:
    control = np.asarray(control, dtype=np.float64)
    if (
        control.ndim != 2
        or control.shape[1] != 2
        or len(control) < 4
        or not np.isfinite(control).all()
    ):
        raise ValueError("At least four finite 2D controls required")
    # Share links store integer quantum coordinates: canonicalize before scaling.
    control = np.round(control / config.quantum) * config.quantum
    perimeter = float(np.linalg.norm(np.roll(control, -1, axis=0) - control, axis=1).sum())
    if perimeter < 1e-8:
        raise ValueError("Control polygon has zero length")
    mean = control.mean(axis=0)
    control = mean + (control - mean) * (config.length / perimeter)
    control = np.round(control / config.quantum) * config.quantum
    centerline = resample_closed(
        catmull_rom(control, config.samples_per_span), config.sample_spacing
    )
    # Avoid a near-coincident closing sample: offset boundaries can reverse on
    # the tiny remainder even when the centerline has gentle curvature.
    if np.linalg.norm(centerline[-1] - centerline[0]) < config.sample_spacing * 0.5:
        centerline = centerline[:-1]
    track = build_track(
        centerline, config.width, config.checkpoint_spacing, track_id="custom", name="Your track"
    )
    return replace(track, crossings=find_crossings(track, config, over))


def editor_issues(track: Track, config: EditorConfig) -> list[str]:
    issues = set()
    if np.any(np.abs(track.curvature) * track.width / 2 >= 1):
        issues.add("fold")
    arcs = [a for c in track.crossings for a in c.arcs]
    if any(c.angle < config.min_crossing_angle for c in track.crossings):
        issues.add("shallow-crossing")
    if any(circular_distance(a, 0, track.length) < config.crossing_start for a in arcs):
        issues.add("crossing-start")
    if any(
        circular_distance(a, b, track.length) < config.crossing_spacing
        for i, a in enumerate(arcs)
        for b in arcs[i + 1 :]
    ):
        issues.add("crossing-spacing")
    n = len(track.centerline)
    for a, b, oa, ob in [
        (track.centerline, None, 0, 0),
        (track.left, None, 0, 0),
        (track.right, None, n, n),
        (track.left, track.right, 0, n),
    ]:
        for i, j, _ in polyline_intersections(a, b):
            # Offset loops caused by a local folded corner already have a precise
            # issue. Never hide remote overlaps merely because a fold exists.
            delta = (j - i) % n
            local = (
                [(i + k) % n for k in range(delta + 1)]
                if delta <= n / 2
                else [(j + k) % n for k in range(n - delta + 1)]
            )
            local_fold = (
                a is not track.centerline
                and b is None
                and circular_distance(
                    float(track.arc_length[i]), float(track.arc_length[j]), track.length
                )
                <= 2 * track.width
                and any(abs(track.curvature[k]) * track.width / 2 >= 1 for k in local)
            )
            if local_fold:
                continue
            if not any(
                (i + oa in c.hidden[0] and j + ob in c.hidden[1])
                or (i + oa in c.hidden[1] and j + ob in c.hidden[0])
                for c in track.crossings
            ):
                issues.add("overlap")
    return sorted(issues)


def tight_sections(track: Track, limit: float) -> list[tuple[float, float]]:
    ranges = []
    for i, value in enumerate(track.curvature):
        if abs(value) > limit:
            start, end = float(track.arc_length[i]), float(track.arc_length[i + 1])
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
    return ranges

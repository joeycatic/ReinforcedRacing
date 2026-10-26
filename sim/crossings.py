"""Crossing zones refer to boundary segment ids, independent of bridge rendering."""

from collections.abc import Sequence
from math import acos, degrees, radians, sin

import numpy as np

from sim.config import EditorConfig
from sim.geometry import polyline_intersections
from sim.track import Crossing, Track


def circular_distance(a: float, b: float, length: float) -> float:
    delta = abs(a - b) % length
    return min(delta, length - delta)


def find_crossings(
    track: Track, config: EditorConfig, over: Sequence[int] | None = None
) -> tuple[Crossing, ...]:
    found = []
    n = len(track.centerline)
    for i, j, point in polyline_intersections(track.centerline):
        if any(np.linalg.norm(point - np.asarray(c.point)) < 1.0 for c in found):
            continue
        arcs = tuple(
            float(track.arc_length[k] + np.linalg.norm(point - track.centerline[k])) for k in (i, j)
        )
        vectors = [track.centerline[(k + 1) % n] - track.centerline[k] for k in (i, j)]
        cosine = abs(
            float(np.dot(*vectors) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
        )
        angle = degrees(acos(min(1.0, cosine)))
        zone = track.width / max(sin(radians(angle)), 1e-12) + config.zone_margin
        hidden = []
        for arc in arcs:
            ids = tuple(
                k
                for k in range(n)
                if circular_distance(float(track.arc_length[k]), arc, track.length) <= zone
            )
            hidden.append(ids + tuple(n + k for k in ids))
        found.append(Crossing(arcs, tuple(float(v) for v in point), angle, 1, tuple(hidden)))
    found.sort(key=lambda c: c.arcs[0])
    if over is not None:
        if len(over) != len(found) or any(
            type(v) not in (int, np.int64) or v not in (0, 1) for v in over
        ):
            raise ValueError("Crossing over must have one 0 or 1 per crossing")
        from dataclasses import replace

        found = [replace(c, over=int(v)) for c, v in zip(found, over, strict=True)]
    return tuple(found)


def branch(crossing: Crossing, arc: float, length: float) -> int:
    return int(
        circular_distance(arc, crossing.arcs[1], length)
        < circular_distance(arc, crossing.arcs[0], length)
    )


def hidden_segments(track: Track, arc: float) -> frozenset[int]:
    return frozenset(
        segment for c in track.crossings for segment in c.hidden[1 - branch(c, arc, track.length)]
    )

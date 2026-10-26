"""Small float64 polyline primitives. Closed polylines omit a repeated endpoint."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cross(a: FloatArray, b: FloatArray) -> FloatArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polylines_intersect(a: FloatArray, b: FloatArray | None = None) -> bool:
    """Include touching/collinear overlaps; exclude neighbors of the same loop."""
    same = b is None
    if b is None:
        b = a
    a_end, b_end = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    a_min, a_max = np.minimum(a, a_end), np.maximum(a, a_end)
    b_min, b_max = np.minimum(b, b_end), np.maximum(b, b_end)
    candidates = np.all(a_max[:, None] >= b_min[None, :] - 1e-10, axis=-1) & np.all(
        b_max[None, :] >= a_min[:, None] - 1e-10, axis=-1
    )
    i, j = np.nonzero(candidates)
    if same:
        mask = (j > i + 1) & ~((i == 0) & (j == len(a) - 1))
        i, j = i[mask], j[mask]
    if len(i) == 0:
        return False
    p, q, r, s = a[i], a_end[i], b[j], b_end[j]
    o1, o2 = cross(q - p, r - p), cross(q - p, s - p)
    o3, o4 = cross(s - r, p - r), cross(s - r, q - r)
    return bool(np.any((o1 * o2 <= 1e-16) & (o3 * o4 <= 1e-16)))


def catmull_rom(control: FloatArray, samples_per_span: int) -> FloatArray:
    t = np.arange(samples_per_span, dtype=np.float64)[:, None] / samples_per_span
    spans = []
    for i in range(len(control)):
        p0, p1, p2, p3 = (control[j % len(control)] for j in (i - 1, i, i + 1, i + 2))
        spans.append(
            0.5
            * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
            )
        )
    return np.concatenate(spans)


def resample_closed(points: FloatArray, spacing: float) -> FloatArray:
    closed = np.vstack((points, points[0]))
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    samples = np.arange(0.0, cumulative[-1], spacing, dtype=np.float64)
    return np.column_stack(
        (np.interp(samples, cumulative, closed[:, 0]), np.interp(samples, cumulative, closed[:, 1]))
    )

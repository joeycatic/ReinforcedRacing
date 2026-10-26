"""Batched boundary rays using a precomputed uniform grid and exact cell traversal."""

from collections.abc import Sequence
from math import ceil

import numpy as np
from numpy.typing import NDArray

from sim.geometry import FloatArray
from sim.track import Track

IntArray = NDArray[np.int64]
EPS = 1e-10  # Geometric tolerance, not a tunable driving parameter.


class BoundaryGrid:
    def __init__(self, tracks: Sequence[Track], cell_size: float) -> None:
        if not tracks:
            raise ValueError("A boundary grid requires at least one track")
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be finite and positive")
        self.cell_size = cell_size
        origins, sizes, offsets = [], [], []
        starts, vectors = [np.zeros((1, 2))], [np.zeros((1, 2))]
        cells: list[list[int]] = []
        tags = [np.zeros(1, dtype=np.int64)]
        self.tag_count = 0
        segment_offset = 1  # Zero is an invalid/sentinel segment in empty grid slots.
        for track in tracks:
            start = np.vstack((track.left, track.right))
            end = np.vstack((np.roll(track.left, -1, axis=0), np.roll(track.right, -1, axis=0)))
            low = np.floor(start.min(axis=0) / cell_size) * cell_size - cell_size
            size = np.ceil((start.max(axis=0) - low) / cell_size).astype(np.int64) + 1
            offset = len(cells)
            cells.extend([] for _ in range(int(size[0] * size[1])))
            for segment, (a, b) in enumerate(zip(start, end, strict=True)):
                first = np.floor((np.minimum(a, b) - low - EPS) / cell_size).astype(int)
                last = np.floor((np.maximum(a, b) - low + EPS) / cell_size).astype(int)
                for y in range(first[1], last[1] + 1):
                    for x in range(first[0], last[0] + 1):
                        cells[offset + y * size[0] + x].append(segment_offset + segment)
            track_tags = np.zeros(len(start), dtype=np.int64)
            for crossing in track.crossings:
                for hidden in crossing.hidden:
                    self.tag_count += 1
                    track_tags[list(hidden)] = self.tag_count
            tags.append(track_tags)
            starts.append(start)
            vectors.append(end - start)
            segment_offset += len(start)
            origins.append(low)
            sizes.append(size)
            offsets.append(offset)
        self.origins = np.asarray(origins, dtype=np.float64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.tags = np.concatenate(tags)
        self.starts = np.concatenate(starts)
        self.vectors = np.concatenate(vectors)
        self.cells = np.zeros((len(cells), max(map(len, cells), default=1)), dtype=np.int64)
        for index, segments in enumerate(cells):
            self.cells[index, : len(segments)] = segments

    def cast(
        self,
        track_ids: IntArray,
        origins: FloatArray,
        directions: FloatArray,
        max_distance: float,
        hidden: NDArray[np.bool_] | None = None,
    ) -> FloatArray:
        """Origins (N,2), unit directions (N,R,2) -> nearest distances (N,R).

        All cars/rays traverse cells together. Only boundary segments belonging to
        the current cell are intersected. Rays outside the grid enter via a slab
        intersection; collinear rays correctly detect touching/overlapping edges.
        """
        shape = directions.shape[:2]
        ray_count = shape[1]
        ids = np.repeat(track_ids, ray_count)
        origin = np.repeat(origins, ray_count, axis=0)
        direction = directions.reshape(-1, 2)
        parallel = np.abs(direction) < EPS
        direction = np.where(parallel, 0.0, direction)
        safe_direction = np.where(parallel, 1.0, direction)
        low = self.origins[ids]
        size = self.sizes[ids]
        high = low + size * self.cell_size
        t0 = (low - origin) / safe_direction
        t1 = (high - origin) / safe_direction
        enter = np.max(np.where(parallel, -np.inf, np.minimum(t0, t1)), axis=1)
        leave = np.min(np.where(parallel, np.inf, np.maximum(t0, t1)), axis=1)
        start = np.maximum(0.0, enter)
        end = np.minimum(max_distance, leave)
        active = (start <= end) & ~np.any(parallel & ((origin < low) | (origin > high)), axis=1)
        start = np.where(active, start, 0.0)
        position = origin + direction * (start[:, None] + EPS)
        cell = np.clip(np.floor((position - low) / self.cell_size).astype(np.int64), 0, size - 1)
        stride = np.sign(direction).astype(np.int64)
        next_boundary = low + (cell + (stride > 0)) * self.cell_size
        crossing = np.where(parallel, np.inf, (next_boundary - origin) / safe_direction)
        increment = np.where(parallel, np.inf, self.cell_size / np.abs(safe_direction))
        result = np.full(len(ids), max_distance, dtype=np.float64)
        # A unit ray can cross at most 2*distance/cell_size + 2 grid lines.
        for _ in range(ceil(2 * max_distance / self.cell_size) + 4):
            rays = np.flatnonzero(active)
            if len(rays) == 0:
                break
            cell_ids = self.offsets[ids[rays]] + cell[rays, 1] * size[rays, 0] + cell[rays, 0]
            segments = self.cells[cell_ids]
            vector = self.vectors[segments]
            relative = self.starts[segments] - origin[rays, None, :]
            dx, dy = direction[rays, 0, None], direction[rays, 1, None]
            denominator = dx * vector[..., 1] - dy * vector[..., 0]
            nonparallel = np.abs(denominator) > EPS
            numerator = relative[..., 0] * vector[..., 1] - relative[..., 1] * vector[..., 0]
            distance = np.divide(
                numerator, denominator, out=np.full_like(denominator, np.inf), where=nonparallel
            )
            lateral = relative[..., 0] * dy - relative[..., 1] * dx
            fraction = np.divide(
                lateral, denominator, out=np.full_like(denominator, np.inf), where=nonparallel
            )
            valid = nonparallel & (fraction >= -EPS) & (fraction <= 1.0 + EPS)
            # Parallel and collinear: intersect the projected 1D intervals.
            along = relative[..., 0] * dx + relative[..., 1] * dy
            other = along + vector[..., 0] * dx + vector[..., 1] * dy
            collinear = ~nonparallel & (np.abs(lateral) <= EPS) & (np.maximum(along, other) >= 0)
            distance = np.where(collinear, np.maximum(0.0, np.minimum(along, other)), distance)
            cell_end = np.minimum(np.min(crossing[rays], axis=1), end[rays])
            valid = (
                (valid | collinear)
                & (segments != 0)
                & (distance >= start[rays, None] - EPS)
                & (distance <= cell_end[:, None] + EPS)
            )
            if hidden is not None:
                valid &= ~hidden[(rays // ray_count)[:, None], self.tags[segments]]
            nearest = np.min(np.where(valid, np.maximum(distance, 0.0), np.inf), axis=1)
            hit = np.isfinite(nearest)
            result[rays[hit]] = np.minimum(nearest[hit], max_distance)
            active[rays[hit | (cell_end >= end[rays])]] = False
            moving = rays[~hit & (cell_end < end[rays])]
            if len(moving):
                times = crossing[moving]
                advance = times <= np.min(times, axis=1)[:, None]
                cell[moving] += advance * stride[moving]
                # Avoid inf*0 for rays parallel to one grid axis.
                crossing[moving] = np.where(advance, crossing[moving] + increment[moving], times)
                start[moving] = np.min(times, axis=1)
                active[moving] &= np.all(
                    (cell[moving] >= 0) & (cell[moving] < size[moving]), axis=1
                )
        return result.reshape(shape)

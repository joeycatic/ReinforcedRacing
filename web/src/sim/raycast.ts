import { Track } from "./track";
import type { Vec2 } from "./types";
import { clamp } from "./physics";

const EPS = 1e-10;
/** Exact uniform-grid DDA, matching sim/raycast.py, including collinear edges. */
export class BoundaryGrid {
  readonly origin: Vec2;
  readonly size: Vec2;
  readonly cells: number[][];
  readonly starts: Vec2[] = [];
  readonly vectors: Vec2[] = [];
  constructor(
    track: Track,
    readonly cellSize: number,
  ) {
    if (!Number.isFinite(cellSize) || cellSize <= 0)
      throw new Error("Invalid grid cell size");
    const boundaries = [track.data.left, track.data.right],
      points = boundaries.flat();
    const min: Vec2 = [
      Math.min(...points.map((p) => p[0])),
      Math.min(...points.map((p) => p[1])),
    ];
    const max: Vec2 = [
      Math.max(...points.map((p) => p[0])),
      Math.max(...points.map((p) => p[1])),
    ];
    this.origin = [
      Math.floor(min[0] / cellSize) * cellSize - cellSize,
      Math.floor(min[1] / cellSize) * cellSize - cellSize,
    ];
    this.size = [
      Math.ceil((max[0] - this.origin[0]) / cellSize) + 1,
      Math.ceil((max[1] - this.origin[1]) / cellSize) + 1,
    ];
    this.cells = Array.from({ length: this.size[0] * this.size[1] }, () => []);
    for (const boundary of boundaries)
      for (let i = 0; i < boundary.length; i++) {
        const a = boundary[i],
          b = boundary[(i + 1) % boundary.length],
          segment = this.starts.length;
        this.starts.push(a);
        this.vectors.push([b[0] - a[0], b[1] - a[1]]);
        const first: Vec2 = [0, 1].map((k) =>
          Math.floor((Math.min(a[k], b[k]) - this.origin[k] - EPS) / cellSize),
        ) as Vec2;
        const last: Vec2 = [0, 1].map((k) =>
          Math.floor((Math.max(a[k], b[k]) - this.origin[k] + EPS) / cellSize),
        ) as Vec2;
        for (let y = first[1]; y <= last[1]; y++)
          for (let x = first[0]; x <= last[0]; x++)
            this.cells[y * this.size[0] + x].push(segment);
      }
  }
  cast(origin: Vec2, unitDirection: Vec2, maxDistance: number): number {
    const parallel = unitDirection.map((v) => Math.abs(v) < EPS);
    const direction = unitDirection.map((v, k) => (parallel[k] ? 0 : v));
    const safe = direction.map((v, k) => (parallel[k] ? 1 : v));
    const high = this.origin.map((v, k) => v + this.size[k] * this.cellSize);
    const t0 = this.origin.map((v, k) => (v - origin[k]) / safe[k]);
    const t1 = high.map((v, k) => (v - origin[k]) / safe[k]);
    const enter = Math.max(
      ...t0.map((v, k) => (parallel[k] ? -Infinity : Math.min(v, t1[k]))),
    );
    const leave = Math.min(
      ...t0.map((v, k) => (parallel[k] ? Infinity : Math.max(v, t1[k]))),
    );
    let start = Math.max(0, enter);
    const end = Math.min(maxDistance, leave);
    if (
      start > end ||
      parallel.some(
        (v, k) => v && (origin[k] < this.origin[k] || origin[k] > high[k]),
      )
    )
      return maxDistance;
    const position = origin.map((v, k) => v + direction[k] * (start + EPS));
    const cell = position.map((v, k) =>
      clamp(
        Math.floor((v - this.origin[k]) / this.cellSize),
        0,
        this.size[k] - 1,
      ),
    );
    const stride = direction.map(Math.sign);
    const crossing = cell.map((v, k) =>
      parallel[k]
        ? Infinity
        : (this.origin[k] +
            (v + (stride[k] > 0 ? 1 : 0)) * this.cellSize -
            origin[k]) /
          safe[k],
    );
    const increment = safe.map((v, k) =>
      parallel[k] ? Infinity : this.cellSize / Math.abs(v),
    );
    for (
      let iteration = 0;
      iteration < Math.ceil((2 * maxDistance) / this.cellSize) + 4;
      iteration++
    ) {
      const cellEnd = Math.min(...crossing, end);
      let nearest = Infinity;
      for (const segment of this.cells[cell[1] * this.size[0] + cell[0]]) {
        const vector = this.vectors[segment],
          a = this.starts[segment];
        const rx = a[0] - origin[0],
          ry = a[1] - origin[1],
          dx = direction[0],
          dy = direction[1];
        const denominator = dx * vector[1] - dy * vector[0],
          nonparallel = Math.abs(denominator) > EPS;
        let distance = nonparallel
          ? (rx * vector[1] - ry * vector[0]) / denominator
          : Infinity;
        const lateral = rx * dy - ry * dx;
        const fraction = nonparallel ? lateral / denominator : Infinity;
        const valid = nonparallel && fraction >= -EPS && fraction <= 1 + EPS;
        const along = rx * dx + ry * dy,
          other = along + vector[0] * dx + vector[1] * dy;
        const collinear =
          !nonparallel &&
          Math.abs(lateral) <= EPS &&
          Math.max(along, other) >= 0;
        if (collinear) distance = Math.max(0, Math.min(along, other));
        if (
          (valid || collinear) &&
          distance >= start - EPS &&
          distance <= cellEnd + EPS
        )
          nearest = Math.min(nearest, Math.max(distance, 0));
      }
      if (Number.isFinite(nearest)) return Math.min(nearest, maxDistance);
      if (cellEnd >= end) return maxDistance;
      const time = Math.min(...crossing);
      for (let k = 0; k < 2; k++)
        if (crossing[k] <= time) {
          cell[k] += stride[k];
          crossing[k] += increment[k];
        }
      start = time;
      if (cell.some((v, k) => v < 0 || v >= this.size[k])) return maxDistance;
    }
    return maxDistance;
  }
}

import type { Vec2 } from "./types";
export const cross = (a: Vec2, b: Vec2): number => a[0] * b[1] - a[1] * b[0];
const sub = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
export const norm = (a: Vec2): number => Math.sqrt(a[0] * a[0] + a[1] * a[1]);

/** Closed loops omit a repeated endpoint. Candidate order matches numpy.nonzero. */
export function polylineIntersections(
  a: Vec2[],
  b?: Vec2[],
  cellSize = 24,
): [number, number, Vec2][] {
  const same = b === undefined;
  b ??= a;
  const buckets = new Map<string, number[]>();
  const bounds = (p: Vec2, q: Vec2): number[] => [
    Math.min(p[0], q[0]),
    Math.min(p[1], q[1]),
    Math.max(p[0], q[0]),
    Math.max(p[1], q[1]),
  ];
  const bb = b.map((p, j) => bounds(p, b![(j + 1) % b!.length]));
  for (let j = 0; j < b.length; j++) {
    const v = bb[j];
    for (
      let x = Math.floor((v[0] - 1e-10) / cellSize);
      x <= Math.floor((v[2] + 1e-10) / cellSize);
      x++
    )
      for (
        let y = Math.floor((v[1] - 1e-10) / cellSize);
        y <= Math.floor((v[3] + 1e-10) / cellSize);
        y++
      ) {
        const key = `${x},${y}`;
        const bucket = buckets.get(key);
        if (bucket) bucket.push(j);
        else buckets.set(key, [j]);
      }
  }
  const result: [number, number, Vec2][] = [];
  for (let i = 0; i < a.length; i++) {
    const p = a[i],
      q = a[(i + 1) % a.length],
      v = bounds(p, q),
      candidates = new Set<number>();
    for (
      let x = Math.floor(v[0] / cellSize);
      x <= Math.floor(v[2] / cellSize);
      x++
    )
      for (
        let y = Math.floor(v[1] / cellSize);
        y <= Math.floor(v[3] / cellSize);
        y++
      )
        for (const j of buckets.get(`${x},${y}`) ?? []) candidates.add(j);
    for (const j of [...candidates].sort((x, y) => x - y)) {
      if (same && (j <= i + 1 || (i === 0 && j === a.length - 1))) continue;
      const w = bb[j];
      if (
        v[2] < w[0] - 1e-10 ||
        w[2] < v[0] - 1e-10 ||
        v[3] < w[1] - 1e-10 ||
        w[3] < v[1] - 1e-10
      )
        continue;
      const r = b[j],
        s = b[(j + 1) % b.length],
        d = sub(q, p),
        other = sub(s, r);
      const o1 = cross(d, sub(r, p)),
        o2 = cross(d, sub(s, p)),
        o3 = cross(other, sub(p, r)),
        o4 = cross(other, sub(q, r));
      if (o1 * o2 > 1e-16 || o3 * o4 > 1e-16) continue;
      const denominator = cross(d, other);
      let fraction: number;
      if (Math.abs(denominator) > 1e-16)
        fraction = cross(sub(r, p), other) / denominator;
      else {
        const axis = Math.abs(d[0]) >= Math.abs(d[1]) ? 0 : 1;
        if (Math.abs(d[axis]) < 1e-16) fraction = 0;
        else {
          const u = (r[axis] - p[axis]) / d[axis],
            v = (s[axis] - p[axis]) / d[axis];
          fraction =
            (Math.max(0, Math.min(u, v)) + Math.min(1, Math.max(u, v))) / 2;
        }
      }
      result.push([i, j, [p[0] + fraction * d[0], p[1] + fraction * d[1]]]);
    }
  }
  return result;
}
export function catmullRom(control: Vec2[], samplesPerSpan: number): Vec2[] {
  const points: Vec2[] = [];
  const n = control.length;
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [
      control[(i + n - 1) % n],
      control[i],
      control[(i + 1) % n],
      control[(i + 2) % n],
    ];
    for (let k = 0; k < samplesPerSpan; k++) {
      const t = k / samplesPerSpan;
      points.push(
        [0, 1].map(
          (axis) =>
            0.5 *
            (2 * p1[axis] +
              (-p0[axis] + p2[axis]) * t +
              (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t ** 2 +
              (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * t ** 3),
        ) as Vec2,
      );
    }
  }
  return points;
}
export function resampleClosed(points: Vec2[], spacing: number): Vec2[] {
  const closed = [...points, points[0]],
    arc = [0];
  for (let i = 1; i < closed.length; i++)
    arc.push(arc[i - 1] + norm(sub(closed[i], closed[i - 1])));
  const result: Vec2[] = [];
  let index = 0;
  const count = Math.ceil(arc[arc.length - 1] / spacing);
  for (let k = 0; k < count; k++) {
    const sample = k * spacing;
    while (index + 1 < arc.length && arc[index + 1] <= sample) index++;
    result.push(
      [0, 1].map(
        (axis) =>
          closed[index][axis] +
          ((closed[index + 1][axis] - closed[index][axis]) /
            (arc[index + 1] - arc[index])) *
            (sample - arc[index]),
      ) as Vec2,
    );
  }
  return result;
}

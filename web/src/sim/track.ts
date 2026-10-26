import { findCrossings } from "./crossings";
import editorConfig from "../editor/editor-config.json";
import type { TrackData, Vec2 } from "./types";
import { VERSIONS, TRACK_V2 } from "../assets/versions";
import { clamp } from "./physics";

// Python's positive modulo, without an extra rounding-prone addition for positive inputs.
export function modulo(x: number, divisor: number): number {
  const remainder = x % divisor;
  return remainder < 0 ? remainder + divisor : remainder === 0 ? 0 : remainder;
}
export function searchSorted(
  values: number[],
  value: number,
  right = false,
): number {
  let low = 0,
    high = values.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (values[mid] < value || (right && values[mid] === value)) low = mid + 1;
    else high = mid;
  }
  return low;
}
export class Track {
  readonly length: number;
  constructor(readonly data: TrackData) {
    if (data.version !== VERSIONS.track && data.version !== TRACK_V2)
      throw new Error(`Unsupported track version: ${data.version}`);
    const n = data.centerline.length;
    if (
      n < 4 ||
      !Number.isInteger(data.seed) ||
      typeof data.id !== "string" ||
      !(data.width > 0) ||
      !Number.isFinite(data.width)
    )
      throw new Error("Invalid track");
    for (const points of [
      data.centerline,
      data.tangents,
      data.left,
      data.right,
    ]) {
      if (
        points.length !== n ||
        points.some((p) => p.length !== 2 || !p.every(Number.isFinite))
      )
        throw new Error("Invalid track vectors");
    }
    if (
      data.arc_length.length !== n + 1 ||
      data.arc_length[0] !== 0 ||
      data.curvature.length !== n ||
      !data.curvature.every(Number.isFinite)
    )
      throw new Error("Invalid track arcs/curvature");
    if (
      data.arc_length.some(
        (v, i) => !Number.isFinite(v) || (i > 0 && v <= data.arc_length[i - 1]),
      )
    )
      throw new Error("Track arcs must increase");
    this.length = data.arc_length[n];
    if (
      data.checkpoints.length < 2 ||
      data.checkpoints[0] !== 0 ||
      data.checkpoints.some(
        (v, i) =>
          !Number.isFinite(v) ||
          v >= this.length ||
          (i > 0 && v <= data.checkpoints[i - 1]),
      )
    )
      throw new Error("Invalid checkpoints");
    for (let i = 0; i < n; i++) {
      const a = data.centerline[i],
        b = data.centerline[(i + 1) % n];
      if (
        Math.abs(
          Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) -
            (data.arc_length[i + 1] - data.arc_length[i]),
        ) > 1e-9
      )
        throw new Error("Inconsistent segment length");
    }
    if (data.version === TRACK_V2) {
      if (
        data.id !== "custom" ||
        !Array.isArray(data.crossings) ||
        !data.crossings.length
      )
        throw new Error("track.v2 requires custom track crossings");
      const expected = findCrossings(
        data,
        editorConfig,
        data.crossings.map((c) => c.over),
      );
      if (expected.length !== data.crossings.length)
        throw new Error("Inconsistent crossings");
      expected.forEach((c, i) => {
        const supplied = data.crossings![i];
        if (
          Object.keys(supplied).sort().join() !==
            Object.keys(c).sort().join() ||
          !Array.isArray(supplied.arcs) ||
          supplied.arcs.length !== 2 ||
          !Array.isArray(supplied.point) ||
          supplied.point.length !== 2 ||
          [...c.arcs, ...c.point, c.angle].some(
            (v, k) =>
              !Number.isFinite(
                [...supplied.arcs, ...supplied.point, supplied.angle][k],
              ) ||
              Math.abs(
                v - [...supplied.arcs, ...supplied.point, supplied.angle][k],
              ) > 1e-9,
          ) ||
          JSON.stringify(c.hidden) !== JSON.stringify(supplied.hidden)
        )
          throw new Error("Inconsistent crossing geometry");
      });
    } else if (data.crossings !== undefined)
      throw new Error("track.v1 cannot contain crossings");
  }
  sample(arc: number): [Vec2, number] {
    const s = modulo(arc, this.length),
      d = this.data,
      n = d.centerline.length;
    const i = Math.min(searchSorted(d.arc_length, s, true) - 1, n - 1),
      end = (i + 1) % n;
    const fraction =
      (s - d.arc_length[i]) / (d.arc_length[i + 1] - d.arc_length[i]);
    const vector: Vec2 = [
      d.centerline[end][0] - d.centerline[i][0],
      d.centerline[end][1] - d.centerline[i][1],
    ];
    return [
      [
        d.centerline[i][0] + fraction * vector[0],
        d.centerline[i][1] + fraction * vector[1],
      ],
      Math.atan2(vector[1], vector[0]),
    ];
  }
  curvatureAt(arc: number): number {
    const d = this.data,
      s = modulo(arc, this.length),
      i = searchSorted(d.arc_length, s, true) - 1,
      next = (i + 1) % d.centerline.length;
    // TrackPool stores the geometric segment lengths, not differences of cumulative arcs.
    const a = d.centerline[i],
      b = d.centerline[next];
    const length = Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);
    const fraction = (s - d.arc_length[i]) / length;
    return d.curvature[i] + fraction * (d.curvature[next] - d.curvature[i]);
  }
}
export interface Projection {
  index: number;
  arc: number;
  offset: number;
  distance: number;
}
export function project(
  track: Track,
  x: number,
  y: number,
  index: number | null,
  window: number,
): Projection {
  const d = track.data,
    n = d.centerline.length;
  let best = Infinity,
    result: Projection | undefined;
  const count = index === null ? n : 2 * window + 1;
  for (let k = 0; k < count; k++) {
    const i = index === null ? k : modulo(index - window + k, n),
      next = (i + 1) % n;
    const a = d.centerline[i],
      b = d.centerline[next],
      dx = b[0] - a[0],
      dy = b[1] - a[1];
    const lengthSq = dx * dx + dy * dy,
      rx = x - a[0],
      ry = y - a[1];
    const fraction = clamp((rx * dx + ry * dy) / lengthSq, 0, 1);
    const ex = rx - fraction * dx,
      ey = ry - fraction * dy,
      distanceSq = ex * ex + ey * ey;
    if (distanceSq < best) {
      best = distanceSq;
      const length = Math.sqrt(lengthSq);
      result = {
        index: i,
        arc: modulo(d.arc_length[i] + fraction * length, track.length),
        offset: (dx * ey - dy * ex) / length,
        distance: Math.sqrt(distanceSq),
      };
    }
  }
  return result!;
}

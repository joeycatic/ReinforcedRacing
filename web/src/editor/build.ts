import { Track } from "../sim/track";
import type { TrackData, Vec2 } from "../sim/types";
import {
  catmullRom,
  resampleClosed,
  norm,
  cross,
  polylineIntersections,
} from "../sim/geometry";
import { circularDistance, findCrossings } from "../sim/crossings";
import { TRACK_V2, VERSIONS } from "../assets/versions";
export interface EditorConfig {
  ui?: { closeDistancePx: number; minStrokePx: number; rebuildMs: number };
  width: number;
  length: number;
  control_points: number;
  samples_per_span: number;
  sample_spacing: number;
  checkpoint_spacing: number;
  min_crossing_angle: number;
  crossing_spacing: number;
  crossing_start: number;
  zone_margin: number;
  max_curvature: number;
  quantum: number;
}
export function buildTrackData(
  centerline: Vec2[],
  width: number,
  checkpointSpacing: number,
  id: string,
  name: string,
): TrackData {
  if (
    centerline.length < 4 ||
    centerline.some((p) => p.length !== 2 || !p.every(Number.isFinite)) ||
    !Number.isFinite(width) ||
    width <= 0 ||
    !Number.isFinite(checkpointSpacing) ||
    checkpointSpacing <= 0
  )
    throw new Error("Invalid track geometry");
  const points = centerline.map((p) => [...p] as Vec2),
    n = points.length;
  const vectors = points.map(
      (p, i) =>
        [points[(i + 1) % n][0] - p[0], points[(i + 1) % n][1] - p[1]] as Vec2,
    ),
    lengths = vectors.map(norm);
  if (lengths.some((l) => l < 1e-8)) throw new Error("Zero-length segment");
  const segments = vectors.map(
    (v, i) => [v[0] / lengths[i], v[1] / lengths[i]] as Vec2,
  );
  const tangents: Vec2[] = [],
    curvature: number[] = [],
    left: Vec2[] = [],
    right: Vec2[] = [],
    arc = [0];
  for (let i = 0; i < n; i++) {
    const previous = segments[(i + n - 1) % n],
      current = segments[i],
      t: Vec2 = [previous[0] + current[0], previous[1] + current[1]],
      l = norm(t);
    if (l < 1e-8) throw new Error("Reversing cusp");
    t[0] /= l;
    t[1] /= l;
    tangents.push(t);
    curvature.push(
      Math.atan2(
        cross(previous, current),
        previous[0] * current[0] + previous[1] * current[1],
      ) /
        (0.5 * (lengths[i] + lengths[(i + n - 1) % n])),
    );
    const normal: Vec2 = [-t[1], t[0]],
      p = points[i];
    left.push([p[0] + normal[0] * (width / 2), p[1] + normal[1] * (width / 2)]);
    right.push([
      p[0] - normal[0] * (width / 2),
      p[1] - normal[1] * (width / 2),
    ]);
    arc.push(arc[i] + lengths[i]);
  }
  const checkpoints = Array.from(
    { length: Math.ceil(arc[n] / checkpointSpacing) },
    (_, i) => i * checkpointSpacing,
  );
  return {
    version: VERSIONS.track,
    id,
    name,
    seed: 0,
    width,
    centerline: points,
    arc_length: arc,
    tangents,
    curvature,
    left,
    right,
    checkpoints,
  };
}
/** numpy.round: ties to even, including negative halfway values. */
function roundEven(value: number): number {
  const lo = Math.floor(value),
    fraction = value - lo;
  return fraction === 0.5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(value);
}
export function buildEditorTrack(
  control: Vec2[],
  config: EditorConfig,
  over?: number[],
): { track: Track; issues: string[]; tight: [number, number][] } {
  if (
    control.length < 4 ||
    control.some((p) => p.length !== 2 || !p.every(Number.isFinite))
  )
    throw new Error("At least four finite 2D controls required");
  const mean: Vec2 = [0, 0];
  for (const p of control) {
    mean[0] += p[0];
    mean[1] += p[1];
  }
  mean[0] /= control.length;
  mean[1] /= control.length;
  let perimeter = 0;
  for (let i = 0; i < control.length; i++)
    perimeter += norm([
      control[(i + 1) % control.length][0] - control[i][0],
      control[(i + 1) % control.length][1] - control[i][1],
    ]);
  if (perimeter < 1e-8) throw new Error("Control polygon has zero length");
  const scaled = control.map(
    (p) =>
      p.map(
        (v, k) =>
          roundEven(
            (mean[k] + (v - mean[k]) * (config.length / perimeter)) /
              config.quantum,
          ) * config.quantum,
      ) as Vec2,
  );
  const data = buildTrackData(
    resampleClosed(
      catmullRom(scaled, config.samples_per_span),
      config.sample_spacing,
    ),
    config.width,
    config.checkpoint_spacing,
    "custom",
    "Your track",
  );
  const crossings = findCrossings(data, config, over);
  if (crossings.length) {
    data.version = TRACK_V2;
    data.crossings = crossings;
  }
  const track = new Track(data);
  return {
    track,
    issues: editorIssues(track, config),
    tight: tightSections(track, config.max_curvature),
  };
}
export function editorIssues(track: Track, config: EditorConfig): string[] {
  const d = track.data,
    n = d.centerline.length,
    issues = new Set<string>(),
    crossings = d.crossings ?? [];
  if (d.curvature.some((v) => (Math.abs(v) * d.width) / 2 >= 1))
    issues.add("fold");
  if (crossings.some((c) => c.angle < config.min_crossing_angle))
    issues.add("shallow-crossing");
  const arcs = crossings.flatMap((c) => c.arcs);
  if (
    arcs.some(
      (a) => circularDistance(a, 0, track.length) < config.crossing_start,
    )
  )
    issues.add("crossing-start");
  if (
    arcs.some((a, i) =>
      arcs
        .slice(i + 1)
        .some(
          (b) => circularDistance(a, b, track.length) < config.crossing_spacing,
        ),
    )
  )
    issues.add("crossing-spacing");
  const zones = crossings.map((c) => c.hidden.map((ids) => new Set(ids)));
  const pairs: [Vec2[], Vec2[] | undefined, number, number][] = [
    [d.centerline, undefined, 0, 0],
    [d.left, undefined, 0, 0],
    [d.right, undefined, n, n],
    [d.left, d.right, 0, n],
  ];
  for (const [a, b, oa, ob] of pairs)
    for (const [i, j] of polylineIntersections(a, b, 2 * d.width)) {
      const delta = (j - i + n) % n,
        local =
          delta <= n / 2
            ? Array.from({ length: delta + 1 }, (_, k) => (i + k) % n)
            : Array.from({ length: n - delta + 1 }, (_, k) => (j + k) % n);
      const localFold =
        a !== d.centerline &&
        b === undefined &&
        circularDistance(d.arc_length[i], d.arc_length[j], track.length) <=
          2 * d.width &&
        local.some((k) => (Math.abs(d.curvature[k]) * d.width) / 2 >= 1);
      if (localFold) continue;
      if (
        !zones.some(
          (c) =>
            (c[0].has(i + oa) && c[1].has(j + ob)) ||
            (c[1].has(i + oa) && c[0].has(j + ob)),
        )
      )
        issues.add("overlap");
    }
  return [...issues].sort();
}
export function tightSections(track: Track, limit: number): [number, number][] {
  const ranges: [number, number][] = [];
  track.data.curvature.forEach((v, i) => {
    if (Math.abs(v) > limit) {
      const start = track.data.arc_length[i],
        end = track.data.arc_length[i + 1];
      if (ranges.length && ranges[ranges.length - 1][1] === start)
        ranges[ranges.length - 1][1] = end;
      else ranges.push([start, end]);
    }
  });
  return ranges;
}

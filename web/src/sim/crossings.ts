import type { EditorConfig } from "../editor/build";
import type { TrackData, Vec2 } from "./types";
import type { Track } from "./track";
import { norm, polylineIntersections } from "./geometry";
export interface Crossing {
  arcs: [number, number];
  point: Vec2;
  angle: number;
  over: number;
  hidden: [number[], number[]];
}
export function circularDistance(a: number, b: number, length: number): number {
  const d = Math.abs(a - b) % length;
  return Math.min(d, length - d);
}
export function findCrossings(
  data: TrackData,
  config: EditorConfig,
  over?: number[],
): Crossing[] {
  const found: Crossing[] = [];
  const n = data.centerline.length,
    length = data.arc_length[n];
  for (const [i, j, point] of polylineIntersections(
    data.centerline,
    undefined,
    2 * data.width,
  )) {
    if (
      found.some(
        (c) => norm([point[0] - c.point[0], point[1] - c.point[1]]) < 1,
      )
    )
      continue;
    const arcs = [i, j].map(
      (k) =>
        data.arc_length[k] +
        norm([
          point[0] - data.centerline[k][0],
          point[1] - data.centerline[k][1],
        ]),
    ) as [number, number];
    const vectors = [i, j].map(
      (k) =>
        [
          data.centerline[(k + 1) % n][0] - data.centerline[k][0],
          data.centerline[(k + 1) % n][1] - data.centerline[k][1],
        ] as Vec2,
    );
    const cosine = Math.abs(
      (vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]) /
        (norm(vectors[0]) * norm(vectors[1])),
    );
    const angle = Math.acos(Math.min(1, cosine)) * (180 / Math.PI),
      zone =
        data.width / Math.max(Math.sin(angle * (Math.PI / 180)), 1e-12) +
        config.zone_margin;
    const hidden = arcs.map((arc) => {
      const ids = data.centerline.flatMap((_, k) =>
        circularDistance(data.arc_length[k], arc, length) <= zone ? [k] : [],
      );
      return [...ids, ...ids.map((k) => n + k)];
    }) as [number[], number[]];
    found.push({ arcs, point, angle, over: 1, hidden });
  }
  found.sort((a, b) => a.arcs[0] - b.arcs[0]);
  if (over !== undefined) {
    if (over.length !== found.length || over.some((v) => v !== 0 && v !== 1))
      throw new Error("Crossing over must have one 0 or 1 per crossing");
    found.forEach((c, i) => (c.over = over[i]));
  }
  return found;
}
export function branchOf(c: Crossing, arc: number, length: number): 0 | 1 {
  return circularDistance(arc, c.arcs[1], length) <
    circularDistance(arc, c.arcs[0], length)
    ? 1
    : 0;
}
export function hiddenSegments(track: Track, arc: number): Set<number> {
  return new Set(
    (track.data.crossings ?? []).flatMap(
      (c) => c.hidden[1 - branchOf(c, arc, track.length)],
    ),
  );
}

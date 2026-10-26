import type { Vec2 } from "../sim/types";
import { visualConfig } from "../viewer/visual-config";

export function kerbStripes(
  edge: Vec2[],
  stripe: number,
): { points: Vec2[]; red: boolean }[] {
  if (stripe <= 0) throw new Error("Stripe length must be positive");
  if (edge.length < 2) return [];
  const points =
    edge.length > 2 &&
    (edge[0][0] !== edge.at(-1)![0] || edge[0][1] !== edge.at(-1)![1])
      ? [...edge, edge[0]]
      : edge;
  const stripes: { points: Vec2[]; red: boolean }[] = [];
  let current: Vec2[] = [points[0]],
    used = 0;
  for (let i = 1; i < points.length; i++) {
    let a = points[i - 1];
    const b = points[i];
    let distance = Math.hypot(b[0] - a[0], b[1] - a[1]);
    while (distance > 1e-9) {
      const length = Math.min(stripe - used, distance);
      const end: Vec2 = [
        a[0] + ((b[0] - a[0]) * length) / distance,
        a[1] + ((b[1] - a[1]) * length) / distance,
      ];
      current.push(end);
      used += length;
      distance -= length;
      a = end;
      if (used >= stripe - 1e-9) {
        stripes.push({ points: current, red: stripes.length % 2 === 0 });
        current = [end];
        used = 0;
      }
    }
  }
  if (current.length > 1)
    stripes.push({ points: current, red: stripes.length % 2 === 0 });
  return stripes;
}
export function gateOrdinals(checkpointCount: number, every: number): number[] {
  if (every <= 0) throw new Error("Gate interval must be positive");
  return Array.from(
    { length: Math.ceil(checkpointCount / every) },
    (_, i) => i * every,
  );
}
export function cameraZoom(speed: number): number {
  const z = visualConfig.stadium.zoom;
  return Math.max(z.minimum, z.rest / (1 + Math.max(0, speed) / z.speedScale));
}

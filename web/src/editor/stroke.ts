import type { Vec2 } from "../sim/types";
import type { EditorConfig } from "./build";
export function strokeToControls(
  points: Vec2[],
  config: EditorConfig,
): { control: Vec2[] | null; problem: "too-short" | "not-closed" | null } {
  const ui = config.ui;
  if (!ui) throw new Error("Editor UI configuration required");
  const distance = (a: Vec2, b: Vec2) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  if (points.some((p) => p.length !== 2 || !p.every(Number.isFinite)))
    return { control: null, problem: "too-short" };
  points = points.filter((p, i) => i === 0 || distance(p, points[i - 1]) > 0);
  let length = 0;
  for (let i = 1; i < points.length; i++)
    length += distance(points[i - 1], points[i]);
  if (length < ui.minStrokePx) return { control: null, problem: "too-short" };
  if (distance(points[0], points[points.length - 1]) > ui.closeDistancePx)
    return { control: null, problem: "not-closed" };
  // Preserve the drawn closing arc. Cutting it at the capture radius introduces
  // a corner at the start, especially on a phone-sized circle. Only remove an
  // endpoint already coincident with the start before closing the polyline.
  const closed = points.slice();
  if (distance(closed[closed.length - 1], closed[0]) < 1e-6) closed.pop();
  closed.push(points[0]);
  const arcs = [0];
  for (let i = 1; i < closed.length; i++)
    arcs.push(arcs[i - 1] + distance(closed[i - 1], closed[i]));
  let j = 1;
  const control = Array.from(
    { length: config.control_points },
    (_, i): Vec2 => {
      const target = (arcs[arcs.length - 1] * i) / config.control_points;
      while (j < arcs.length - 1 && arcs[j] < target) j++;
      const t = (target - arcs[j - 1]) / (arcs[j] - arcs[j - 1]);
      return [
        closed[j - 1][0] + t * (closed[j][0] - closed[j - 1][0]),
        -(closed[j - 1][1] + t * (closed[j][1] - closed[j - 1][1])),
      ];
    },
  );
  return { control, problem: null };
}

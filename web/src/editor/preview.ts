import type { Vec2 } from "../sim/types";
import type { buildEditorTrack } from "./build";
import { kerbStripes } from "../race/stadium";
import config from "./editor-config.json";
export type PreviewView = { scale: number; offset: Vec2 };
type Built = ReturnType<typeof buildEditorTrack>;
export function crossingAt(
  built: Built,
  view: PreviewView,
  point: Vec2,
): number | null {
  const index = (built.track.data.crossings ?? []).findIndex(
    (c) =>
      Math.hypot(
        c.point[0] * view.scale + view.offset[0] - point[0],
        -c.point[1] * view.scale + view.offset[1] - point[1],
      ) <= 20,
  );
  return index < 0 ? null : index;
}
export function drawPreview(
  ctx: CanvasRenderingContext2D,
  built: Built,
  view: PreviewView,
): void {
  const { track, tight } = built,
    d = track.data;
  if (!(view.scale > 0)) return;
  // Fill in screen space: shared controls may be translated far from the origin.
  const density = ctx.getTransform().a,
    canvasWidth = ctx.canvas.width / density,
    canvasHeight = ctx.canvas.height / density,
    band = 10 * view.scale;
  ctx.fillStyle = "#68a842";
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  ctx.fillStyle = "#5f9e3a";
  for (
    let y = (view.offset[1] % (2 * band)) - 2 * band;
    y < canvasHeight;
    y += 2 * band
  )
    ctx.fillRect(0, y, canvasWidth, band);
  ctx.save();
  ctx.translate(...view.offset);
  ctx.scale(view.scale, -view.scale);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  const stroke = (
    points: Vec2[],
    color: string,
    width: number,
    closed = false,
  ) => {
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    if (closed) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  };
  stroke(d.centerline, "#e9e4d6", d.width + 2.4, true);
  stroke(d.centerline, "#5d6470", d.width, true);
  const kerbs = (edge: Vec2[], closed: boolean) => {
    for (const stripe of kerbStripes(edge, 3, closed))
      stroke(stripe.points, stripe.red ? "#d6312c" : "#e9e4d6", 1.2);
  };
  kerbs(d.left, true);
  kerbs(d.right, true);
  for (const c of d.crossings ?? []) {
    if (!Number.isFinite(c.angle) || c.angle < 1) continue;
    const half =
        d.width / Math.sin((c.angle * Math.PI) / 180) + config.zone_margin,
      start = c.arcs[c.over] - half;
    const center: Vec2[] = [],
      left: Vec2[] = [],
      right: Vec2[] = [];
    for (let i = 0; i <= 32; i++) {
      const [p, a] = track.sample(start + (2 * half * i) / 32);
      center.push(p);
      left.push([
        p[0] - (Math.sin(a) * d.width) / 2,
        p[1] + (Math.cos(a) * d.width) / 2,
      ]);
      right.push([
        p[0] + (Math.sin(a) * d.width) / 2,
        p[1] - (Math.cos(a) * d.width) / 2,
      ]);
    }
    ctx.save();
    ctx.lineCap = "butt";
    ctx.shadowColor = "#18201799";
    ctx.shadowBlur = 5 * view.scale;
    ctx.shadowOffsetX = 3 * view.scale;
    ctx.shadowOffsetY = 4 * view.scale;
    stroke(center, "#e9e4d6", d.width + 2.4);
    ctx.restore();
    ctx.save();
    ctx.lineCap = "butt";
    stroke(center, "#5d6470", d.width);
    kerbs(left, false);
    kerbs(right, false);
    ctx.restore();
  }
  for (const [start, end] of tight) {
    const points: Vec2[] = [];
    for (let a = start; a <= end + 1; a += 0.5) points.push(track.sample(a)[0]);
    stroke(points, "#ff3d39", d.width * 0.3);
  }
  const [start, heading] = track.sample(0);
  ctx.save();
  ctx.translate(...start);
  ctx.rotate(heading);
  const cell = d.width / 6;
  for (let row = 0; row < 6; row++)
    for (let col = 0; col < 2; col++) {
      ctx.fillStyle = (row + col) % 2 ? "#171b18" : "#fff";
      ctx.fillRect((col - 1) * cell, -d.width / 2 + row * cell, cell, cell);
    }
  ctx.restore();
  const [arrow, a] = track.sample(16);
  ctx.save();
  ctx.translate(...arrow);
  ctx.rotate(a);
  ctx.beginPath();
  ctx.moveTo(-3, -2);
  ctx.lineTo(1, 0);
  ctx.lineTo(-3, 2);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
  ctx.restore();
}

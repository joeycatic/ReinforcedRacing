import { Application, Container, Graphics } from "pixi.js";
import type { CarState, Vec2 } from "../sim/types";
import { speed } from "../sim/physics";
import { Track } from "../sim/track";
import { interpolate, type RaceEngine } from "./engine";

const PLAYER = 0x75f4c2,
  AI = 0xffaa73;
const line = (
  graphics: Graphics,
  points: Vec2[],
  color: number,
  width: number,
  alpha = 1,
): void => {
  graphics.moveTo(points[0][0], -points[0][1]);
  for (const point of points.slice(1)) graphics.lineTo(point[0], -point[1]);
  graphics.closePath().stroke({ color, width, alpha, join: "round" });
};
function car(color: number, ghost = false): Container {
  const body = new Container();
  body.addChild(
    new Graphics()
      .roundRect(-2.3, -1.15, 4.6, 2.3, 0.4)
      .fill({ color: 0x000000, alpha: 0.35 }),
  );
  const shape = new Graphics();
  for (const x of [-1.4, 1.2])
    for (const y of [-1.05, 0.75])
      shape.rect(x - 0.4, y, 0.8, 0.3).fill(0x080d12);
  shape
    .poly([
      -2, -0.85, 1.25, -0.85, 2.15, -0.58, 2.15, 0.58, 1.25, 0.85, -2, 0.85,
    ])
    .fill(color);
  shape.poly([-0.7, -0.64, 0.55, -0.6, 0.85, 0.6, -0.7, 0.64]).fill(0x1c3437);
  shape.rect(-1.85, -0.93, 0.25, 1.86).fill(0xeff9ef);
  shape
    .rect(1.9, -0.63, 0.2, 0.38)
    .fill(0xf5ffe9)
    .rect(1.9, 0.25, 0.2, 0.38)
    .fill(0xf5ffe9);
  body.addChild(shape);
  if (ghost) body.alpha = 0.28;
  return body;
}
export class RaceScene {
  readonly app = new Application();
  private world = new Container();
  private player = car(PLAYER);
  private ai = car(AI);
  private ghost = car(0xe7f3e9, true);
  private terrain = new Container();
  private marks = new Graphics();
  private marksCount = 0;
  private camera = { x: 0, y: 0, zoom: 1 };
  private track: Track | null = null;
  private observer: ResizeObserver;
  private reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  constructor(private readonly host: HTMLElement) {
    this.observer = new ResizeObserver(() =>
      this.app.renderer?.resize(host.clientWidth, host.clientHeight),
    );
  }
  async init(): Promise<void> {
    await this.app.init({
      width: this.host.clientWidth,
      height: this.host.clientHeight,
      background: 0x131e22,
      antialias: false,
      resolution: Math.min(devicePixelRatio, 2),
      autoDensity: true,
      preference: "webgl",
      autoStart: false,
    });
    this.app.canvas.setAttribute(
      "aria-label",
      "Race circuit. Your car is mint; the live AI is orange.",
    );
    this.app.canvas.setAttribute("role", "img");
    this.host.append(this.app.canvas);
    this.app.stage.addChild(this.world);
    this.world.addChild(
      this.terrain,
      this.marks,
      this.ghost,
      this.ai,
      this.player,
    );
    this.observer.observe(this.host);
  }
  setTrack(track: Track): void {
    this.track = track;
    this.terrain.removeChildren().forEach((child) => child.destroy());
    this.clearMarks();
    const d = track.data,
      bg = new Graphics();
    const xs = d.centerline.map((p) => p[0]),
      ys = d.centerline.map((p) => p[1]);
    const lowX = Math.min(...xs) - 100,
      highX = Math.max(...xs) + 100,
      lowY = Math.min(...ys) - 100,
      highY = Math.max(...ys) + 100;
    for (let x = Math.floor(lowX / 20) * 20; x < highX; x += 20)
      bg.moveTo(x, -lowY).lineTo(x, -highY);
    for (let y = Math.floor(lowY / 20) * 20; y < highY; y += 20)
      bg.moveTo(lowX, -y).lineTo(highX, -y);
    bg.stroke({ width: 0.12, color: 0x2c3e40, alpha: 0.35 });
    this.terrain.addChild(bg);
    const route = new Graphics();
    line(route, d.centerline, 0x0c1419, d.width + 4);
    line(route, d.centerline, 0x84948d, d.width + 1.3);
    line(route, d.centerline, 0x293237, d.width);
    // Narrow edge marks read at speed without obscuring the road.
    for (const edge of [d.left, d.right])
      for (let i = 0; i < edge.length; i += 8) {
        route.moveTo(edge[i][0], -edge[i][1]);
        for (let k = 1; k <= 4; k++) {
          const p = edge[(i + k) % edge.length];
          route.lineTo(p[0], -p[1]);
        }
        route.stroke({ width: 0.45, color: 0xced6c8, alpha: 0.8 });
      }
    this.terrain.addChild(route);
    const [start, heading] = track.sample(0),
      grid = new Container();
    grid.position.set(start[0], -start[1]);
    grid.rotation = -heading;
    const checkers = new Graphics(),
      size = d.width / 10;
    for (let x = 0; x < 2; x++)
      for (let y = 0; y < 10; y++)
        checkers
          .rect((x - 1) * size, (y - 5) * size, size, size)
          .fill((x + y) % 2 ? 0x233035 : 0xe3e9d9);
    grid.addChild(checkers);
    this.terrain.addChild(grid);
    const bounds = this.bounds();
    this.camera = { x: bounds.x, y: bounds.y, zoom: this.fitZoom() };
  }
  clearMarks(): void {
    this.marks.clear();
    this.marksCount = 0;
  }
  private bounds(): { x: number; y: number; width: number; height: number } {
    const points = this.track!.data.centerline,
      xs = points.map((p) => p[0]),
      ys = points.map((p) => -p[1]);
    return {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: (Math.min(...ys) + Math.max(...ys)) / 2,
      width: Math.max(...xs) - Math.min(...xs) + this.track!.data.width * 2,
      height: Math.max(...ys) - Math.min(...ys) + this.track!.data.width * 2,
    };
  }
  private fitZoom(): number {
    const b = this.bounds();
    return Math.min(
      (this.host.clientWidth - 100) / b.width,
      (this.host.clientHeight - 130) / b.height,
    );
  }
  render(
    engine: RaceEngine,
    alpha: number,
    seconds: number,
    ghostFrame: number[] | null,
  ): void {
    if (!this.track) return;
    const player = interpolate(
        engine.previousPlayer,
        engine.player.state,
        alpha,
      ),
      ai = interpolate(engine.previousAi, engine.ai.state, alpha);
    for (const [sprite, state] of [
      [this.player, player],
      [this.ai, ai],
    ] as const) {
      sprite.position.set(state.x, -state.y);
      sprite.rotation = -state.psi;
    }
    this.ghost.visible = ghostFrame !== null;
    if (ghostFrame) {
      this.ghost.position.set(ghostFrame[0], -ghostFrame[1]);
      this.ghost.rotation = -ghostFrame[2];
    }
    if (
      engine.phase === "racing" &&
      Math.abs(player.vy) > 3 &&
      this.marksCount < 5000
    ) {
      this.marks
        .moveTo(engine.previousPlayer.x, -engine.previousPlayer.y)
        .lineTo(player.x, -player.y)
        .stroke({ color: 0x0b1115, width: 1.1, alpha: 0.23 });
      this.marksCount++;
    }
    const overview = engine.phase === "ready";
    const bounds = this.bounds();
    const zoom = overview
      ? this.fitZoom()
      : Math.max(3.1, 7.5 / (1 + speed(player) / 70));
    const response =
      overview || this.reduced ? 1 : 1 - Math.exp(-Math.max(0, seconds) * 5);
    const ahead = overview ? 0 : 7;
    const x = overview ? bounds.x : player.x + Math.cos(player.psi) * ahead;
    const y = overview ? bounds.y : -player.y - Math.sin(player.psi) * ahead;
    this.camera.x += (x - this.camera.x) * response;
    this.camera.y += (y - this.camera.y) * response;
    this.camera.zoom += (zoom - this.camera.zoom) * response;
    this.world.pivot.set(this.camera.x, this.camera.y);
    this.world.position.set(
      this.host.clientWidth / 2,
      this.host.clientHeight * 0.53,
    );
    this.world.scale.set(this.camera.zoom);
    this.app.render();
  }
  destroy(): void {
    this.observer.disconnect();
    this.app.destroy(true, { children: true });
  }
}
export function minimapPath(track: Track): { path: string; viewBox: string } {
  const points = track.data.centerline,
    xs = points.map((p) => p[0]),
    ys = points.map((p) => -p[1]);
  const x = Math.min(...xs) - 20,
    y = Math.min(...ys) - 20,
    width = Math.max(...xs) - x + 20,
    height = Math.max(...ys) - y + 20;
  return {
    path:
      points.map((p, i) => `${i ? "L" : "M"}${p[0]},${-p[1]}`).join(" ") + " Z",
    viewBox: `${x} ${y} ${width} ${height}`,
  };
}

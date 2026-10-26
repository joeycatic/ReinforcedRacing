import type { AttemptRun, WatchTimeline } from "../watch/timeline";
import { Application, Container, Graphics, Sprite, Texture } from "pixi.js";
import type { CarState, Vec2 } from "../sim/types";
import { cameraZoom, kerbStripes, gateOrdinals } from "./stadium";
import { speed } from "../sim/physics";
import { Track } from "../sim/track";
import { interpolate, type RaceEngine } from "./engine";

import { visualConfig } from "../viewer/visual-config";
import type { Heatmap } from "../assets/load";
import { heatmapPixels, type HeatmapScale } from "../viewer/heatmap";
import { snapshotColor } from "../watch/timeline";

import { formulaCar, LIVERIES, CAR_PIXELS_PER_METER } from "./car";
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
export class RaceScene {
  follow: "player" | "ai" = "player";
  playerVisible = true;
  readonly app = new Application();
  private world = new Container();
  private player!: Sprite;
  private ai!: Sprite;
  private ghost!: Sprite;
  private terrain = new Container();
  private heatLayer = new Container();
  private heatTextures: Texture[] = [];
  private attempts = new Container();
  private attemptRuns: AttemptRun[] = [];
  private attemptTrails: Graphics[] = [];
  private attemptCars: Sprite[] = [];
  private partialTrail = new Graphics();
  private partialKey = "";
  private particles = new Graphics();
  private sensors = new Graphics();
  showSensors = false;
  private sparks: {
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
  }[] = [];
  private lastDriftTick = -1;

  private marks = new Container();
  private markChunk: Graphics | null = null;
  private markChunkCount = 0;
  private camera = { x: 0, y: 0, zoom: 1 };
  private track: Track | null = null;
  private trackBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null = null;
  private dirty = true;
  private slowFrames = 0;
  private frameSeconds = 1 / 60;
  private lastView: "race" | null = null;
  private lastSensors = false;
  private observer: ResizeObserver;
  private motion = matchMedia("(prefers-reduced-motion: reduce)");
  private reduced = this.motion.matches;
  private onMotionChange = (event: MediaQueryListEvent): void => {
    this.reduced = event.matches;
    this.dirty = true;
    if (this.reduced) {
      this.sparks = [];
      this.particles.clear();
    }
  };
  constructor(private readonly host: HTMLElement) {
    this.observer = new ResizeObserver(() => {
      this.app.renderer?.resize(host.clientWidth, host.clientHeight);
      this.dirty = true;
    });
    this.motion.addEventListener("change", this.onMotionChange);
  }
  async init(): Promise<void> {
    await this.app.init({
      width: this.host.clientWidth,
      height: this.host.clientHeight,
      background: visualConfig.stadium.grass[1],
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
    this.player = formulaCar(LIVERIES.player, this.app.renderer);
    this.ai = formulaCar(LIVERIES.ai, this.app.renderer);
    this.ghost = formulaCar(
      { ...LIVERIES.ai, body: 0xe7f3e9 },
      this.app.renderer,
    );
    this.ghost.alpha = 0.28;
    this.world.addChild(
      this.terrain,
      this.heatLayer,
      this.marks,
      this.particles,
      this.sensors,
      this.attempts,
      this.ghost,
      this.ai,
      this.player,
    );
    this.observer.observe(this.host);
  }
  setTrack(track: Track): void {
    this.track = track;
    this.terrain.cacheAsTexture(false);
    this.terrain.removeChildren().forEach((child) => child.destroy());
    this.clearMarks();
    this.setHeatmap(null, false, false, { visits: 0, crashes: 0 });
    const d = track.data,
      bg = new Graphics();
    const xs = d.centerline.map((p) => p[0]),
      ys = d.centerline.map((p) => p[1]);
    this.trackBounds = {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: -(Math.min(...ys) + Math.max(...ys)) / 2,
      width: Math.max(...xs) - Math.min(...xs) + d.width * 2,
      height: Math.max(...ys) - Math.min(...ys) + d.width * 2,
    };
    const lowX = Math.min(...xs) - 100,
      highX = Math.max(...xs) + 100,
      lowY = Math.min(...ys) - 100,
      highY = Math.max(...ys) + 100;
    const stadium = visualConfig.stadium;
    for (
      let y = Math.floor(lowY / stadium.grassBand) * stadium.grassBand;
      y < highY;
      y += stadium.grassBand
    )
      bg.rect(
        lowX,
        -y - stadium.grassBand,
        highX - lowX,
        stadium.grassBand,
      ).fill(stadium.grass[Math.abs(Math.round(y / stadium.grassBand)) % 2]);
    this.terrain.addChild(bg);
    const route = new Graphics();
    line(route, d.centerline, 0x46792e, d.width + 4);
    line(route, d.centerline, stadium.tarmac, d.width);
    for (const edge of [d.left, d.right])
      for (const stripe of kerbStripes(edge, stadium.kerbStripe)) {
        route.moveTo(stripe.points[0][0], -stripe.points[0][1]);
        for (const p of stripe.points.slice(1)) route.lineTo(p[0], -p[1]);
        route.stroke({
          color: stadium.kerb[stripe.red ? 0 : 1],
          width: stadium.kerbWidth,
        });
      }
    for (let arc = 0; arc < track.length; arc += 12) {
      const [a] = track.sample(arc),
        [b] = track.sample(arc + 3);
      route
        .moveTo(a[0], -a[1])
        .lineTo(b[0], -b[1])
        .stroke({ color: 0xe9e4d6, width: 0.16, alpha: 0.65 });
    }
    for (const ordinal of gateOrdinals(
      d.checkpoints.length,
      stadium.gateEvery,
    )) {
      const [p, heading] = track.sample(d.checkpoints[ordinal]);
      const nx = -Math.sin(heading),
        ny = Math.cos(heading),
        half = d.width / 2;
      route
        .moveTo(p[0] - nx * half, -p[1] + ny * half)
        .lineTo(p[0] + nx * half, -p[1] - ny * half)
        .stroke({ color: stadium.gate, width: 0.55, alpha: 0.85 });
      for (const side of [-1, 1])
        route
          .circle(
            p[0] + side * nx * (half + 0.7),
            -p[1] - side * ny * (half + 0.7),
            0.8,
          )
          .fill(stadium.gate);
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
    this.terrain.cacheAsTexture({
      resolution: visualConfig.rendering.terrainResolution,
      antialias: false,
    });
    const bounds = this.bounds();
    this.camera = { x: bounds.x, y: bounds.y, zoom: this.fitZoom() };
  }
  clearMarks(): void {
    this.dirty = true;
    this.marks.removeChildren().forEach((child) => child.destroy());
    this.markChunk = null;
    this.markChunkCount = 0;
    this.lastDriftTick = -1;
    this.sparks = [];
    this.particles.clear();
  }
  private bounds(): { x: number; y: number; width: number; height: number } {
    return this.trackBounds!;
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
    // Keep the full-screen view responsive on slow renderers without changing dt.
    if (engine.phase === "countdown" || engine.phase === "racing") {
      const quality = visualConfig.rendering;
      this.frameSeconds =
        (1 - quality.averageWeight) * this.frameSeconds +
        quality.averageWeight * seconds;
      this.slowFrames =
        this.frameSeconds > 1 / quality.targetFps ? this.slowFrames + 1 : 0;
      if (
        this.slowFrames >= quality.slowFrames &&
        this.app.renderer.resolution > quality.minimumResolution
      ) {
        this.app.renderer.resize(
          this.host.clientWidth,
          this.host.clientHeight,
          Math.max(
            quality.minimumResolution,
            this.app.renderer.resolution * quality.resolutionScale,
          ),
        );
        this.slowFrames = 0;
        this.frameSeconds = 1 / 60;
        this.dirty = true;
      }
    }
    this.attempts.visible = false;
    this.player.visible = this.playerVisible;
    this.ai.visible = true;
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
    this.drift(engine, seconds);
    this.sensors.clear();
    if (this.showSensors && engine.lastObservation.length === 22) {
      const state = engine.observationState;
      engine.observations.rayAngles.forEach((angle, index) => {
        const distance =
          engine.lastObservation[index] *
          engine.config.observation.ray_max_distance;
        this.sensors
          .moveTo(state.x, -state.y)
          .lineTo(
            state.x + Math.cos(state.psi + angle) * distance,
            -state.y - Math.sin(state.psi + angle) * distance,
          )
          .stroke({ color: 0xffbc87, width: 0.18, alpha: 0.6 });
      });
    }
    const target = this.follow === "ai" ? ai : player;
    const overview = engine.phase === "ready";
    const bounds = this.bounds();
    const zoom = overview ? this.fitZoom() : cameraZoom(speed(target));
    const response =
      overview || this.reduced ? 1 : 1 - Math.exp(-Math.max(0, seconds) * 5);
    const ahead = overview ? 0 : 7;
    const x = overview ? bounds.x : target.x + Math.cos(target.psi) * ahead;
    const y = overview ? bounds.y : -target.y - Math.sin(target.psi) * ahead;
    const moving =
      Math.abs(x - this.camera.x) > 1e-4 ||
      Math.abs(y - this.camera.y) > 1e-4 ||
      Math.abs(zoom - this.camera.zoom) > 1e-5;
    this.camera.x += (x - this.camera.x) * response;
    this.camera.y += (y - this.camera.y) * response;
    this.camera.zoom += (zoom - this.camera.zoom) * response;
    this.world.pivot.set(this.camera.x, this.camera.y);
    this.world.position.set(
      this.host.clientWidth / 2,
      this.host.clientHeight * 0.53,
    );
    this.world.scale.set(this.camera.zoom);
    if (
      this.dirty ||
      moving ||
      engine.phase === "racing" ||
      this.lastView !== "race" ||
      this.lastSensors !== this.showSensors
    )
      this.app.render();
    this.lastView = "race";
    this.lastSensors = this.showSensors;
    this.dirty = false;
  }
  setHeatmap(
    map: Heatmap | null,
    visits: boolean,
    crashes: boolean,
    scale: HeatmapScale,
  ): void {
    this.dirty = true;
    this.heatLayer.removeChildren().forEach((child) => child.destroy());
    this.heatTextures.forEach((texture) => texture.destroy(true));
    this.heatTextures = [];
    if (!map) return;
    const layers = (
      [
        [visits, "visits", scale.visits],
        [crashes, "crash_bins", scale.crashes],
      ] as const
    ).filter(([enabled, layer]) => enabled && map[layer].length);
    if (layers.length) {
      let minX = map.grid.width,
        minY = map.grid.height,
        maxX = 0,
        maxY = 0;
      for (const [, layer] of layers)
        for (const [index] of map[layer]) {
          const x = index % map.grid.width,
            y = Math.floor(index / map.grid.width);
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      const canvas = document.createElement("canvas");
      canvas.width = maxX - minX + 1;
      canvas.height = maxY - minY + 1;
      const ctx = canvas.getContext("2d")!;
      for (const [, layer, peak] of layers) {
        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = canvas.width;
        layerCanvas.height = canvas.height;
        const layerCtx = layerCanvas.getContext("2d")!,
          image = layerCtx.createImageData(map.grid.width, map.grid.height);
        image.data.set(heatmapPixels(map, layer, peak));
        layerCtx.putImageData(image, -minX, -(map.grid.height - 1 - maxY));
        ctx.drawImage(layerCanvas, 0, 0);
      }
      const texture = Texture.from(canvas);
      texture.source.scaleMode = "nearest";
      this.heatTextures.push(texture);
      const sprite = new Sprite(texture);
      sprite.position.set(
        map.grid.origin[0] + minX * map.grid.cell_size,
        -(map.grid.origin[1] + (maxY + 1) * map.grid.cell_size),
      );
      sprite.scale.set(map.grid.cell_size);
      this.heatLayer.addChild(sprite);
    }
    if (crashes) {
      const points = new Graphics();
      for (const [x, y] of map.crash_positions)
        points.circle(x, -y, 0.9).fill({ color: 0xff786b, alpha: 0.75 });
      this.heatLayer.addChild(points);
    }
  }
  /** Each path is open: never connect an attempt's final point to its start. */
  private drawTrail(graphics: Graphics, index: number, tick: number): void {
    graphics.clear();
    const run = this.attemptRuns[index];
    const end = Math.min(run.frames.length - 1, Math.max(0, tick));
    const first = run.frames[0];
    graphics.moveTo(first[0], -first[1]);
    for (
      let i = visualConfig.watch.trailEveryTicks;
      i < end;
      i += visualConfig.watch.trailEveryTicks
    ) {
      const frame = run.frames[i];
      graphics.lineTo(frame[0], -frame[1]);
    }
    const frame = run.frames[Math.floor(end)];
    graphics.lineTo(frame[0], -frame[1]).stroke({
      width: visualConfig.watch.trailWidth,
      color: snapshotColor(index, this.attemptRuns.length),
      alpha: visualConfig.watch.trailAlpha,
      join: "round",
      cap: "round",
    });
  }
  setAttempts(runs: AttemptRun[]): void {
    this.player.visible = this.ai.visible = this.ghost.visible = false;
    this.attempts.removeChildren().forEach((child) => child.destroy());
    this.attemptRuns = runs;
    this.attemptCars = [];
    this.partialTrail = new Graphics();
    this.partialKey = "";
    this.attemptTrails = runs.map((run, index) => {
      const trail = new Graphics();
      trail.label = `trail-${index}`;
      const seconds = (run.frames.length - 1) * run.data.dt;
      const cut =
        run.attempt.outcome === "stalled"
          ? Math.min(seconds, visualConfig.watch.stallSeconds)
          : seconds;
      this.drawTrail(trail, index, cut / run.data.dt);
      trail.visible = false;
      this.attempts.addChild(trail);
      return trail;
    });
    this.attempts.addChild(this.partialTrail);
    runs.forEach((_, index) => {
      const color = snapshotColor(index, runs.length);
      const car = formulaCar(
        {
          ...LIVERIES.ai,
          body: color,
          light: color,
          number: String(index + 1),
        },
        this.app.renderer,
      );
      car.label = "attempt-car";
      car.visible = false;
      this.attemptCars.push(car);
      this.attempts.addChild(car);
    });
    this.dirty = true;
  }
  renderAttempt(timeline: WatchTimeline): void {
    if (!this.track) return;
    this.player.visible = this.ai.visible = this.ghost.visible = false;
    this.attempts.visible = true;
    this.particles.clear();
    this.sensors.clear();
    const current = timeline.current();
    const earlier = new Set(timeline.trails());
    this.attemptTrails.forEach((trail, i) => {
      trail.visible = earlier.has(i);
    });
    this.partialTrail.label = `trail-${current.index}`;
    const tick =
      current.replaySeconds / this.attemptRuns[current.index].data.dt;
    const key = `${current.index}:${tick}`;
    if (key !== this.partialKey) {
      this.drawTrail(this.partialTrail, current.index, tick);
      this.partialKey = key;
    }
    const bounds = this.bounds(),
      zoom = this.fitZoom();
    this.world.pivot.set(bounds.x, bounds.y);
    this.world.position.set(
      this.host.clientWidth / 2,
      this.host.clientHeight * 0.53,
    );
    this.world.scale.set(zoom);
    this.camera = { x: bounds.x, y: bounds.y, zoom };
    this.attemptCars.forEach((car, index) => {
      car.visible = index === current.index && !current.holding;
      if (!car.visible) return;
      car.position.set(current.frame[0], -current.frame[1]);
      car.rotation = -current.frame[2];
      car.scale.set(
        Math.max(1, visualConfig.watch.minimumPixelsPerMeter / zoom) /
          CAR_PIXELS_PER_METER,
      );
    });
    this.app.render();
    // Invalidate the race render cache when drawing a Watch frame.
    this.lastView = null;
    this.dirty = false;
  }
  private drift(engine: RaceEngine, seconds: number): void {
    const active = engine.phase === "racing",
      c = visualConfig.drift;
    if (active && engine.player.ticks !== this.lastDriftTick) {
      this.lastDriftTick = engine.player.ticks;
      for (const [state, previous] of [
        [engine.player.state, engine.previousPlayer],
        [engine.ai.state, engine.previousAi],
      ]) {
        if (
          Math.abs(state.vy) <= c.slipThreshold ||
          Math.hypot(state.x - previous.x, state.y - previous.y) > 4
        )
          continue;
        {
          if (!this.markChunk || this.markChunkCount >= 48) {
            this.markChunk =
              this.marks.children.length >= Math.ceil(c.maxMarkPairs / 48)
                ? (this.marks.removeChildAt(0) as Graphics)
                : new Graphics();
            this.markChunk.clear();
            this.marks.addChild(this.markChunk);
            this.markChunkCount = 0;
          }
          for (const side of [-0.72, 0.72]) {
            const wheel = (s: CarState): Vec2 => [
              s.x - 1.4 * Math.cos(s.psi) - side * Math.sin(s.psi),
              s.y - 1.4 * Math.sin(s.psi) + side * Math.cos(s.psi),
            ];
            const a = wheel(previous),
              b = wheel(state);
            this.markChunk!.moveTo(a[0], -a[1])
              .lineTo(b[0], -b[1])
              .stroke({ color: 0x091114, width: c.markWidth, alpha: 0.5 });
          }
          this.markChunkCount++;
        }
        if (!this.reduced && this.sparks.length < c.maxParticles) {
          const phase = engine.player.ticks * 2.3999632297;
          this.sparks.push({
            x: state.x - 1.5 * Math.cos(state.psi),
            y: state.y - 1.5 * Math.sin(state.psi),
            vx: Math.cos(phase) * c.particleVelocity,
            vy: Math.sin(phase) * c.particleVelocity,
            life: c.particleLifetime,
          });
        }
      }
    }
    this.particles.clear();
    const dt = active ? Math.min(seconds, 0.05) : 0;
    this.sparks = this.sparks.filter((p) => {
      p.life -= dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      if (p.life <= 0) return false;
      this.particles
        .circle(p.x, -p.y, 0.12 + 0.2 * (1 - p.life / c.particleLifetime))
        .fill({ color: 0xd1d8c3, alpha: (p.life / c.particleLifetime) * 0.35 });
      return true;
    });
  }
  destroy(): void {
    this.heatTextures.forEach((texture) => texture.destroy(true));
    this.motion.removeEventListener("change", this.onMotionChange);
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

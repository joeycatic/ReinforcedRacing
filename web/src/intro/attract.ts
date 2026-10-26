import { AssetLoader, type SimulationConfig } from "../assets/load";
import { RaceEngine, FixedClock } from "../race/engine";
import { RaceScene } from "../race/scene";
import { OnnxPolicy } from "../race/policy";
import type { Track } from "../sim/track";

export class AttractLoop {
  private scene: RaceScene;
  private engine?: RaceEngine;
  private clock?: FixedClock;
  private policy?: OnnxPolicy;
  private stopped = false;
  private activating = false;
  private initialized = false;
  private visible = true;
  private frameId = 0;
  private lastFrame = 0;
  private pending: Promise<void> | null = null;
  private observer: IntersectionObserver;
  private motion = matchMedia("(prefers-reduced-motion: reduce)");
  private onVisibility = (): void => {
    if (!this.engine || !this.clock || this.stopped) return;
    if (document.hidden || !this.visible || this.motion.matches) {
      this.engine.pause();
      this.clock.clear();
      cancelAnimationFrame(this.frameId);
      this.frameId = 0;
    } else if (this.policy) {
      this.engine.resume();
      this.lastFrame = performance.now();
      if (!this.frameId && !this.pending)
        this.frameId = requestAnimationFrame(this.frame);
    }
  };
  private onMotion = (): void => {
    if (this.motion.matches && this.engine) {
      this.engine.reset();
      this.scene.render(this.engine, 1, 0, null);
      this.status("STATIC · reduced motion");
    } else if (!this.policy) {
      void this.activate();
    } else if (this.engine) {
      this.engine.launch();
      this.status("LIVE · final AI lapping Broad Sweep");
    }
    this.onVisibility();
  };
  constructor(
    private readonly host: HTMLElement,
    private readonly loader: AssetLoader,
    private readonly config: SimulationConfig,
    private readonly track: Track,
  ) {
    this.scene = new RaceScene(host);
    this.scene.follow = "ai";
    this.scene.playerVisible = false;
    this.observer = new IntersectionObserver((entries) => {
      this.visible = entries[0].isIntersecting;
      this.onVisibility();
    });
  }
  private status(text: string): void {
    document.getElementById("hero-status")!.textContent = text;
  }
  async start(): Promise<void> {
    try {
      await this.scene.init();
      if (this.stopped) {
        this.scene.destroy();
        return;
      }
      this.scene.setTrack(this.track);
      this.initialized = true;
      this.engine = new RaceEngine(this.track, this.config, {
        predict: async () => ({ action: [0, 0], value: 0 }),
      });
      this.clock = new FixedClock(this.engine);
      this.scene.render(this.engine, 1, 0, null);
      this.host.querySelector(".static-track")?.setAttribute("hidden", "");
      this.observer.observe(this.host);
      document.addEventListener("visibilitychange", this.onVisibility);
      this.motion.addEventListener("change", this.onMotion);
      if (this.motion.matches) {
        this.status("STATIC · reduced motion");
      } else {
        await this.activate();
      }
    } catch (cause) {
      this.status("STATIC · live preview unavailable");
      console.warn("Live hero unavailable; keeping the static circuit", cause);
    }
  }
  private async activate(): Promise<void> {
    if (this.activating || this.stopped) return;
    this.activating = true;
    try {
      const author = this.loader.manifest.medals.find(
        (m) => m.medal === "author",
      )!;
      const policy = await OnnxPolicy.create(
        await this.loader.bytes(author.model),
      );
      if (this.stopped) {
        await policy.release();
        return;
      }
      this.policy = policy;
      this.engine = new RaceEngine(this.track, this.config, policy);
      this.clock = new FixedClock(this.engine);
      this.engine.launch();
      this.onMotion();
    } catch (cause) {
      if (!this.stopped) this.status("STATIC · live preview unavailable");
      console.warn("Live hero unavailable; keeping the static circuit", cause);
    } finally {
      this.activating = false;
    }
  }
  private frame = (now: number): void => {
    this.frameId = 0;
    if (this.stopped || !this.engine || !this.clock) return;
    const elapsed = Math.min(0.1, Math.max(0, (now - this.lastFrame) / 1000));
    this.lastFrame = now;
    this.pending = this.clock.advance(elapsed, [0, 0]);
    void this.pending
      .then(() => {
        this.pending = null;
        if (this.stopped) return;
        if (this.engine!.aiLapTime !== null) {
          this.engine!.launch();
          this.scene.clearMarks();
          this.clock!.clear();
        }
        this.scene.render(this.engine!, this.clock!.alpha, elapsed, null);
        if (!document.hidden && this.visible && !this.motion.matches)
          this.frameId = requestAnimationFrame(this.frame);
      })
      .catch((cause) => {
        this.pending = null;
        this.status("STATIC · live preview unavailable");
        console.warn("Hero stopped", cause);
      });
  };
  stop(): void {
    this.stopped = true;
    cancelAnimationFrame(this.frameId);
    this.engine?.pause();
    this.observer.disconnect();
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.motion.removeEventListener("change", this.onMotion);
    void Promise.resolve(this.pending)
      .catch(() => {})
      .then(async () => {
        await this.policy?.release();
        if (this.initialized) {
          this.scene.destroy();
          this.initialized = false;
        }
      });
  }
}

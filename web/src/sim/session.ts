import type { Action, CarState, PhysicsConfig, TrackConfig } from "./types";
import { Track } from "./track";
import { step } from "./physics";
import { ProgressTracker } from "./progress";

export class DriveSession {
  progress!: ProgressTracker;
  state!: CarState;
  ticks = 0;
  respawn_ticks = 0;
  crashed = false;
  crashes = 0;
  terminated = false;
  constructor(
    readonly track: Track,
    readonly physics: PhysicsConfig,
    readonly tracks: TrackConfig,
    readonly mode: "play" | "training" = "play",
  ) {
    this.restart();
  }
  restart(): void {
    this.progress = new ProgressTracker(this.track, this.tracks);
    this.respawn();
    this.ticks = 0;
    this.respawn_ticks = 0;
    this.crashed = false;
    this.crashes = 0;
    this.terminated = false;
  }
  get time(): number {
    return this.ticks * this.physics.dt;
  }
  get lap_time(): number {
    return this.time - this.progress.lap_started_at;
  }
  respawn(): void {
    const [x, y, psi] = this.progress.respawn();
    this.state = { x, y, psi, vx: 0, vy: 0, r: 0, steering: 0 };
    this.respawn_ticks = 0;
    this.crashed = false;
  }
  tick(action: Action): void {
    if (this.terminated) return;
    this.ticks++;
    if (this.respawn_ticks) {
      this.respawn_ticks--;
      if (!this.respawn_ticks) this.respawn();
      return;
    }
    this.state = step(this.state, action, this.physics);
    this.progress.update(
      this.state.x,
      this.state.y,
      this.time,
      this.physics.dt,
    );
    if (
      this.progress.projection.distance >
      (this.track.data.width + this.physics.car_width) / 2
    ) {
      this.crashed = true;
      this.crashes++;
      if (this.mode === "training") this.terminated = true;
      else
        this.respawn_ticks = Math.ceil(
          this.physics.respawn_delay / this.physics.dt,
        );
    }
  }
}

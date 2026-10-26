import type { TrackConfig } from "./types";
import { Track, project, modulo, searchSorted, type Projection } from "./track";

export class ProgressTracker {
  projection: Projection;
  unwrapped = 0;
  delta = 0;
  wrong_way = false;
  laps = 0;
  lap_times: number[] = [];
  lap_started_at = 0;
  last_checkpoint = 0;
  next_checkpoint = 1;
  constructor(
    readonly track: Track,
    readonly config: TrackConfig,
  ) {
    this.projection = project(
      track,
      ...track.data.centerline[0],
      null,
      config.projection_window,
    );
  }
  checkpointArc(ordinal: number): number {
    const count = this.track.data.checkpoints.length;
    return (
      Math.floor(ordinal / count) * this.track.length +
      this.track.data.checkpoints[ordinal % count]
    );
  }
  update(x: number, y: number, tickEndTime: number, dt: number): void {
    const projection = project(
      this.track,
      x,
      y,
      this.projection.index,
      this.config.projection_window,
    );
    let delta = projection.arc - this.projection.arc;
    delta =
      modulo(delta + this.track.length / 2, this.track.length) -
      this.track.length / 2;
    const previous = this.unwrapped;
    this.unwrapped += delta;
    this.delta = delta;
    this.wrong_way = delta < -this.config.wrong_way_min_delta;
    this.projection = projection;
    while (
      delta > 0 &&
      this.unwrapped >= this.checkpointArc(this.next_checkpoint)
    ) {
      const checkpoint = this.checkpointArc(this.next_checkpoint);
      this.last_checkpoint = checkpoint;
      if (this.next_checkpoint % this.track.data.checkpoints.length === 0) {
        const fraction = (checkpoint - previous) / delta;
        const crossedAt = tickEndTime - dt + fraction * dt;
        this.lap_times.push(crossedAt - this.lap_started_at);
        this.lap_started_at = crossedAt;
        this.laps++;
      }
      this.next_checkpoint++;
    }
  }
  respawn(): [number, number, number] {
    const [position, heading] = this.track.sample(this.last_checkpoint);
    this.unwrapped = this.last_checkpoint;
    const index =
      searchSorted(
        this.track.data.arc_length,
        modulo(this.last_checkpoint, this.track.length),
      ) % this.track.data.centerline.length;
    this.projection = project(
      this.track,
      ...position,
      index,
      this.config.projection_window,
    );
    this.delta = 0;
    this.wrong_way = false;
    return [...position, heading];
  }
  get checkpoint_index(): number {
    return (this.next_checkpoint - 1) % this.track.data.checkpoints.length;
  }
  get lap_fraction(): number {
    return (
      this.unwrapped / this.track.length -
      Math.floor(this.unwrapped / this.track.length)
    );
  }
}

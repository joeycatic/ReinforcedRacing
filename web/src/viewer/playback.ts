import { replayFrame } from "../watch/timeline";
export { replayFrame, snapshotLabel } from "../watch/timeline";
import type { Replay, Snapshot } from "../assets/load";

export interface GhostRun {
  snapshot: Snapshot;
  data: Replay;
  frames: number[][];
}
export interface GhostPose {
  run: GhostRun;
  frame: number[];
  status: "ready" | "running" | "finished" | "crashed" | "timeout";
}
/** Continuous cool-to-warm colors, also shared by the on-track labels and legend. */
export function snapshotColor(index: number, count: number): number {
  const stops = [0x70abff, 0x80e4d0, 0xff9366],
    t = count <= 1 ? 1 : index / (count - 1);
  const section = Math.min(1, Math.floor(t * 2)),
    fraction = t * 2 - section;
  const a = stops[section],
    b = stops[section + 1];
  return [16, 8, 0].reduce(
    (value, shift) =>
      value |
      (Math.round(
        ((a >> shift) & 255) +
          (((b >> shift) & 255) - ((a >> shift) & 255)) * fraction,
      ) <<
        shift),
    0,
  );
}
export class GhostPlayback {
  seconds = 0;
  playing = false;
  rate = 1;
  readonly duration: number;
  constructor(readonly runs: GhostRun[]) {
    if (!runs.length)
      throw new Error("Ghost playback requires at least one run");
    for (const run of runs) {
      if (
        !run.frames.length ||
        !Number.isFinite(run.data.dt) ||
        run.frames.length !== run.data.metadata.physics_ticks + 1 ||
        run.data.dt <= 0
      )
        throw new Error("Invalid ghost duration");
    }
    this.duration = Math.max(
      ...runs.map((run) => (run.frames.length - 1) * run.data.dt),
    );
  }
  setRate(rate: number): void {
    if (![0.5, 1, 2, 4].includes(rate))
      throw new Error("Unsupported playback speed");
    this.rate = rate;
  }
  play(): void {
    if (this.seconds >= this.duration) this.seconds = 0;
    this.playing = true;
  }
  pause(): void {
    this.playing = false;
  }
  restart(): void {
    this.seconds = 0;
    this.playing = true;
  }
  seek(seconds: number): void {
    if (!Number.isFinite(seconds)) throw new Error("Invalid playback position");
    this.seconds = Math.max(0, Math.min(this.duration, seconds));
    if (this.seconds === this.duration) this.pause();
  }
  advance(seconds: number): void {
    if (!Number.isFinite(seconds) || seconds < 0)
      throw new Error("Invalid playback elapsed time");
    if (this.playing) this.seek(this.seconds + seconds * this.rate);
  }
  poses(): GhostPose[] {
    return this.runs.map((run) => {
      const ended = this.seconds >= (run.frames.length - 1) * run.data.dt;
      const status =
        this.seconds === 0
          ? "ready"
          : !ended
            ? "running"
            : run.data.metadata.finished
              ? "finished"
              : run.data.metadata.crashed
                ? "crashed"
                : "timeout";
      return { run, frame: replayFrame(run, this.seconds), status };
    });
  }
}

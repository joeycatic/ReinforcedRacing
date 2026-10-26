import type { Attempt, Replay } from "../assets/load";

export function snapshotLabel(snapshot: { step: number; id?: string }): string {
  const step = snapshot.step;
  const label =
    step >= 1e6
      ? `${Number((step / 1e6).toFixed(2))}M`
      : step >= 1000
        ? `${Number((step / 1000).toFixed(1))}K`
        : String(step);
  return snapshot.id === "final" ? `${label} FINAL` : label;
}
export function replayFrame(
  run: { data: Replay; frames: number[][] },
  seconds: number,
): number[] {
  const tick = Math.max(
      0,
      Math.min(run.frames.length - 1, seconds / run.data.dt),
    ),
    index = Math.floor(tick),
    next = Math.min(index + 1, run.frames.length - 1),
    fraction = tick - index;
  return run.frames[index].map((value, column) =>
    column === 2
      ? value +
        Math.atan2(
          Math.sin(run.frames[next][column] - value),
          Math.cos(run.frames[next][column] - value),
        ) *
          fraction
      : value + (run.frames[next][column] - value) * fraction,
  );
}
export interface AttemptRun {
  attempt: Attempt;
  data: Replay;
  frames: number[][];
}
export interface Slot {
  index: number;
  rate: number;
  replaySeconds: number;
  displaySeconds: number;
  start: number;
}
const HOLD = 1.5;
/** Compress eligible attempts together, preserving stalls and the final lap. */
export function planTimeline(runs: AttemptRun[]): Slot[] {
  if (!runs.length) throw new Error("Watch timeline requires at least one run");
  const slots = runs.map((run, index) => {
    if (
      !run.frames.length ||
      !Number.isFinite(run.data.dt) ||
      run.data.dt <= 0 ||
      run.frames.length !== run.data.metadata.physics_ticks + 1
    )
      throw new Error("Invalid attempt duration");
    const seconds = (run.frames.length - 1) * run.data.dt;
    return {
      index,
      rate:
        run.attempt.source === "final"
          ? 2
          : run.attempt.outcome === "finished"
            ? 3
            : 1,
      replaySeconds:
        run.attempt.outcome === "stalled" ? Math.min(3, seconds) : seconds,
      displaySeconds: 0,
      start: 0,
    };
  });
  const eligible = (index: number) =>
    runs[index].attempt.source !== "final" &&
    runs[index].attempt.outcome !== "stalled";
  const total = (factor: number) =>
    slots.reduce(
      (sum, slot) =>
        sum +
        HOLD +
        slot.replaySeconds /
          (eligible(slot.index) ? Math.min(8, slot.rate * factor) : slot.rate),
      0,
    );
  let factor = 1;
  if (total(1) > 65) {
    let low = 1,
      high = 8;
    for (let i = 0; i < 60; i++) {
      const middle = (low + high) / 2;
      if (total(middle) > 60) low = middle;
      else high = middle;
    }
    factor = high;
  }
  let start = 0;
  for (const slot of slots) {
    if (eligible(slot.index)) slot.rate = Math.min(8, slot.rate * factor);
    slot.start = start;
    slot.displaySeconds = slot.replaySeconds / slot.rate + HOLD;
    start += slot.displaySeconds;
  }
  return slots;
}
export function caption(
  run: AttemptRun,
  index: number,
  count: number,
): { kicker: string; headline: string | null; outcome: string } {
  const metadata = run.data.metadata;
  return {
    kicker: `ATTEMPT ${index + 1} / ${count} · ${snapshotLabel({ step: Number(metadata.snapshot_step) })} STEPS`,
    headline: run.attempt.headline,
    outcome: metadata.finished
      ? `LAP ${Number(metadata.lap_time).toFixed(2)} s`
      : `${metadata.crashed ? "CRASHED" : "STALLED"} AT ${Math.round(Number(metadata.progress))} m`,
  };
}
export class WatchTimeline {
  seconds = 0;
  playing = false;
  readonly slots: Slot[];
  readonly duration: number;
  constructor(readonly runs: AttemptRun[]) {
    this.slots = planTimeline(runs);
    const last = this.slots.at(-1)!;
    this.duration = last.start + last.displaySeconds;
  }
  play(): void {
    if (this.seconds < this.duration) this.playing = true;
  }
  pause(): void {
    this.playing = false;
  }
  replay(): void {
    this.seconds = 0;
    this.playing = true;
  }
  skipToFinal(): void {
    this.jump(this.runs.length - 1);
  }
  jump(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.slots.length)
      throw new Error("Invalid attempt index");
    this.seconds = this.slots[index].start;
  }
  advance(realSeconds: number): void {
    if (!Number.isFinite(realSeconds) || realSeconds < 0)
      throw new Error("Invalid elapsed time");
    if (this.playing) {
      this.seconds = Math.min(this.duration, this.seconds + realSeconds);
      if (this.seconds >= this.duration) this.pause();
    }
  }
  current(): {
    index: number;
    replaySeconds: number;
    holding: boolean;
    frame: number[];
  } {
    const slot = this.slots
      .slice()
      .reverse()
      .find((slot) => slot.start <= this.seconds)!;
    const elapsed = (this.seconds - slot.start) * slot.rate;
    const replaySeconds = Math.min(slot.replaySeconds, elapsed);
    return {
      index: slot.index,
      replaySeconds,
      holding: elapsed >= slot.replaySeconds,
      frame: replayFrame(this.runs[slot.index], replaySeconds),
    };
  }
  trails(): number[] {
    return Array.from({ length: this.current().index }, (_, i) => i);
  }
}

import type { RaceEngine } from "../race/engine";

export type TrialResult =
  | { kind: "lap"; time: number }
  | { kind: "crashed"; at: number; tight: boolean }
  | { kind: "timeout" };

export class TrialLap {
  result: TrialResult | null = null;
  private ticking = false;
  constructor(
    private readonly engine: RaceEngine,
    private readonly tight: [number, number][],
    private readonly limitSeconds = 90,
  ) {
    engine.launch();
  }
  async tick(): Promise<void> {
    if (this.result || this.ticking) return;
    this.ticking = true;
    try {
      await this.engine.tick([0, 0]);
      if (this.engine.ai.crashes > 0) {
        const at = Math.round(this.engine.ai.progress.unwrapped);
        const length = this.engine.track.length;
        const arc = ((at % length) + length) % length;
        const tight = this.tight.some(([start, end]) =>
          [-length, 0, length].some(
            (offset) => arc + offset >= start - 15 && arc + offset <= end + 15,
          ),
        );
        this.result = { kind: "crashed", at, tight };
      } else if (this.engine.aiLapTime !== null) {
        this.result = { kind: "lap", time: this.engine.aiLapTime };
      } else if (
        this.engine.raceTicks * this.engine.config.physics.dt + 1e-12 >=
        this.limitSeconds
      ) {
        this.result = { kind: "timeout" };
      }
      if (this.result) this.engine.phase = "finished";
    } finally {
      this.ticking = false;
    }
  }
}

export function describe(result: TrialResult): string {
  if (result.kind === "timeout") return "DID NOT FINISH IN 90 S";
  if (result.kind === "crashed")
    return `CRASHED AT ${result.at} M${result.tight ? " — this corner is tighter than anything in training" : ""}`;
  const milliseconds = Math.round(result.time * 1000);
  const minutes = Math.floor(milliseconds / 60000);
  const seconds = ((milliseconds % 60000) / 1000).toFixed(3).padStart(6, "0");
  return `LAP ${minutes}:${seconds}`;
}

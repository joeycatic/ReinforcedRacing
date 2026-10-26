import { expect, it } from "vitest";
import loop from "../../tests/fixtures/editor/loop.json";
import parity from "../../tests/fixtures/parity/sweep.json";
import { TrialLap, describe } from "../src/editor/trial";
import { RaceEngine } from "../src/race/engine";
import { Track } from "../src/sim/track";
import type { SimulationConfig } from "../src/assets/load";
import type { TrackData, Action } from "../src/sim/types";
const create = (action: Action) =>
  new RaceEngine(
    new Track(loop.track as TrackData),
    parity.config as SimulationConfig,
    {
      predict: async () => ({ action, value: 0 }),
    },
  );
it("stops on the first straight-line crash and reports tight corners", async () => {
  const engine = create([0, 1]);
  const trial = new TrialLap(engine, [[0, engine.track.length]]);
  for (let i = 0; i < 5400 && !trial.result; i++) await trial.tick();
  expect(trial.result).toMatchObject({ kind: "crashed", tight: true });
  if (trial.result?.kind !== "crashed") throw Error("Expected crash");
  expect(trial.result.at).toBeGreaterThan(0);
  expect(describe(trial.result)).toContain("tighter than anything in training");
  const ticks = engine.raceTicks;
  await trial.tick();
  expect(engine.raceTicks).toBe(ticks);
  expect(engine.ai.crashes).toBe(1);
  expect(engine.player.state.vx).toBe(0);
});
it("times out after exactly 90 simulated seconds with no countdown", async () => {
  const engine = create([0, 0]),
    trial = new TrialLap(engine, []);
  expect(engine.phase).toBe("racing");
  for (let i = 0; i < 5400; i++) await trial.tick();
  expect(trial.result).toEqual({ kind: "timeout" });
  expect(engine.raceTicks).toBe(5400);
  expect(engine.phase).toBe("finished");
});
it("stops at the first recorded AI lap", async () => {
  const engine = create([0, 0]),
    trial = new TrialLap(engine, []);
  engine.aiLapTime = 31.482;
  await trial.tick();
  expect(trial.result).toEqual({ kind: "lap", time: 31.482 });
  expect(engine.phase).toBe("finished");
});
it("formats trial results", () => {
  expect(describe({ kind: "lap", time: 31.482 })).toBe("LAP 0:31.482");
  expect(describe({ kind: "lap", time: 59.9996 })).toBe("LAP 1:00.000");
  expect(describe({ kind: "crashed", at: 412, tight: false })).toBe(
    "CRASHED AT 412 M",
  );
  expect(describe({ kind: "timeout" })).toBe("DID NOT FINISH IN 90 S");
});
it("includes the 15 metre margin, excludes farther sections, and wraps at the start", async () => {
  const baseline = create([0, 1]),
    first = new TrialLap(baseline, []);
  while (!first.result) await first.tick();
  if (first.result.kind !== "crashed") throw Error("Expected crash");
  const at = first.result.at;
  for (const [section, expected] of [
    [[at + 15, at + 16], true],
    [[at + 16, at + 17], false],
    [[at - 16, at - 15], true],
  ] as const) {
    const engine = create([0, 1]),
      trial = new TrialLap(engine, [[...section]]);
    while (!trial.result) await trial.tick();
    expect(trial.result).toMatchObject({ kind: "crashed", tight: expected });
  }
  const engine = create([0, 0]),
    trial = new TrialLap(engine, [
      [engine.track.length - 5, engine.track.length],
    ]);
  engine.ai.crashes = 1;
  await trial.tick();
  expect(trial.result).toMatchObject({ kind: "crashed", at: 0, tight: true });
});

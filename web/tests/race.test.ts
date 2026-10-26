import { readFileSync } from "node:fs";
import { describe, it, expect } from "vitest";
import { RaceEngine, FixedClock, interpolate } from "../src/race/engine";
import type { Policy, Prediction } from "../src/race/policy";
import type { SimulationConfig } from "../src/assets/load";
import { Track } from "../src/sim/track";
import type { Action } from "../src/sim/types";
import { clamp, speed } from "../src/sim/physics";
import { DriveSession } from "../src/sim/session";
const f = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/parity/sweep.json", import.meta.url),
    "utf8",
  ),
);
const config = f.config as SimulationConfig;
const stationary: Policy = {
  predict: async () => ({ action: [0, 0], value: 0 }),
};
const create = (policy = stationary): RaceEngine =>
  new RaceEngine(new Track(f.track), config, policy);
describe("live race engine", () => {
  it("counts down without driving and requests a mean action every two physics ticks", async () => {
    let calls = 0;
    const engine = create({
      predict: async () => {
        calls++;
        return { action: [0.1, 1], value: 3 };
      },
    });
    engine.start();
    for (let i = 0; i < 179; i++) await engine.tick([0, 1]);
    expect(engine.phase).toBe("countdown");
    expect(engine.countdown).toBe(1);
    expect(engine.ai.ticks).toBe(0);
    await engine.tick([0, 1]);
    expect(engine.phase).toBe("racing");
    expect(calls).toBe(0);
    for (let i = 0; i < 5; i++) await engine.tick([0, 1]);
    expect(calls).toBe(3);
    expect(engine.inferences).toBe(3);
    expect(engine.ai.ticks).toBe(5);
    expect(engine.value).toBe(3);
    engine.pause();
    await engine.tick([1, 1]);
    expect(engine.ai.ticks).toBe(5);
    engine.resume();
    await engine.tick([0, 1]);
    expect(engine.ai.ticks).toBe(6);
    expect(calls).toBe(3);
  });
  it("discards a pending prediction after a restart", async () => {
    let finish!: (value: Prediction) => void;
    const engine = create({
      predict: () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    });
    engine.phase = "racing";
    const pending = engine.tick([1, 1]);
    engine.start();
    finish({ action: [1, 1], value: 9 });
    await pending;
    expect(engine.ai.ticks).toBe(0);
    expect(engine.player.ticks).toBe(0);
    expect(engine.phase).toBe("countdown");
    expect(engine.countdownTicks).toBe(180);
  });
  it("keeps physics independent of render frequency and bounds catch-up", async () => {
    const slow = create(),
      fast = create();
    slow.phase = fast.phase = "racing";
    const a = new FixedClock(slow),
      b = new FixedClock(fast);
    for (let i = 0; i < 30; i++) await a.advance(1 / 30, [0, 1]);
    for (let i = 0; i < 144; i++) await b.advance(1 / 144, [0, 1]);
    expect(slow.player.ticks).toBe(60);
    expect(fast.player.ticks).toBe(60);
    expect(slow.player.state).toEqual(fast.player.state);
    expect(slow.inferences).toBe(30);
    expect(fast.inferences).toBe(30);
    slow.pause();
    await a.advance(1, [1, 1]);
    expect(slow.player.ticks).toBe(60);
    slow.resume();
    await a.advance(10, [0, 1]);
    expect(slow.player.ticks).toBe(68);
    expect(a.droppedSeconds).toBeGreaterThan(9);
  });
  it("finishes a full driven lap, records interpolated splits and resets a rematch", async () => {
    function follow(session: DriveSession): Action {
      const s = session.state,
        c = session.physics,
        [target] = session.track.sample(session.progress.projection.arc + 10);
      const error = Math.atan2(target[1] - s.y, target[0] - s.x) - s.psi;
      const angle = Math.atan2(
        2 * (c.front_axle + c.rear_axle) * Math.sin(error),
        10,
      );
      return [
        clamp(
          (angle * (1 + speed(s) / c.steer_speed_falloff)) / c.max_steer,
          -1,
          1,
        ),
        clamp((12 - s.vx) * 0.5, -1, 1),
      ];
    }
    let engine: RaceEngine;
    const policy: Policy = {
      predict: async () => ({ action: follow(engine.ai), value: 0 }),
    };
    engine = create(policy);
    engine.phase = "racing";
    for (let i = 0; i < 6000 && engine.phase === "racing"; i++)
      await engine.tick(follow(engine.player));
    expect(engine.phase).toBe("finished");
    expect(engine.player.crashes).toBe(0);
    expect(engine.ai.crashes).toBe(0);
    expect(engine.playerLapTime).toBeCloseTo(engine.aiLapTime!, 10);
    expect(engine.playerSplits.size).toBeGreaterThan(30);
    expect(engine.split!.delta).toBe(0);
    const final = engine.player.state;
    await engine.tick([1, 1]);
    expect(engine.player.state).toEqual(final);
    engine.start();
    expect(engine.phase).toBe("countdown");
    expect(engine.playerLapTime).toBeNull();
    expect(engine.playerSplits.size).toBe(0);
    expect(engine.aiLapTime).toBeNull();
    expect(engine.ai.ticks).toBe(0);
  });
  it("interpolates across heading wrap along the shortest arc", () => {
    const a = {
      x: 0,
      y: 0,
      psi: Math.PI - 0.1,
      vx: 0,
      vy: 0,
      r: 0,
      steering: 0,
    };
    const b = { ...a, x: 2, y: 4, psi: -Math.PI + 0.1 };
    const midpoint = interpolate(a, b, 0.5);
    expect(midpoint.x).toBe(1);
    expect(midpoint.y).toBe(2);
    expect(midpoint.psi).toBeCloseTo(Math.PI, 12);
  });
});

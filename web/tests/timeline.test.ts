import { expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { decodeReplay, type Manifest, type Replay } from "../src/assets/load";
import {
  caption,
  planTimeline,
  snapshotLabel,
  WatchTimeline,
  type AttemptRun,
} from "../src/watch/timeline";
const root = new URL("../public/assets/", import.meta.url);
const manifest: Manifest = JSON.parse(
  readFileSync(new URL("manifest.json", root), "utf8"),
);
const tracks = Object.entries(manifest.attempts).map(
  ([id, attempts]) =>
    [
      id,
      attempts.map((attempt) => {
        const data: Replay = JSON.parse(
          readFileSync(new URL(attempt.replay.path, root), "utf8"),
        );
        return { attempt, data, frames: decodeReplay(data) };
      }),
    ] as const,
);
const sample = (
  outcome: "stalled" | "crashed" | "finished",
  seconds = 10,
  source = "checkpoint",
): AttemptRun => ({
  attempt: { ...tracks[0][1][0].attempt, outcome, source },
  data: {
    ...tracks[0][1][0].data,
    dt: 1,
    metadata: { ...tracks[0][1][0].data.metadata, physics_ticks: seconds },
  },
  frames: Array.from({ length: seconds + 1 }, (_, i) => [i, 0, 0, 0, 0, 0]),
});
it("planTimeline cuts stalls to 3 s", () =>
  expect(planTimeline([sample("stalled", 200)])[0].replaySeconds).toBe(3));
it("final plays at 2×", () =>
  expect(planTimeline([sample("finished", 20, "final")])[0].rate).toBe(2));
it("crashes play at 1× when the total fits", () =>
  expect(planTimeline([sample("crashed")])[0].rate).toBe(1));
for (const [id, runs] of tracks) {
  it(`${id}: real attempts fit 50–70 s`, () => {
    const last = planTimeline(runs).at(-1)!;
    expect(last.start + last.displaySeconds).toBeGreaterThanOrEqual(50);
    expect(last.start + last.displaySeconds).toBeLessThanOrEqual(70);
  });
  it(`${id}: captions match replay outcomes`, () =>
    runs.forEach((run, i) => {
      const m = run.data.metadata;
      expect(caption(run, i, runs.length).outcome).toBe(
        m.finished
          ? `LAP ${Number(m.lap_time).toFixed(2)} s`
          : `${m.crashed ? "CRASHED" : "STALLED"} AT ${Math.round(Number(m.progress))} m`,
      );
      expect(caption(run, i, runs.length).kicker).toBe(
        `ATTEMPT ${i + 1} / ${runs.length} · ${snapshotLabel({ step: Number(m.snapshot_step) })} STEPS`,
      );
      expect(caption(run, i, runs.length).headline).toBe(run.attempt.headline);
    }));
}
it("trails are the attempts before the current one after any jump", () => {
  const t = new WatchTimeline(tracks[0][1]);
  t.jump(7);
  expect(t.trails()).toEqual([0, 1, 2, 3, 4, 5, 6]);
  t.jump(2);
  expect(t.trails()).toEqual([0, 1]);
  t.skipToFinal();
  expect(t.trails()).toEqual(tracks[0][1].slice(1).map((_, i) => i));
});
it("only one attempt is current at any time and playback stops", () => {
  const t = new WatchTimeline(tracks[0][1]);
  t.play();
  let previous = 0;
  for (let s = 0; s <= t.duration + 1; s += 0.25) {
    const current = t.current();
    expect(current.index).toBeGreaterThanOrEqual(previous);
    expect(current.frame).toHaveLength(6);
    previous = current.index;
    t.advance(0.25);
  }
  expect(t.seconds).toBe(t.duration);
  expect(t.playing).toBe(false);
});
it("holds the endpoint, pauses, replays and validates navigation", () => {
  const t = new WatchTimeline([
    sample("crashed", 2),
    sample("finished", 4, "final"),
  ]);
  t.play();
  t.advance(2);
  expect(t.current()).toMatchObject({
    index: 0,
    holding: true,
    replaySeconds: 2,
  });
  t.pause();
  t.advance(1);
  expect(t.seconds).toBe(2);
  t.replay();
  expect(t.seconds).toBe(0);
  expect(t.playing).toBe(true);
  expect(() => t.jump(-1)).toThrow();
  expect(() => t.advance(NaN)).toThrow();
  expect(() => new WatchTimeline([])).toThrow();
});

it("compression shares one factor, caps eligible rates, and preserves fixed rates", () => {
  const slots = planTimeline([
    sample("crashed", 2000),
    sample("finished", 2000),
    sample("stalled", 200),
    sample("finished", 20, "final"),
  ]);
  expect(slots.map((s) => s.rate)).toEqual([8, 8, 1, 2]);
  const paced = planTimeline([
    sample("crashed", 60),
    sample("finished", 60),
    sample("finished", 20, "final"),
  ]);
  expect(paced[1].rate / paced[0].rate).toBeCloseTo(3);
  expect(paced.at(-1)!.start + paced.at(-1)!.displaySeconds).toBeCloseTo(60);
});

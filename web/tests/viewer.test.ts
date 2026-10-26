import { describe, it, expect } from "vitest";
import type { Heatmap, Replay, Snapshot } from "../src/assets/load";
import {
  GhostPlayback,
  replayFrame,
  snapshotColor,
  snapshotLabel,
  type GhostRun,
} from "../src/viewer/playback";
import { heatmapPixels, heatmapScale } from "../src/viewer/heatmap";
function run(
  id: string,
  frames: number[][],
  finished = false,
  crashed = false,
): GhostRun {
  const snapshot = {
    id,
    step: id === "final" ? 6000000 : Number(id),
    checkpoint_sha256: "a".repeat(64),
    model: { path: "m.onnx", bytes: 1, sha256: "a".repeat(64) },
    replays: {},
    heatmaps: {},
  } satisfies Snapshot;
  const data = {
    version: "replay.v1",
    dt: 0.1,
    columns: [],
    scales: [],
    deltas: [],
    metadata: {
      track_id: "test",
      snapshot_id: id,
      physics_ticks: frames.length - 1,
      finished,
      crashed,
    },
  } satisfies Replay;
  return { snapshot, data, frames };
}
const frame = (x: number, heading = 0): number[] => [x, 0, heading, 10, 0, 1];
describe("synchronized recorded ghosts", () => {
  it("uses a common start clock and holds ended ghosts at their recorded last pose", () => {
    const a = run("0", [frame(0), frame(1)], false, true),
      b = run("final", [frame(0), frame(10), frame(20)], true);
    const p = new GhostPlayback([a, b]);
    expect(p.poses().map((g) => g.status)).toEqual(["ready", "ready"]);
    p.play();
    p.advance(0.15);
    const poses = p.poses();
    expect(poses[0].frame).toEqual(frame(1));
    expect(poses[0].status).toBe("crashed");
    expect(poses[1].frame[0]).toBeCloseTo(15, 12);
    expect(poses[1].status).toBe("running");
    p.advance(1);
    expect(p.seconds).toBe(0.2);
    expect(p.playing).toBe(false);
    expect(p.poses()[1].status).toBe("finished");
    expect(p.poses()[1].frame).toEqual(frame(20));
    expect(a.frames[0]).toEqual(frame(0));
  });
  it("pauses, resumes, seeks and restarts at every supported rate", () => {
    for (const rate of [0.5, 1, 2, 4]) {
      const p = new GhostPlayback([
        run(
          "0",
          Array.from({ length: 101 }, (_, i) => frame(i)),
        ),
      ]);
      p.setRate(rate);
      p.play();
      p.advance(1);
      expect(p.seconds).toBe(rate);
      p.pause();
      p.advance(2);
      expect(p.seconds).toBe(rate);
      p.seek(4.5);
      expect(p.seconds).toBe(4.5);
      expect(p.playing).toBe(false);
      p.restart();
      expect(p.seconds).toBe(0);
      expect(p.playing).toBe(true);
      p.seek(99);
      expect(p.seconds).toBe(10);
      expect(p.poses()[0].status).toBe("timeout");
      p.play();
      expect(p.seconds).toBe(0);
      expect(() => p.setRate(0)).toThrow();
      expect(() => p.advance(NaN)).toThrow();
      expect(() => p.seek(Infinity)).toThrow();
    }
  });
  it("interpolates heading through the shortest angle and clamps negative/late times", () => {
    const r = run("0", [frame(0, Math.PI - 0.1), frame(2, -Math.PI + 0.1)]);
    expect(replayFrame(r, 0.05)[0]).toBe(1);
    expect(replayFrame(r, 0.05)[2]).toBeCloseTo(Math.PI, 12);
    expect(replayFrame(r, -1)).toEqual(r.frames[0]);
    expect(replayFrame(r, 10)).toEqual(r.frames[1]);
  });
  it("labels actual steps and uses distinct cool-to-warm stage colors", () => {
    expect(
      snapshotLabel({ ...run("0", [frame(0)]).snapshot, step: 50048 }),
    ).toBe("50K");
    expect(snapshotLabel(run("final", [frame(0)]).snapshot)).toBe("6M FINAL");
    const colors = Array.from({ length: 6 }, (_, i) => snapshotColor(i, 6));
    expect(colors[0]).toBe(0x70abff);
    expect(colors[5]).toBe(0xff9366);
    expect(new Set(colors).size).toBe(6);
    expect(snapshotColor(0, 1)).toBe(0xff9366);
  });
});
function map(): Heatmap {
  return {
    version: "heatmap.v1",
    metadata: { track_id: "test", snapshot_id: "0" },
    grid: {
      encoding: "sparse.flat.v1",
      origin: [-1, -2],
      width: 2,
      height: 2,
      cell_size: 0.5,
    },
    visits: [
      [0, 1],
      [3, 9],
    ],
    crash_bins: [[2, 3]],
    crash_positions: [[0, 0]],
    visit_count: 10,
    crash_count: 3,
  };
}
describe("heatmap rasterization", () => {
  it("flips world y into texture y and leaves absent cells transparent", () => {
    const pixels = heatmapPixels(map(), "visits", 9);
    expect(pixels[3]).toBe(0);
    expect(pixels[7]).toBe(165);
    expect(pixels[11]).toBeGreaterThan(0);
    expect(pixels[15]).toBe(0);
    expect([...pixels.slice(4, 7)]).toEqual([134, 255, 217]);
    const crash = heatmapPixels(map(), "crash_bins", 3);
    expect(crash[3]).toBe(220);
    expect(crash[7]).toBe(0);
  });
  it("uses one per-track scale over every training stage and handles empty crash layers", () => {
    const early = map(),
      late = {
        ...map(),
        visits: [[0, 90]] as [number, number][],
        crash_bins: [],
      };
    const scale = heatmapScale([early, late]);
    expect(scale).toEqual({ visits: 90, crashes: 3 });
    const relative = heatmapPixels(early, "visits", scale.visits),
      normalized = heatmapPixels(early, "visits", 9);
    expect(relative[7]).toBeLessThan(normalized[7]);
    expect(heatmapPixels(late, "crash_bins", 3).every((x) => x === 0)).toBe(
      true,
    );
    expect(heatmapPixels(early, "visits", 0).every((x) => x === 0)).toBe(true);
  });
});

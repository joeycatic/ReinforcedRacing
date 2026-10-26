import { describe, it, expect } from "vitest";
import type { Heatmap } from "../src/assets/load";
import { snapshotColor, snapshotLabel } from "../src/watch/timeline";
import { heatmapPixels, heatmapScale } from "../src/viewer/heatmap";
it("labels actual steps and cool-to-warm colours", () => {
  expect(snapshotLabel({ step: 50048 })).toBe("50K");
  expect(snapshotLabel({ step: 6000000, id: "final" })).toBe("6M FINAL");
  const colors = Array.from({ length: 6 }, (_, i) => snapshotColor(i, 6));
  expect(colors[0]).toBe(0x70abff);
  expect(colors[5]).toBe(0xff9366);
  expect(new Set(colors).size).toBe(6);
  expect(snapshotColor(0, 1)).toBe(0xff9366);
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

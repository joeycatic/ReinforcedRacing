import { Track } from "../src/sim/track";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { buildEditorTrack, type EditorConfig } from "../src/editor/build";
import type { TrackData, Vec2 } from "../src/sim/types";
const directory = fileURLToPath(
  new URL("../../tests/fixtures/editor/", import.meta.url),
);
interface Fixture {
  control: Vec2[];
  config: EditorConfig;
  over: number[] | null;
  track: TrackData;
  issues: string[];
  tight: [number, number][];
}
let maxError = 0;
function compare(actual: unknown, expected: unknown): void {
  if (typeof expected === "number") {
    const error = Math.abs((actual as number) - expected);
    maxError = Math.max(maxError, error);
    expect(error).toBeLessThanOrEqual(Number.isInteger(expected) ? 0 : 1e-9);
  } else if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expect((actual as unknown[]).length).toBe(expected.length);
    expected.forEach((value, i) => compare((actual as unknown[])[i], value));
  } else if (expected && typeof expected === "object") {
    expect(Object.keys(actual as object).sort()).toEqual(
      Object.keys(expected).sort(),
    );
    for (const key of Object.keys(expected))
      compare(
        (actual as Record<string, unknown>)[key],
        (expected as Record<string, unknown>)[key],
      );
  } else expect(actual).toEqual(expected);
}
for (const name of readdirSync(directory))
  describe(name, () => {
    const f = JSON.parse(
      readFileSync(directory + "/" + name, "utf8"),
    ) as Fixture;
    it("matches Python geometry, issues and tight sections", () => {
      const result = buildEditorTrack(f.control, f.config, f.over ?? undefined);
      compare(result.track.data, f.track);
      expect(result.issues).toEqual(f.issues);
      compare(result.tight, f.tight);
      console.log(`${name}: geometry max error ${maxError}`);
    });
  });
it("rebuild is fast", () => {
  const f = JSON.parse(
    readFileSync(directory + "/figure8.json", "utf8"),
  ) as Fixture;
  buildEditorTrack(f.control, f.config);
  const start = performance.now();
  for (let i = 0; i < 20; i++) buildEditorTrack(f.control, f.config);
  const average = (performance.now() - start) / 20;
  console.log(`figure8 rebuild average ${average} ms`);
  expect(average).toBeLessThan(50);
});

it("rejects tampered crossing metadata and keeps over rendering-only", () => {
  const f = JSON.parse(
    readFileSync(directory + "/figure8.json", "utf8"),
  ) as Fixture;
  const a = buildEditorTrack(f.control, f.config, [0]),
    b = buildEditorTrack(f.control, f.config, [1]);
  expect(a.track.data.crossings![0].over).toBe(0);
  expect({ ...a.track.data, crossings: undefined }).toEqual({
    ...b.track.data,
    crossings: undefined,
  });
  const tampered = structuredClone(a.track.data);
  tampered.crossings![0].hidden[0].pop();
  expect(() => new Track(tampered)).toThrow("Inconsistent crossing");
});

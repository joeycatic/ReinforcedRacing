import { expect, it } from "vitest";
import figure8 from "../../tests/fixtures/editor/figure8.json";
import pretzel from "../../tests/fixtures/editor/pretzel.json";
import config from "../src/editor/editor-config.json";
import { buildEditorTrack, roundEven } from "../src/editor/build";
import type { Vec2 } from "../src/sim/types";
import { encodeTrack, decodeTrack } from "../src/editor/share";
import { strokeToControls } from "../src/editor/stroke";
import { canSubmit, MESSAGES } from "../src/editor/issues";
const circle = Array.from({ length: 101 }, (_, i): Vec2 => [
  100 + 100 * Math.cos((i * Math.PI) / 50),
  100 * Math.sin((i * Math.PI) / 50),
]);
it("open stroke is not-closed", () =>
  expect(
    strokeToControls(
      [
        [0, 0],
        [400, 0],
      ],
      config,
    ).problem,
  ).toBe("not-closed"));
it("tiny stroke is too-short", () =>
  expect(
    strokeToControls(
      [
        [0, 0],
        [1, 0],
        [0, 0],
      ],
      config,
    ).problem,
  ).toBe("too-short"));
it("closed circle gives 24 controls starting at first point", () => {
  const r = strokeToControls(circle, config);
  expect(r.problem).toBeNull();
  expect(r.control).toHaveLength(24);
  expect(r.control![0]).toEqual([200, -0]);
});
it("scribble is blocked", () => {
  const p: Vec2[] = Array.from({ length: 40 }, (_, i) => [
    i * 10,
    (i % 2) * 200,
  ]);
  p.push(p[0]);
  const r = strokeToControls(p, config);
  const b = buildEditorTrack(r.control!, config);
  expect(b.issues.length).toBeGreaterThan(0);
  expect(canSubmit(r.problem, b.issues)).toBe(false);
});
for (const [name, f] of [
  ["figure8", figure8],
  ["pretzel", pretzel],
] as const)
  it(`${name} share round trip reproduces the track`, () => {
    const c = f.control as Vec2[];
    const o = f.track.crossings.map(() => 0);
    const decoded = decodeTrack(encodeTrack(c, o))!;
    expect(decoded.over).toEqual(o);
    expect(buildEditorTrack(decoded.control, config, decoded.over)).toEqual(
      buildEditorTrack(c, config, o),
    );
  });
it("decode rejects corrupt codes", () => {
  for (const c of ["2.AA", "1.!", "1.gA", "1._wE", "1.BAE", "1.", "1.AA=="])
    expect(decodeTrack(c)).toBeNull();
});
it("tight warns but allows submit", () => {
  expect(MESSAGES.tight).toBe(
    "Too tight — the AI never trained on corners this sharp",
  );
  expect(canSubmit(null, ["tight"])).toBe(true);
  expect(canSubmit("too-short", [])).toBe(false);
  expect(canSubmit(null, ["fold", "tight"])).toBe(false);
});

function stream(...values: number[]): string {
  const bytes: number[] = [];
  for (let n of values) {
    do {
      const b = n % 128;
      n = Math.floor(n / 128);
      bytes.push(b + (n ? 128 : 0));
    } while (n);
  }
  return (
    "1." +
    btoa(String.fromCharCode(...bytes))
      .replaceAll("+", "-")
      .replaceAll("/", "_")
      .replace(/=+$/, "")
  );
}
it("rejects bounded stream violations and over count mismatch", () => {
  const bad = [
    stream(257, 0),
    stream(4, 129),
    stream(4, 1, 2),
    stream(4, 0, 40000002, 0, 0, 0, 0, 0, 0, 0),
    stream(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    "1.gAAA",
  ];
  for (const code of bad) expect(decodeTrack(code)).toBeNull();
  // An otherwise valid figure eight cannot claim no bridge selection.
  expect(decodeTrack(encodeTrack(figure8.control as Vec2[], []))).toBeNull();
  const code = encodeTrack(figure8.control as Vec2[], [0]);
  expect(decodeTrack(code.slice(0, -1))).toBeNull();
});
it("uses ties-even quantum rounding including negative halfways", () => {
  expect([2.5, -2.5, 3.5, -3.5].map(roundEven)).toEqual([2, -2, 4, -4]);
  const controls: Vec2[] = [
    [0.25, -0.25],
    [100.25, -0.25],
    [100.25, 100.25],
    [0.25, 100.25],
  ];
  expect(decodeTrack(encodeTrack(controls, []))!.control[0]).toEqual([
    0.2, -0.2,
  ]);
});
it("has exact issue copy", () =>
  expect(MESSAGES).toEqual({
    "not-closed": "Finish near the start to close the loop",
    "too-short": "Draw a bigger loop",
    fold: "The road folds over itself in a corner",
    overlap: "Roads overlap — cross them clearly or keep them apart",
    "shallow-crossing": "Roads cross at too shallow an angle",
    "crossing-spacing": "Crossings are too close together",
    "crossing-start": "A crossing is too close to the start line",
    tight: "Too tight — the AI never trained on corners this sharp",
  }));
it("handles duplicate stroke samples", () => {
  expect(strokeToControls([circle[0], ...circle], config)).toEqual(
    strokeToControls(circle, config),
  );
});

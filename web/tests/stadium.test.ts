import { it, expect } from "vitest";
import { kerbStripes, gateOrdinals, cameraZoom } from "../src/race/stadium";
it("stripes a straight every three meters", () => {
  const stripes = kerbStripes(
    [
      [0, 0],
      [30, 0],
    ],
    3,
  );
  expect(stripes).toHaveLength(10);
  expect(stripes.map((s) => s.red)).toEqual(
    Array.from({ length: 10 }, (_, i) => i % 2 === 0),
  );
  expect(stripes[9].points.at(-1)).toEqual([30, 0]);
});
it("wraps a closed loop back to the first stripe", () => {
  const stripes = kerbStripes(
    [
      [0, 0],
      [9, 0],
      [9, 9],
      [0, 9],
    ],
    3,
  );
  expect(stripes.at(-1)!.points.at(-1)).toEqual(stripes[0].points[0]);
});
it("places gates at every fourth checkpoint", () => {
  expect(gateOrdinals(33, 4)).toEqual([0, 4, 8, 12, 16, 20, 24, 28, 32]);
});
it("eases camera zoom down to its minimum", () => {
  expect(cameraZoom(0)).toBe(12);
  expect(cameraZoom(1000)).toBe(6);
  for (let v = 0; v < 100; v++)
    expect(cameraZoom(v + 1)).toBeLessThanOrEqual(cameraZoom(v));
});

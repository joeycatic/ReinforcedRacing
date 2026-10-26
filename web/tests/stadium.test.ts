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

import figure8 from "../../tests/fixtures/editor/figure8.json";
import { Track } from "../src/sim/track";
import type { TrackData } from "../src/sim/types";
import { carAboveBridges } from "../src/race/stadium";
it("places only cars on the selected overpass above bridges", () => {
  const track = new Track(structuredClone(figure8.track) as TrackData);
  const crossing = track.data.crossings![0];
  expect(crossing.over).toBe(1);
  const half =
    track.data.width / Math.sin((crossing.angle * Math.PI) / 180) + 2;
  expect(carAboveBridges(track, crossing.arcs[1] + half - 0.01)).toBe(true);
  expect(carAboveBridges(track, crossing.arcs[1] + half + 0.01)).toBe(false);
  expect(carAboveBridges(track, crossing.arcs[1])).toBe(true);
  expect(carAboveBridges(track, crossing.arcs[0])).toBe(false);
  expect(carAboveBridges(track, crossing.arcs[1] + 100)).toBe(false);
  expect(carAboveBridges(track, crossing.arcs[1] + track.length)).toBe(true);
  crossing.over = 0;
  expect(carAboveBridges(track, crossing.arcs[0])).toBe(true);
  expect(carAboveBridges(track, crossing.arcs[1])).toBe(false);
});
it("keeps open bridge kerbs open and ordinary tracks below bridges", () => {
  const edge: [number, number][] = [
    [0, 0],
    [3, 0],
    [6, 0],
  ];
  expect(kerbStripes(edge, 3, false)).toHaveLength(2);
  const data = structuredClone(figure8.track) as TrackData;
  data.version = "track.v1";
  delete data.crossings;
  expect(carAboveBridges(new Track(data), 0)).toBe(false);
});

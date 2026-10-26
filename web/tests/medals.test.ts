import { it, expect } from "vitest";
import type { Medal, MedalName } from "../src/assets/load";
import { medalsForTrack, earnedMedal } from "../src/race/medals";
const medals = (["bronze", "silver", "gold", "author"] as MedalName[]).map(
  (medal, i) =>
    ({
      medal,
      lap_times: {
        sweep: [28, 26, 24.25, 24][i],
        technical: i === 1 ? null : 30 - i,
      },
    }) as unknown as Medal,
);
it("medalsForTrack drops medals without a lap time", () =>
  expect(medalsForTrack(medals, "technical").map((m) => m.medal)).toEqual([
    "bronze",
    "gold",
    "author",
  ]));
it("medalsForTrack keeps bronze→author order", () =>
  expect(medalsForTrack(medals, "sweep")).toEqual(medals));
it("awards the best medal including an equal lap", () =>
  expect(earnedMedal(24, medals, "sweep")).toBe("author"));
it("awards no medal for a slower lap than bronze", () =>
  expect(earnedMedal(29, medals, "sweep")).toBeNull());
it("earnedMedal ignores medals without a lap on the track", () =>
  expect(earnedMedal(29, medals, "technical")).toBe("bronze"));

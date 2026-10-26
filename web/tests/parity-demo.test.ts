import { it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { Track } from "../src/sim/track";
import { replayParity } from "../src/intro/parity";
it("replays 600 exported Python poses with sub-micro precision", () => {
  const read = (path: string) =>
    JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
  const demo = read("../public/assets/parity-demo.json");
  const result = replayParity(
    demo,
    new Track(read("../public/assets/tracks/sweep.track.v1.json")),
    read("../public/assets/simulation.json"),
  );
  expect(result.typescript).toHaveLength(600);
  expect(result.maxError).toBeLessThanOrEqual(1e-6);
});

import { it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { formatStats } from "../src/intro/format";
it("formats all four source-backed stats exactly", () => {
  const site = JSON.parse(
    readFileSync(
      new URL("../public/assets/site.json", import.meta.url),
      "utf8",
    ),
  );
  expect(formatStats(site.stats).map((s) => s.value)).toEqual([
    "98.7%",
    "85.2%",
    "0 RL libs",
    "5e-13",
  ]);
});

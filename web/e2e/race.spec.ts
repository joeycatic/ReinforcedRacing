import { test, expect } from "@playwright/test";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import type {} from "./runner";
const measurements: unknown[] = [];
test.afterAll(() => {
  mkdirSync("test-results", { recursive: true });
  writeFileSync(
    "test-results/m5-browser-parity.json",
    JSON.stringify(measurements, null, 2) + "\n",
  );
});
for (const track of ["sweep", "ribbon", "technical"])
  test(`live WASM policy matches Python lap time on ${track}`, async ({
    page,
  }) => {
    await page.goto("/e2e/runner.html");
    await page.waitForFunction(
      () => typeof window.validatePolicy === "function",
    );
    const result = await page.evaluate(
      (track) => window.validatePolicy(track),
      track,
    );
    console.log(JSON.stringify(result));
    measurements.push(result);
    expect(result.crashes).toBe(0);
    expect(result.relativeError).toBeLessThanOrEqual(0.02);
    expect(result.inferences).toBe(Math.ceil(result.ticks / 2));
  });
test("race controls, countdown, pause, restart and responsive layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/race.html");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator(".track-card")).toHaveCount(3);
  await expect(page.locator("#track-select,#snapshot-select")).toHaveCount(0);
  await expect(page.locator('[data-medal="gold"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator('[data-medal="author"]')).toContainText("AUTHOR");
  await page.locator('[data-track="technical"]').click();
  await page.locator('[data-medal="author"]').click();
  await page.locator('[data-medal="gold"]').click();
  await page.locator('[data-medal="author"]').click();
  await expect(page.locator("#ai-label")).toHaveText("AUTHOR MEDAL");
  await expect(page.locator("#start-button")).toBeEnabled();
  await page.locator('[data-track="sweep"]').click();
  await expect(page.locator("#start-button")).toBeEnabled();
  await page.screenshot({
    path: "test-results/m7-prerace.png",
    fullPage: true,
  });
  await page.screenshot({ path: "test-results/m5-ready.png", fullPage: true });
  await page.locator("#start-button").click();
  await expect(page.locator("#center-message")).toContainText("3");
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.keyboard.down("ArrowUp");
  await page.waitForTimeout(1400);
  await page.keyboard.up("ArrowUp");
  await page.screenshot({ path: "test-results/m7-race.png", fullPage: true });
  expect(Number(await page.locator("#speed").innerText())).toBeGreaterThan(0);
  await page.keyboard.press("Escape");
  await expect(page.locator("#race-state")).toHaveText("PAUSED");
  const frozen = await page.locator("#lap-timer").innerText();
  await page.waitForTimeout(250);
  await expect(page.locator("#lap-timer")).toHaveText(frozen);
  await page.locator("#pause-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS");
  await page.keyboard.press("KeyR");
  await expect(page.locator("#speed")).toHaveText("000");
  await page.keyboard.press("Enter");
  await expect(page.locator("#race-state")).toHaveText("START SEQUENCE");
  await page.locator(".race-actions .change-race").click();
  await page.locator('[data-track="technical"]').click();
  await expect(page.locator("#circuit-name")).toHaveText("TECHNICAL LOOP");
  await expect(page.locator("#race-state")).toHaveText("GRID READY");
  await page.locator('[data-medal="gold"]').click();
  await expect(page.locator("#start-button")).toBeEnabled();
  await expect(page.locator("#ai-label")).toHaveText("GOLD MEDAL");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "test-results/m7-prerace-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.screenshot({
    path: "test-results/m7-race-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
test("render frame timing during live inference", async ({ page }) => {
  await page.goto("/race.html");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#visits-toggle").check();
  await page.locator("#crashes-toggle").check();
  await page.locator("#sensors-toggle").check();
  await page.locator("#heatmap-snapshot").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 50 crashes",
  );
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.keyboard.down("ArrowUp");
  const result = await page.evaluate(
    () =>
      new Promise<{
        fps: number;
        p95: number;
        frames: number;
        duration: number;
        userAgent: string;
      }>((resolve) => {
        const intervals: number[] = [];
        let previous = performance.now(),
          started = previous;
        function sample(now: number): void {
          intervals.push(now - previous);
          previous = now;
          if (now - started < 10000) requestAnimationFrame(sample);
          else {
            const sorted = [...intervals].sort((a, b) => a - b);
            resolve({
              fps: intervals.length / ((now - started) / 1000),
              p95: sorted[Math.floor(sorted.length * 0.95)],
              frames: intervals.length,
              duration: now - started,
              userAgent: navigator.userAgent,
            });
          }
        }
        requestAnimationFrame(sample);
      }),
  );
  await page.keyboard.up("ArrowUp");
  console.log("RENDER", JSON.stringify(result));
  measurements.push({ render: result });
  expect(result.fps).toBeGreaterThanOrEqual(57);
  expect(result.p95).toBeLessThanOrEqual(22);
});

for (const track of ["sweep", "ribbon", "technical"])
  test(`Gold policy finishes with its exported lap on ${track}`, async ({
    page,
  }) => {
    await page.goto("/e2e/runner.html");
    await page.waitForFunction(
      () => typeof window.validatePolicy === "function",
    );
    const result = await page.evaluate(
      (track) => window.validatePolicy(track, "gold"),
      track,
    );
    expect(result.crashes).toBe(0);
    expect(result.relativeError).toBeLessThanOrEqual(0.02);
    const author = await page.evaluate(
      (track) => window.validatePolicy(track, "final"),
      track,
    );
    expect(result.browserLapTime).toBeGreaterThan(author.browserLapTime);
    measurements.push(result);
  });

for (const drawn of [false, true])
  test(`gate split uses the blue ahead or red behind timing box${drawn ? " on a drawn track" : ""}`, async ({
    page,
  }) => {
    const url = drawn
      ? "/race.html#track=" + (await drawnCode(page))
      : "/race.html";
    await page.goto(url);
    await page.evaluate(async () => {
      // A test-only driver supplies live policy controls to the real race tick.
      // No production state, split times or rendering methods are replaced.
      const modulePath = "/src/race/engine.ts";
      const { RaceEngine } = (await import(
        modulePath
      )) as typeof import("../src/race/engine");
      const tick = RaceEngine.prototype.tick;
      const previous = new WeakMap<
        InstanceType<typeof RaceEngine>,
        import("../src/sim/types").Action
      >();
      RaceEngine.prototype.tick = async function (input) {
        if (
          this.phase === "racing" &&
          this.raceTicks % this.config.physics.action_repeat === 0
        ) {
          const prediction = await this.policy.predict(
            this.observations.build(
              this.player.state,
              this.player.progress.projection,
              previous.get(this) ?? [0, 0],
            ),
          );
          input = prediction.action;
          previous.set(this, input);
        }
        await tick.call(this, input);
      };
    });
    await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
    await page.locator("#start-button").click();
    await expect(page.locator("#split-delta")).toHaveClass(/ahead|behind/, {
      timeout: 20000,
    });
    await expect(page.locator("#checkpoint-label")).toHaveText(
      /CHECKPOINT 0[48]/,
    );
    const ahead = await page
      .locator("#split-delta")
      .evaluate((el) => el.classList.contains("ahead"));
    await expect(page.locator("#split-delta")).toHaveCSS(
      "background-color",
      ahead ? "rgb(47, 124, 246)" : "rgb(214, 49, 44)",
    );
    if (drawn) {
      await expect(page.locator("#result")).toBeVisible({ timeout: 60000 });
      await expect(page.locator("#result-title")).toHaveText("LAP COMPLETE", {
        timeout: 30000,
      });
      await expect(page.locator("#result-ai")).toHaveText(/\d{2}:\d{2}\.\d{3}/);
      await expect(page.locator("#result-description")).not.toContainText(
        "MEDAL",
      );
    }
  });

async function drawnCode(page: import("@playwright/test").Page, name = "loop") {
  const fixture = JSON.parse(
    readFileSync(`../tests/fixtures/editor/${name}.json`, "utf8"),
  );
  await page.goto("/e2e/runner.html");
  return page.evaluate(
    async ({ control, over }) => {
      const { encodeTrack } = await import(
        /* @vite-ignore */ "/src/editor/" + "share.ts"
      );
      return encodeTrack(control, over);
    },
    {
      control: fixture.control,
      over: (fixture.track.crossings ?? []).map(
        (c: { over: number }) => c.over,
      ),
    },
  );
}

test("race a drawn track", async ({ page }) => {
  const code = await drawnCode(page);
  const errors: string[] = [],
    unavailable: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (/\/assets\/.*(?:heatmap|replay)/.test(r.url()))
      unavailable.push(r.url());
  });
  await page.goto("/race.html#track=" + code);
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator(".track-card")).toHaveCount(1);
  await expect(page.locator(".track-card strong")).toHaveText("YOUR TRACK");
  await expect(page.locator(".track-card span")).toHaveText(/\d+ m/);
  await expect(page.locator(".track-card path")).toHaveAttribute("d", /M/);
  await expect(page.locator(".medal-card")).toHaveCount(1);
  await expect(page.locator(".medal-card")).toHaveText("AI · FINAL POLICY");
  await expect(page.locator("#ai-label")).toHaveText("FINAL POLICY");
  await expect(page.locator("#snapshot-note")).toContainText(
    "original controls",
  );
  await page.evaluate(async () => {
    const { RaceEngine } = await import(
      /* @vite-ignore */ "/src/race/" + "engine.ts"
    );
    const tick = RaceEngine.prototype.tick;
    RaceEngine.prototype.tick = async function (
      action: import("../src/sim/types").Action,
    ) {
      await tick.call(this, action);
      document.body.dataset.maxPositiveThrottle = String(
        Math.max(
          Number(document.body.dataset.maxPositiveThrottle ?? 0),
          this.controls[1],
        ),
      );
      if (this.aiLapTime !== null)
        document.body.dataset.completedAiLap = String(this.aiLapTime);
    };
  });
  const captures =
    "../.superpowers/sdd/2026-10-26-m9-track-editor/scratch/task-8";
  mkdirSync(captures, { recursive: true });
  await page.screenshot({
    path: captures + "/prerace-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: captures + "/prerace-mobile.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1050 });
  for (const id of [
    "ghost-toggle",
    "visits-toggle",
    "crashes-toggle",
    "heatmap-snapshot",
  ]) {
    await expect(page.locator("#" + id)).toBeDisabled();
    await expect(page.locator("#" + id)).toHaveAttribute(
      "title",
      "Not available on drawn tracks",
    );
  }
  await expect(page.locator("#sensors-toggle")).toBeEnabled();
  await page.locator("#sensors-toggle").check();
  await expect(page.locator("#back-to-editor")).toHaveAttribute(
    "href",
    "./editor.html#t=" + code,
  );
  await page.locator("#start-button").click();
  await expect(page.locator("#center-message")).toContainText("3");
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await expect(page.locator("#ai-inspector")).toBeVisible();
  // Observe the real tick's completion; a running timer is not evidence of a lap.
  await expect(page.locator("body")).toHaveAttribute(
    "data-completed-ai-lap",
    /\d+/,
    { timeout: 60000 },
  );
  const lap = Number(
    await page.locator("body").getAttribute("data-completed-ai-lap"),
  );
  const maximumThrottle = Number(
    await page.locator("body").getAttribute("data-max-positive-throttle"),
  );
  expect(maximumThrottle).toBeGreaterThan(0);
  expect(maximumThrottle).toBeGreaterThan(0.2);
  expect(lap).toBeGreaterThan(0);
  expect(lap).toBeLessThan(90);
  const milliseconds = Math.round(lap * 1000);
  const controlledLap = `${String(Math.floor(milliseconds / 60000)).padStart(2, "0")}:${String(Math.floor(milliseconds / 1000) % 60).padStart(2, "0")}.${String(milliseconds % 1000).padStart(3, "0")}`;
  await expect(page.locator("#ai-time")).toHaveText(controlledLap);
  const displayedLap = await page.locator("#ai-time").innerText();
  await page.waitForTimeout(700);
  await expect(page.locator("#ai-time")).toHaveText(displayedLap);
  console.log("DRAWN RACE LAP", lap);
  await page.screenshot({
    path: captures + "/race-desktop.png",
    fullPage: true,
  });
  await page.locator("#pause-button").click();
  await expect(page.locator("#race-state")).toHaveText("PAUSED");
  await page.locator(".race-actions .change-race").click();
  await expect(page.locator(".track-card")).toHaveCount(1);
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("START SEQUENCE");
  await expect(page.locator("#ai-time")).not.toHaveText(displayedLap);
  await page.locator("#back-to-editor").click();
  await expect(page).toHaveURL(/editor\.html#t=/);
  await expect(page.locator("#submit")).toBeEnabled();
  expect(errors).toEqual([]);
  expect(unavailable).toEqual([]);
});

test("damaged race link falls back", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/race.html#track=1.broken");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator("#notice")).toContainText(
    "This track link is damaged",
  );
  await expect(page.locator(".track-card")).toHaveCount(3);
  await expect(page.locator("#back-to-editor")).toBeHidden();
  await expect(page.locator("#ghost-toggle")).toBeEnabled();
  await page.locator('[data-track="technical"]').click();
  await expect(page.locator("#circuit-name")).toHaveText("TECHNICAL LOOP");
  expect(errors).toEqual([]);
});

test("invalid drawn geometry falls back to showcase racing", async ({
  page,
}) => {
  const code = await drawnCode(page, "fold");
  await page.goto("/race.html#track=" + code);
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator("#notice")).toContainText(
    "This track link is damaged",
  );
  await expect(page.locator(".track-card")).toHaveCount(3);
});

test("race a drawn overpass track completes a capped AI lap", async ({
  page,
}) => {
  const code = await drawnCode(page, "figure8");
  await page.goto("/race.html#track=" + code);
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator("#snapshot-note")).toContainText(
    "positive throttle capped at 20%",
  );
  await page.evaluate(async () => {
    const { RaceEngine } = await import(
      /* @vite-ignore */ "/src/race/" + "engine.ts"
    );
    const tick = RaceEngine.prototype.tick;
    RaceEngine.prototype.tick = async function (
      action: import("../src/sim/types").Action,
    ) {
      await tick.call(this, action);
      document.body.dataset.maximumThrottle = String(
        Math.max(
          Number(document.body.dataset.maximumThrottle ?? 0),
          this.controls[1],
        ),
      );
      if (this.aiLapTime !== null)
        document.body.dataset.completedAiLap = String(this.aiLapTime);
    };
  });
  const started = Date.now();
  await page.locator("#start-button").click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-completed-ai-lap",
    /\d+/,
    { timeout: 60000 },
  );
  const lap = Number(
    await page.locator("body").getAttribute("data-completed-ai-lap"),
  );
  const throttle = Number(
    await page.locator("body").getAttribute("data-maximum-throttle"),
  );
  expect(lap).toBeGreaterThan(0);
  expect(lap).toBeLessThan(90);
  expect(throttle).toBeGreaterThan(0);
  expect(throttle).toBeLessThanOrEqual(0.2);
  console.log(
    "CAPPED OVERPASS RACE",
    JSON.stringify({
      lap,
      wallSeconds: (Date.now() - started) / 1000,
      maximumThrottle: throttle,
    }),
  );
});

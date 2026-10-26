import { test, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
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
  await page.goto("/");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await expect(page.locator("#track-select option")).toHaveCount(3);
  await expect(page.locator("#snapshot-select option")).toHaveCount(6);
  await page.screenshot({ path: "test-results/m5-ready.png", fullPage: true });
  await page.locator("#start-button").click();
  await expect(page.locator("#center-message")).toContainText("3");
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.keyboard.down("ArrowUp");
  await page.waitForTimeout(1400);
  await page.keyboard.up("ArrowUp");
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
  await page.locator("#track-select").selectOption("technical");
  await expect(page.locator("#circuit-name")).toHaveText("TECHNICAL LOOP");
  await expect(page.locator("#race-state")).toHaveText("GRID READY");
  await page.locator("#snapshot-select").selectOption("0");
  await expect(page.locator("#start-button")).toBeEnabled();
  await expect(page.locator("#ai-label")).toHaveText("0 STEPS");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "test-results/m5-mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});
test("render frame timing during live inference", async ({ page }) => {
  await page.goto("/");
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

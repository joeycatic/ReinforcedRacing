import { test, expect } from "@playwright/test";
test("synchronized ghosts, playback controls, heatmap stages and return to live racing", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("#mode-ghosts")).toBeEnabled({ timeout: 30000 });
  await page.locator("#mode-ghosts").click();
  await expect(page.locator("#system-status")).toHaveText("GHOSTS READY");
  await expect(page.locator("#ghost-lineup>div")).toHaveCount(6);
  await expect(page.locator("#ghost-lineup")).toContainText("6.01M FINAL");
  await expect(page.locator("#race-state")).toHaveText("REPLAY PAUSED");
  await page.locator("#playback-play").click();
  await page.waitForTimeout(250);
  await page.locator("#playback-play").click();
  const before = await page.locator("#playback-time").innerText();
  await page.waitForTimeout(200);
  await expect(page.locator("#playback-time")).toHaveText(before);
  await page.locator("#playback-speed").selectOption("4");
  await page.locator("#playback-restart").click();
  await page.waitForTimeout(250);
  await page.locator("#playback-play").click();
  expect(
    Number((await page.locator("#playback-time").innerText()).split(" / ")[0]),
  ).toBeGreaterThan(0.5);
  await page.locator("#playback-seek").evaluate((el: HTMLInputElement) => {
    el.value = "200";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#race-state")).toHaveText("REPLAY COMPLETE");
  await expect(page.locator('[data-ghost-status="0"]')).toHaveText(
    "TIME LIMIT",
  );
  await expect(page.locator('[data-ghost-status="final"]')).toHaveText(
    "FINISHED",
  );
  await page.locator("#playback-seek").evaluate((el: HTMLInputElement) => {
    el.value = "12";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const modelRequests: string[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith(".onnx")) modelRequests.push(r.url());
  });
  await page.locator("#visits-toggle").check();
  await page.locator("#crashes-toggle").check();
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 0 crashes",
  );
  await page.locator("#heatmap-snapshot").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#heatmap-stage")).toHaveText("0");
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 50 crashes",
  );
  await page.screenshot({
    path: "test-results/m6-ghosts-heatmap.png",
    fullPage: true,
  });
  await page.locator("#heatmap-snapshot").focus();
  await page.keyboard.press("End");
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 0 crashes",
  );
  expect(modelRequests).toEqual([]);
  await page.locator("#track-select").selectOption("technical");
  await expect(page.locator("#circuit-name")).toHaveText("TECHNICAL LOOP");
  await expect(page.locator("#ghost-lineup>div")).toHaveCount(6);
  await expect(page.locator("#playback-time")).toHaveText("0.0 / 200.0 s");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "test-results/m6-mobile.png", fullPage: true });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator("#mode-race").click();
  await expect(page.locator("#system-status")).toHaveText("LIVE POLICY READY");
  await expect(page.locator("#start-button")).toBeVisible();
  await expect(page.locator("#playback-controls")).toBeHidden();
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  expect(errors).toEqual([]);
});
test("discard stale heatmap responses after switching tracks", async ({
  page,
}) => {
  await page.route("**/heatmaps/**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.continue();
  });
  await page.goto("/");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#visits-toggle").check();
  await page.locator("#track-select").selectOption("ribbon");
  await expect(page.locator("#circuit-name")).toHaveText("LONG RIBBON");
  await expect(page.locator("#heatmap-summary")).toContainText("50 runs ·");
  await page.locator("#visits-toggle").uncheck();
  await expect(page.locator("#heatmap-summary")).toHaveText(
    "50 stochastic runs per stage",
  );
});

test("a failed heatmap stays unavailable while the live race still works", async ({
  page,
}) => {
  await page.route("**/heatmaps/**", (route) =>
    route.fulfill({ status: 503, body: "unavailable" }),
  );
  await page.goto("/");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#visits-toggle").check();
  await expect(page.locator("#heatmap-summary")).toHaveText(
    "Heatmap unavailable",
  );
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
});

test("watching ghosts pauses and preserves a race on the same circuit", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.waitForTimeout(100);
  await page.keyboard.press("Escape");
  await expect(page.locator("#race-state")).toHaveText("PAUSED");
  const time = await page.locator("#lap-timer").innerText();
  await page.locator("#mode-ghosts").click();
  await expect(page.locator("#system-status")).toHaveText("GHOSTS READY");
  await page.locator("#playback-play").click();
  await page.waitForTimeout(200);
  await page.locator("#mode-race").click();
  await expect(page.locator("#race-state")).toHaveText("PAUSED");
  await expect(page.locator("#lap-timer")).toHaveText(time);
  await page.locator("#pause-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS");
});

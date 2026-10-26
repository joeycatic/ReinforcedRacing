import { test, expect } from "@playwright/test";
test("discard stale heatmap responses after switching tracks", async ({
  page,
}) => {
  await page.route("**/heatmaps/**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.continue();
  });
  await page.goto("/race.html");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#visits-toggle").check();
  await page.locator('[data-track="ribbon"]').click();
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
  await page.goto("/race.html");
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

test("resuming after Change race reveals the live circuit", async ({
  page,
}) => {
  await page.goto("/race.html");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.locator(".race-actions .change-race").click();
  await expect(page.locator("#prerace")).toBeVisible();
  await expect(page.locator("#pause-button")).toContainText("Resume race");
  await page.locator("#pause-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS");
  await expect(page.locator("#prerace")).toBeHidden();
});

test("race heatmaps retain training-stage controls without loading models", async ({
  page,
}) => {
  await page.goto("/race.html");
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  const models: string[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith(".onnx")) models.push(r.url());
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
  expect(models).toEqual([]);
  await expect(page.locator("#mode-ghosts")).toHaveCount(0);
});

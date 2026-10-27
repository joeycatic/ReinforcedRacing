import { test, expect } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
test("production site works under a GitHub Pages repository prefix with local assets", async ({
  page,
  request,
}) => {
  const errors: string[] = [],
    failed: string[] = [],
    external: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400) failed.push(response.url());
  });
  page.on("request", (request) => {
    if (
      request.url().startsWith("http") &&
      !request.url().startsWith("http://127.0.0.1:4174/")
    )
      external.push(request.url());
  });
  await page.goto("/racing-rl/");
  await expect(page.locator("h1")).toContainText(
    "A CAR THAT TAUGHT ITSELF TO DRIVE",
  );
  await expect(page.locator(".stat")).toHaveCount(4);
  await expect(page.locator("#hero-status")).toHaveText(
    "LIVE · final AI lapping Broad Sweep",
    { timeout: 30000 },
  );
  await page.getByRole("link", { name: "Race the AI", exact: true }).click();
  await expect(page).toHaveURL(/race\.html$/);
  await expect(page.locator("#start-button")).toBeEnabled({ timeout: 30000 });
  await page.locator("#start-button").click();
  await expect(page.locator("#race-state")).toHaveText("RACE IN PROGRESS", {
    timeout: 5000,
  });
  await page.keyboard.down("ArrowUp");
  await page.waitForTimeout(900);
  await page.keyboard.up("ArrowUp");
  expect(Number(await page.locator("#speed").innerText())).toBeGreaterThan(0);
  await page.locator("#visits-toggle").check();
  await page.locator("#crashes-toggle").check();
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 0 crashes",
  );
  await page.locator("#heatmap-snapshot").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#heatmap-summary")).toContainText(
    "50 runs · 50 crashes",
  );
  await page.goto("/racing-rl/watch.html");
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
  await page.locator("#watch-skip").click();
  await expect(page.locator("#attempt-outcome")).toHaveText(/LAP \d+\.\d\d s/);
  await page.screenshot({
    path: "test-results/m6-production.png",
    fullPage: true,
  });
  await page.goto("/racing-rl/editor.html");
  await expect(page.locator("#draw-canvas")).toBeVisible();
  await expect(page.locator("#submit")).toBeDisabled();
  expect(errors).toEqual([]);
  expect(failed).toEqual([]);
  expect(external).toEqual([]);
  expect(existsSync("dist/.nojekyll")).toBe(true);
  expect(existsSync("dist/e2e/runner.html")).toBe(false);
  const manifest = JSON.parse(
    readFileSync("dist/assets/manifest.json", "utf8"),
  );
  for (const snapshot of manifest.snapshots) {
    expect(existsSync("dist/assets/" + snapshot.model.path)).toBe(true);
    for (const file of [
      ...Object.values(snapshot.replays),
      ...Object.values(snapshot.heatmaps),
    ] as { path: string }[])
      expect(existsSync("dist/assets/" + file.path)).toBe(true);
  }
  for (const asset of [
    manifest.site,
    manifest.parity_demo,
    ...manifest.medals.flatMap(
      (m: {
        model: { path: string };
        replays: Record<string, { path: string }>;
      }) => [m.model, ...Object.values(m.replays)],
    ),
  ])
    expect(existsSync("dist/assets/" + asset.path)).toBe(true);
  // A real static host does not rewrite a missing route to the application entry point.
  expect((await request.get("/racing-rl/e2e/runner.html")).status()).toBe(404);
});

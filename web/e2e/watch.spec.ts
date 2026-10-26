import { test, expect, type Page } from "@playwright/test";

async function scene(page: Page) {
  return page.evaluate(() => {
    const app = (
      window as Window & { watchApp?: import("pixi.js").Application }
    ).watchApp;
    const cars: number[] = [],
      trails: string[] = [];
    function visit(node: import("pixi.js").Container) {
      if (!node.visible) return;
      if ("anchor" in node && !("text" in node) && "texture" in node)
        cars.push(node.getBounds().width);
      if (node.label?.startsWith("trail-")) trails.push(node.label);
      node.children.forEach(visit);
    }
    if (app) visit(app.stage);
    return { cars, trails };
  });
}
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    globalThis.__PIXI_APP_INIT__ = (app) => {
      if ("stage" in app)
        (
          window as Window & { watchApp?: import("pixi.js").Application }
        ).watchApp = app;
    };
  });
  await page.goto("/watch.html");
});
test("renderAttempt shows exactly one visible car", async ({ page }) => {
  await expect.poll(async () => (await scene(page)).cars.length).toBe(1);
  const [width] = (await scene(page)).cars;
  expect(width).toBeGreaterThan(6);
  expect(width).toBeLessThan(60);
  await page.screenshot({
    path: "test-results/m8-watch-scene.png",
    fullPage: true,
  });
});
test("renderAttempt shows trails only for earlier attempts", async ({
  page,
}) => {
  await page.locator('[data-attempt="4"]').click();
  await expect
    .poll(async () => (await scene(page)).trails)
    .toEqual(["trail-0", "trail-1", "trail-2", "trail-3", "trail-4"]);
  await page.locator('[data-attempt="1"]').click();
  await expect
    .poll(async () => (await scene(page)).trails)
    .toEqual(["trail-0", "trail-1"]);
});
test("holding hides the car", async ({ page }) => {
  await page.locator("#watch-replay").click();
  await expect(page.locator("#watch-state")).toHaveText("HOLDING", {
    timeout: 10000,
  });
  expect((await scene(page)).cars).toEqual([]);
});

test("plays attempts one at a time", async ({ page }) => {
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
  await page.locator('[data-attempt="3"]').click();
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 4 / 10");
  await expect(page.locator(".segment.done")).toHaveCount(3);
  await page.locator("#watch-skip").click();
  await expect(page.locator("#attempt-kicker")).toContainText(
    "ATTEMPT 10 / 10",
  );
  await expect(page.locator("#attempt-outcome")).toHaveText(/LAP \d+\.\d\d s/);
  await page.locator('[data-attempt="7"]').click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: "test-results/m8-watch.png", fullPage: true });
});
test("rapid track switching shows only the last track", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/replays/attempt-*/**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 300));
    await route.continue();
  });
  await page.locator('[data-track="ribbon"]').click();
  await page.locator('[data-track="technical"]').click();
  await page.locator('[data-track="sweep"]').click();
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
  await expect(page.locator("#watch-canvas canvas")).toHaveAttribute(
    "aria-label",
    /Broad Sweep/,
  );
  await page.waitForTimeout(500);
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
  expect(errors).toEqual([]);
});
test("reduced motion starts paused and explicit play never advances", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await expect(page.locator("#watch-play")).toBeEnabled();
  await page.waitForTimeout(1000);
  await expect(page.locator("#watch-play")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.locator("#watch-play").click();
  await expect(page.locator("#watch-play")).toHaveAttribute(
    "aria-pressed",
    "false",
    { timeout: 10000 },
  );
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
  await page.locator('[data-attempt="3"]').click();
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 4 / 10");
});
test("watch works at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".segment")).toHaveCount(10);
  for (const segment of await page.locator(".segment").all()) {
    await expect(segment).toBeVisible();
    await segment.click();
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/m8-watch-phone.png",
    fullPage: true,
  });
});
test("failed replay shows a notice", async ({ page }) => {
  await page.route("**/replays/attempt-*/**", (r) => r.abort());
  await page.reload();
  await expect(page.locator("#notice")).toBeVisible();
  await expect(page.locator("#notice")).toContainText("Could not load");
  for (const button of await page.locator("#watch-tracks button").all())
    await expect(button).toBeEnabled();
});
test("full playthrough takes 50-70 s", async ({ page }) => {
  test.setTimeout(120000);
  await expect(page.locator("#watch-play")).toBeEnabled();
  await page.locator("#watch-play").click();
  await page.locator("#watch-replay").click();
  const start = Date.now();
  await expect(page.locator("#watch-play")).toHaveText("Play", {
    timeout: 90000,
  });
  const elapsed = (Date.now() - start) / 1000;
  console.log(`Watch wall playthrough: ${elapsed}s`);
  expect(elapsed).toBeGreaterThanOrEqual(50);
  expect(elapsed).toBeLessThanOrEqual(70);
});
test("old race watch URL redirects", async ({ page }) => {
  await page.goto("/race.html#watch");
  await expect(page).toHaveURL(/watch.html$/);
  await expect(page.locator("#mode-ghosts")).toHaveCount(0);
});

test("hidden tab pauses without resuming", async ({ page }) => {
  await expect(page.locator("#watch-play")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(page.locator("#watch-play")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: false,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(page.locator("#watch-play")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
});
test("crashes use nearest lower training stage and stay off by default", async ({
  page,
}) => {
  await expect(page.locator("#crash-toggle")).not.toBeChecked();
  await page.locator("#watch-play").click();
  await page.locator('[data-attempt="3"]').click();
  await page.locator("#crash-toggle").check();
  await expect(page.locator("#crash-stage")).toHaveText(
    "CRASHES · 200K-STEP STAGE · 50 RUNS",
  );
  await page.locator("#watch-skip").click();
  await expect(page.locator("#crash-stage")).toHaveText(
    /CRASHES · 6.01M-STEP STAGE · 50 RUNS/,
  );
});

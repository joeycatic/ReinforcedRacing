import { test, expect } from "@playwright/test";
test("intro shows real stats", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".stat")).toHaveCount(4);
  for (const value of ["98.7%", "85.2%", "0 RL libs", "5e-13"])
    await expect(page.locator("#stats")).toContainText(value);
  await expect(page.locator("#hero-status")).toHaveText(
    "LIVE · final AI lapping Broad Sweep",
    { timeout: 30000 },
  );
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: "test-results/m7-intro.png", fullPage: true });
});
test("intro works at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".stat")).toHaveCount(4);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(
    page.getByRole("link", { name: "Race the AI", exact: true }),
  ).toBeInViewport();
  await expect(
    page.getByRole("link", { name: "Watch it learn", exact: true }),
  ).toBeInViewport();
  await expect(
    page.getByRole("button", { name: /Draw a track/ }),
  ).toBeInViewport();
  await page.screenshot({
    path: "test-results/m7-intro-mobile.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Race the AI", exact: true }).click();
  await expect(page).toHaveURL(/race\.html$/);
  await expect(page.locator("#notice")).toContainText("keyboard or gamepad", {
    timeout: 30000,
  });
});
test("intro survives a failed model load", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/*.onnx", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("h1")).toContainText(
    "A CAR THAT TAUGHT ITSELF TO DRIVE",
  );
  await expect(page.locator(".stat")).toHaveCount(4);
  for (const id of ["ppo", "parity", "ship"])
    await expect(page.locator("#" + id)).toBeVisible();
  await expect(page.locator("#hero-status")).toContainText("STATIC");
  expect(errors).toEqual([]);
});
test("chart has a tooltip and parity is computed live", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#learning-chart svg")).toBeVisible();
  await page.locator("#learning-chart svg").hover();
  await expect(page.locator("#chart-tooltip")).toContainText("steps");
  await expect(page.locator("#parity-demo")).toContainText(
    /0 \(exact match|\d(\.\d+)?e-\d+/,
  );
});
test("intro hero runs the final policy", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#hero-canvas canvas")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.locator("#hero-status")).toHaveText(
    "LIVE · final AI lapping Broad Sweep",
    { timeout: 30000 },
  );
  await expect(page.locator(".static-track")).toBeHidden();
});
test("Watch it learn opens directly in replay mode", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Watch it learn", exact: true }).click();
  await expect(page).toHaveURL(/watch\.html$/);
  await expect(page.locator("#attempt-kicker")).toContainText("ATTEMPT 1 / 10");
});

test("reduced motion stays static and can resume the live hero", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator("#hero-status")).toContainText(
    "STATIC · reduced motion",
  );
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.locator("#hero-status")).toHaveText(
    "LIVE · final AI lapping Broad Sweep",
    { timeout: 30000 },
  );
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator("#hero-status")).toContainText(
    "STATIC · reduced motion",
  );
});
test("intro survives unavailable WASM", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/*.wasm", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator(".stat")).toHaveCount(4);
  await expect(page.locator("#hero-status")).toContainText("STATIC", {
    timeout: 30000,
  });
  await expect(page.locator("#ship")).toBeVisible();
  expect(errors).toEqual([]);
});

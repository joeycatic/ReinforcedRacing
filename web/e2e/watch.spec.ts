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
  await page.getByRole("button", { name: "Attempt 5", exact: true }).click();
  await expect
    .poll(async () => (await scene(page)).trails)
    .toEqual(["trail-0", "trail-1", "trail-2", "trail-3", "trail-4"]);
  await page.getByRole("button", { name: "Attempt 2", exact: true }).click();
  await expect
    .poll(async () => (await scene(page)).trails)
    .toEqual(["trail-0", "trail-1"]);
});
test("holding hides the car", async ({ page }) => {
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#watch-state")).toHaveText("HOLDING", {
    timeout: 10000,
  });
  expect((await scene(page)).cars).toEqual([]);
});

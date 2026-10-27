import { test, expect, type Page } from "@playwright/test";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
const code = async (page: Page, name: string, flip = false) => {
  const f = JSON.parse(
    readFileSync(`../tests/fixtures/editor/${name}.json`, "utf8"),
  );
  if (page.url() === "about:blank") await page.goto("/e2e/runner.html");
  return page.evaluate(
    async ({ control, over }) => {
      const { encodeTrack } = await import(
        /* @vite-ignore */ "/src/editor/" + "share.ts"
      );
      return encodeTrack(control, over);
    },
    {
      control: f.control,
      over: (f.track.crossings ?? []).map((c: { over: number }) =>
        flip ? 1 - c.over : c.over,
      ),
    },
  );
};
const pageErrors = new WeakMap<Page, string[]>();
test.beforeEach(({ page }) => {
  const errors: string[] = [];
  pageErrors.set(page, errors);
  page.on("pageerror", (e) => errors.push(e.message));
});
test.afterEach(({ page }) => expect(pageErrors.get(page)).toEqual([]));
const captures =
  "../.superpowers/sdd/2026-10-26-m9-track-editor/scratch/task-7";
async function loop(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  const b = (await page.locator("#draw-canvas").boundingBox())!;
  const r = Math.min(b.width, b.height) * 0.34;
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * Math.PI * 2;
    await page.mouse.move(
      b.x + b.width / 2 + r * Math.cos(t),
      b.y + b.height / 2 + r * Math.sin(t),
    );
    if (!i) await page.mouse.down();
  }
  await page.mouse.up();
}
test("draw a loop and the AI laps it", async ({ page }) => {
  await page.goto("/editor.html");
  await loop(page);
  await expect(page.locator("#submit")).toBeEnabled();
  await expect(page.locator("#trial-policy-note")).toContainText(
    "original controls",
  );
  const started = Date.now();
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toHaveText(/LAP \d:\d\d\.\d{3}/, {
    timeout: 60000,
  });
  const outcome = {
    lap: await page.locator("#trial-result").innerText(),
    wallSeconds: (Date.now() - started) / 1000,
  };
  console.log("RAW drawn-loop", JSON.stringify(outcome));
  mkdirSync(captures, { recursive: true });
  writeFileSync(
    captures + "/raw-drawn-loop.json",
    JSON.stringify(outcome, null, 2) + "\n",
  );
});
test("figure-eight AI lap crosses the overpass", async ({ page }) => {
  const sharedCode = await code(page, "figure8");
  mkdirSync(captures, { recursive: true });
  writeFileSync(captures + "/figure8-share-code.txt", sharedCode + "\n");
  await page.goto("/editor.html#t=" + sharedCode);
  await expect(page.locator(".trial-step")).toContainText(
    "positive throttle capped at 20%",
  );
  await expect(page.locator("#track-stats")).toContainText("1 OVERPASS");
  mkdirSync(captures, { recursive: true });
  await page.screenshot({
    path: captures + "/figure8-desktop.png",
    fullPage: true,
  });
  const started = Date.now();
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toHaveText(/LAP \d:\d\d\.\d{3}/, {
    timeout: 60000,
  });
  const outcome = {
    lap: await page.locator("#trial-result").innerText(),
    wallSeconds: (Date.now() - started) / 1000,
  };
  console.log("CAPPED figure8", JSON.stringify(outcome));
  mkdirSync(captures, { recursive: true });
  writeFileSync(
    captures + "/capped-figure8.json",
    JSON.stringify(outcome, null, 2) + "\n",
  );
  await page.screenshot({
    path: captures + "/figure8-capped-lap.png",
    fullPage: true,
  });
});
test("invalid drawing is blocked", async ({ page }) => {
  await page.goto("/editor.html");
  await page.evaluate(() => document.fonts.ready);
  const b = (await page.locator("#draw-canvas").boundingBox())!;
  await page.mouse.move(b.x + 25, b.y + 40);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width - 25, b.y + 150, { steps: 30 });
  await page.mouse.up();
  await expect(page.locator("#issues")).toContainText(
    "Finish near the start to close the loop",
  );
  await expect(page.locator("#submit")).toBeDisabled();
  await page.goto("/editor.html#t=" + (await code(page, "shallow")));
  await expect(page.locator("#issues")).toContainText(
    "Roads cross at too shallow an angle",
  );
  await expect(page.locator("#submit")).toBeDisabled();
});
test("tight corner warns but allows", async ({ page }) => {
  await page.goto("/editor.html#t=" + (await code(page, "tight")));
  await expect(page.locator("#issues")).toContainText(
    "Too tight — the AI never trained",
  );
  await expect(page.locator("#submit")).toBeEnabled();
  mkdirSync(captures, { recursive: true });
  await page.screenshot({
    path: captures + "/tight-desktop.png",
    fullPage: true,
  });
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toHaveText(
    /LAP|CRASHED AT|DID NOT FINISH/,
    { timeout: 60000 },
  );
  await expect(page.locator("#race-it")).toHaveAttribute(
    "href",
    /^\.\/race.html#track=1\./,
  );
});
test("flip survives sharing", async ({ page }) => {
  const original = await code(page, "figure8");
  await page.goto("/editor.html#t=" + original);
  await page.evaluate(() => document.fonts.ready);
  const b = (await page.locator("#draw-canvas").boundingBox())!;
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
  const crossing = page.locator("#crossing-controls button").first();
  await crossing.focus();
  await page.keyboard.press("Enter");
  await expect(crossing).toBeFocused();
  await expect(crossing).toHaveAttribute("aria-label", /later pass on top$/);
  await page.keyboard.press("Space");
  await expect(crossing).toBeFocused();
  await expect(crossing).toHaveAttribute("aria-label", /earlier pass on top$/);
  await page.locator("#share").click();
  await expect(page.locator("#share")).toBeFocused();
  const hash = await page.evaluate(() => location.hash);
  expect(hash).toBe("#t=" + (await code(page, "figure8", true)));
  await page.reload();
  await expect(page.locator("#track-stats")).toContainText("1 OVERPASS");
  await page.locator("#share").click();
  expect(await page.evaluate(() => location.hash)).toBe(hash);
});
test("bad share link", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/editor.html#t=1.!!!!");
  await expect(page.locator("#issues")).toContainText(
    "This track link is damaged",
  );
  await expect(page.locator("#submit")).toBeDisabled();
  expect(errors).toEqual([]);
});
test("draw on a phone", async ({ browser }) => {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });
  const page = await context.newPage();
  await page.goto("/editor.html");
  await page.evaluate(() => document.fonts.ready);
  const b = (await page.locator("#draw-canvas").boundingBox())!;
  const client = await context.newCDPSession(page),
    r = Math.min(b.width, b.height) * 0.34;
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * Math.PI * 2;
    await client.send("Input.dispatchTouchEvent", {
      type: i === 0 ? "touchStart" : "touchMove",
      touchPoints: [
        {
          x: b.x + b.width / 2 + r * Math.cos(t),
          y: b.y + b.height / 2 + r * Math.sin(t),
        },
      ],
    });
  }
  await client.send("Input.dispatchTouchEvent", {
    type: "touchEnd",
    touchPoints: [],
  });
  await expect(page.locator("#submit")).toBeEnabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  mkdirSync(captures, { recursive: true });
  await page.screenshot({ path: captures + "/phone.png", fullPage: true });
  await context.close();
});
test("clear wipes the drawing", async ({ page }) => {
  await page.goto("/editor.html#t=" + (await code(page, "figure8")));
  await page.locator("#clear").click();
  await expect(page.locator("#submit")).toBeDisabled();
  await expect(page.locator("#track-stats")).toHaveText("YOUR NEXT CIRCUIT");
  await expect(page.locator("#issues")).toBeEmpty();
});
test("clear cancels a pending model load", async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>((r) => (release = r));
  await page.route("**/*.onnx", async (route) => {
    await held;
    await route.continue();
  });
  await page.goto("/editor.html#t=" + (await code(page, "loop")));
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toContainText("LOADING");
  await page.locator("#clear").click();
  release();
  await page.waitForTimeout(2000);
  await expect(page.locator("#trial-result")).toBeEmpty();
  await expect(page.locator("#trial canvas")).toHaveCount(0);
  await expect(page.locator("#submit")).toBeDisabled();
});
test("replacement drawing cancels a pending trial", async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>((r) => (release = r));
  await page.route("**/*.onnx", async (route) => {
    await held;
    await route.continue();
  });
  await page.goto("/editor.html#t=" + (await code(page, "figure8")));
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toContainText("LOADING");
  await loop(page);
  release();
  await page.waitForTimeout(1000);
  await expect(page.locator("#trial-result")).toBeEmpty();
  await expect(page.locator("#trial canvas")).toHaveCount(0);
  await expect(page.locator("#track-stats")).toContainText("0 OVERPASSES");
  await expect(page.locator("#submit")).toBeEnabled();
});
test("clear stops a running trial", async ({ page }) => {
  await page.goto("/editor.html#t=" + (await code(page, "loop")));
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toContainText("DRIVING", {
    timeout: 30000,
  });
  await expect(page.locator("#submit")).toBeDisabled();
  await page.locator("#clear").click();
  await page.waitForTimeout(300);
  await expect(page.locator("#trial canvas")).toHaveCount(0);
  await expect(page.locator("#trial-result")).toBeEmpty();
  await expect(page.locator("#share")).toBeDisabled();
});
test("clipboard fallback selects the complete share URL", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: () => Promise.reject(Error("Unavailable")) },
    });
  });
  await page.goto("/editor.html#t=" + (await code(page, "loop")));
  await page.locator("#share").click();
  await expect(page.locator("#share-fallback")).toBeVisible();
  await expect(page.locator("#share-url")).toHaveValue(
    await page.evaluate(() => location.href),
  );
  expect(
    await page
      .locator("#share-url")
      .evaluate(
        (input: HTMLInputElement) =>
          input.selectionEnd! - input.selectionStart!,
      ),
  ).toBeGreaterThan(20);
});
test("model load failure keeps the drawing usable", async ({ page }) => {
  await page.route("**/*.onnx", (r) => r.abort());
  await page.goto("/editor.html#t=" + (await code(page, "loop")));
  await page.locator("#submit").click();
  await expect(page.locator("#trial-result")).toContainText(
    "Trial unavailable",
  );
  await expect(page.locator("#submit")).toBeEnabled();
  await expect(page.locator("#share")).toBeEnabled();
});
test("bridge and tight warning stay legible", async ({ page }) => {
  await page.goto("/e2e/runner.html");
  const fixture = JSON.parse(
    readFileSync("../tests/fixtures/editor/figure8.json", "utf8"),
  );
  const encoded = await page.evaluate(async (control) => {
    const { buildEditorTrack } = await import(
      /* @vite-ignore */ "/src/editor/" + "build.ts"
    );
    const { encodeTrack } = await import(
      /* @vite-ignore */ "/src/editor/" + "share.ts"
    );
    const { default: config } = await import(
      /* @vite-ignore */ "/src/editor/" + "editor-config.json"
    );
    for (const factor of [
      1, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
    ])
      for (const extra of [0, 5, 10, 20, 30, 40, 50, 60, 80, 100]) {
        const candidate = control.map((p: number[]) => [p[0], p[1] * factor]);
        candidate[0][0] += extra;
        const built = buildEditorTrack(candidate, config);
        if (
          !built.issues.length &&
          built.tight.length &&
          built.track.data.crossings?.length === 1
        )
          return encodeTrack(candidate, [1]);
      }
    throw Error("No valid warning preview candidate");
  }, fixture.control);
  await page.goto("/editor.html#t=" + encoded);
  await expect(page.locator("#track-stats")).toContainText("1 OVERPASS");
  await expect(page.locator("#issues")).toContainText("Too tight");
  await expect(page.locator("#submit")).toBeEnabled();
  mkdirSync(captures, { recursive: true });
  await page.screenshot({ path: captures + "/m9-editor.png", fullPage: true });
  await page.screenshot({ path: "test-results/m9-editor.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: captures + "/figure8-tight-phone.png",
    fullPage: true,
  });
});

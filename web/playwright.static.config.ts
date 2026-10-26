import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "e2e",
  outputDir: "test-results/static",
  testMatch: "static.spec.ts",
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: "http://127.0.0.1:4174",
    viewport: { width: 1366, height: 900 },
    headless: true,
    launchOptions: { args: ["--enable-unsafe-swiftshader"] },
  },
  webServer: {
    command: "node e2e/serve-static.mjs",
    url: "http://127.0.0.1:4174/racing-rl/",
    reuseExistingServer: false,
  },
});

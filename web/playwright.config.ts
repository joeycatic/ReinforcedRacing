import { defineConfig } from "@playwright/test";
const port = Number(process.env.E2E_PORT ?? 5173);
export default defineConfig({
  testDir: "e2e",
  testIgnore: "static.spec.ts",
  timeout: 60000,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 1050 },
    headless: true,
    launchOptions: { args: ["--enable-unsafe-swiftshader"] },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `npm run dev -- --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
  },
});

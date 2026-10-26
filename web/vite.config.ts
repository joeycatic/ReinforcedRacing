import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { copyFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
const runtime = fileURLToPath(
  new URL("./node_modules/onnxruntime-web/dist/", import.meta.url),
);
export default defineConfig({
  base: "./",
  resolve: {
    conditions: [
      "onnxruntime-web-use-extern-wasm",
      "module",
      "browser",
      "development|production",
    ],
  },
  plugins: [
    {
      name: "local-onnx-wasm",
      buildStart() {
        if (
          !existsSync(
            fileURLToPath(
              new URL("./public/assets/manifest.json", import.meta.url),
            ),
          )
        )
          throw new Error(
            "Missing race assets. Run make export before starting or building the web app.",
          );
        const target = fileURLToPath(
          new URL("./public/runtime/", import.meta.url),
        );
        mkdirSync(target, { recursive: true });
        for (const file of [
          "ort-wasm-simd-threaded.mjs",
          "ort-wasm-simd-threaded.wasm",
        ])
          if (
            !existsSync(target + file) ||
            !readFileSync(runtime + file).equals(readFileSync(target + file))
          ) {
            copyFileSync(runtime + file, target + file);
          }
      },
    },
  ],
  server: { host: "127.0.0.1" },
  build: {
    rollupOptions: {
      input: {
        intro: fileURLToPath(new URL("./index.html", import.meta.url)),
        race: fileURLToPath(new URL("./race.html", import.meta.url)),
      },
    },
    target: "es2022",
    chunkSizeWarningLimit: 800,
  },
});

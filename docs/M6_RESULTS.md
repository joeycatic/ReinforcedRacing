# Ghosts, heatmaps and static delivery (M6)

M6 completes the MVP with synchronized ghost playback, training-stage heatmap layers,
drift effects, the AI-vision stretch feature and a complete README. The browser still
uses the unchanged v1 simulation and M4 exports. Work is on `codex/m6-ghosts-heatmaps`,
based on M5 commit `fabadec`.

## Behavior

- **Watch it learn:** all six scheduled ghosts share one start and one playback clock.
  Cars are translucent, cool-to-warm by training stage and labeled with actual counts.
  Play/pause, restart, 0.5×–4× speed and seeking sample the original quantized recordings.
  A ghost holds its final pose and reports finished, crashed or time-limit status.
- **Analysis layers:** visitation cells, crash density and raw crash points can be shown
  in race or replay view. A separate stage slider chooses the displayed heatmap without
  fetching another model or filtering the synchronized ghost fleet. The summary reports
  real trial, visit and crash counts. A common logarithmic scale across all stages of a
  track prevents per-snapshot contrast changes from concealing training differences.
- **Loading and failure:** hashes/versions/identities use the existing loader. Late
  responses cannot overwrite a new track/stage. Old layers clear while loading; a failed
  layer is marked unavailable while the live race remains usable. Successful heatmaps
  are cached per track; failed fetches can be retried.
- **Race continuity:** leaving race pauses its clock and clears held input. Returning
  to the same track/opponent preserves the paused race. Changing either selection
  starts a fresh race. Replay time freezes when the tab is hidden.
- **AI vision:** rays use the actual last observation and pose used for inference;
  output bars show its actual mean controls. The raw critic value is shown beside a
  bounded tanh(value/10) display meter. It uses training reward units.
- **Visual effects:** lateral slip adds rear tire marks and light dust. Geometry is
  recycled in small batches, particles have a fixed lifetime/cap, and reduced motion
  disables emission. Effects, camera and playback never change physics or policy input.

## Verification

```sh
make test
make web-build
npm run --prefix web test:e2e
npm run --prefix web test:static
```

The unit suite checks playback synchronization/rates/pause/seek/end states, angle-wrap
interpolation, metadata labels/colors, world-to-texture row mapping, transparent empty
cells and shared density scaling. Existing 600-tick physics/observation fixtures and
all Python tests remain in the same `make test` run.

Real Chromium checks run the live WASM policy on all three tracks, exercise race
controls, play/pause/speed/seek, all six ghost outcomes, heatmap stage selection,
track changes, delayed/failed heatmap responses, preserving a paused race, and a
390×844 layout without horizontal overflow. The render check keeps both heatmap
layers and AI vision enabled during live inference. It retains the original
≥57 FPS / ≤22 ms p95 acceptance checks. Initial unoptimized overlays measured about
50 FPS; precomposed textures and batched marks restored the check without changing
thresholds. Current measurements are recorded in `m6-browser-results.json`.

Final results: **86 pytest tests, 30 Vitest tests, 9 development-browser tests and
1 production-static test passed**; Ruff, strict TypeScript and `make web-build`
also passed. The live render check measured **59.60 FPS**, **16.70 ms p95** over
597 frames / 10.016 seconds at 1440×1050 with both heatmaps and AI vision enabled.

The final-policy browser lap times remain 24.225082 / 24.652338 / 19.621723 seconds
for Broad Sweep / Long Ribbon / Technical Loop, respectively; all are within 2% of
Python's deterministic reference. Capturing sensor telemetry and adding overlays do
not alter the policy or physical integration.

## Static deployment proof

`npm run test:static` builds `web/dist/` and serves exactly that directory under a
`/racing-rl/` repository prefix. It runs a live keyboard-driven race, opens all six
ghosts, plays them and changes both heatmap layers/stages. It checks:

- No page errors, failed requests or external HTTP requests.
- All manifest-listed models, replays and heatmaps are present.
- Fonts and WASM load through the repository prefix without a CDN or isolation headers.
- `.nojekyll` is present and development acceptance helpers are absent from the build.
- A missing route returns 404 instead of being masked by a development SPA fallback.

The compiled directory is ready for publication as a GitHub Pages root. Publication
itself is not performed. See the main README for Pages configuration and recording
instructions. As before, local/retained training artifacts are required to rebuild
the asset package; generated bundles are intentionally ignored by Git.

## Remaining checks

Actual mid-range laptop performance, physical gamepad behavior and subjective keyboard
feel still need hardware/human checks. The automated rendering measurements use this
host's i5-11400F and Chromium 156 with SwiftShader software WebGL. Screenshots are real
browser captures. GIF recording slots remain explicit documentation placeholders, as
requested by the spec; no footage is fabricated.

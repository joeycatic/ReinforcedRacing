# Live browser race (M5)

M5 adds the static Vite/TypeScript race interface with PixiJS rendering and a live
ONNX Runtime Web opponent. Physics/track/observation versions remain v1. M4's source
simulation and exports are unchanged; this milestone adds their browser integration.

```sh
make setup
make export                 # retained seed-2 run by default; --run-dir selects another complete run
make web-dev                # open the printed localhost URL
make web-build              # static output: web/dist/
make test                   # Python, TS types, parity, loaders and race engine
cd web
npx playwright install chromium
npm run test:e2e            # real WASM laps, controls/responsive layout, 10-second render sample
```

Select one of three circuits and any scheduled policy snapshot. Start a 3–2–1
countdown, then drive with arrows/WASD; Space brakes/reverses, R respawns at the last
checkpoint, Enter restarts, and Escape pauses. Standard gamepad stick/triggers also
work. A full first lap produces a result and rematch button. If the AI finishes first,
it waits at the finish while the player completes their lap. The optional reference
ghost is a separate replay overlay, disabled by default. Orange is always the live AI;
mint is the player. Both cars can pass through one another.

The HUD displays a lap clock, speed, checkpoint splits against the live AI, progress,
and AI lap time. Camera follow/zoom and car interpolation are presentation only.
Pausing/tab hiding stops simulation time; wall contact and manual respawn retain
elapsed lap time. The app loads all race assets through M4's version/hash-checking
loader. Selecting a new opponent loads and warms its ONNX graph before countdown.

## Acceptance measurements

The final seed-2 policy completes all three circuits without crashing in Chromium
using actual ONNX Runtime Web WASM. Reference times come from M4's deterministic
Python evaluation, verified against the replay's checkpoint hash. The checked browser
harness uses the same `RaceEngine`, `OnnxPolicy`, observation builder, and physics as
the game; it does not replay recorded controls or positions.

| Track | Python mean-policy lap | Browser live-policy lap | Difference |
| --- | ---: | ---: | ---: |
| Broad Sweep | 24.246155 s | 24.225082 s | 0.0869% |
| Fast Ribbon | 24.591262 s | 24.652338 s | 0.2484% |
| Technical Loop | 19.812376 s | 19.621723 s | 0.9623% |

All differences are below the required **2%**. Browser inference counts equal
ceil(physics ticks / 2): 727 / 740 / 589, respectively. These are closed-loop lap
comparisons; paths are not bit-identical. Maximum position differences from the
quantized Python ghosts were 0.86 / 2.60 / 7.21 m. Separately, an exploratory check of
120 actual observation vectors (40 per track) found a maximum browser-vs-PyTorch
action difference of 7.15e-7. Golden state/observation parity remains ≤1e-6 for
identical scripted inputs. See `m5-browser-results.json` for the browser measurements.

With live inference running, a 1440×1050 headless Chromium test measured approximately
**60 FPS**, with a **16.8 ms 95th-percentile frame interval**, over ten seconds. Host:
Intel Core i5-11400F, Linux, Chromium 156, software WebGL rendering (SwiftShader),
single-threaded WASM, device scale 1. Disabling WebGL multisample antialiasing removed
a measured rendering bottleneck (~40 FPS with it enabled). Font files and runtime
WASM are served locally. No CDN or cross-origin-isolation headers are required.

The automated layout check uses a 390×844 viewport and asserts no horizontal overflow.
Browser checks cover policy/track selection, countdown, keyboard acceleration,
pause-clock freezing, resume, respawn and restart. Unit tests drive a complete real
lap, compare render-frequency independence, capture splits, reject stale predictions,
and reset a rematch. M4 fixtures continue to test the unchanged simulation.

## Limits and next work

The frame measurement is on this host, not an actual mid-range laptop; laptop/browser
hardware and physical gamepad checks remain. Keyboard feel also benefits from human
playtesting. Small screens require a keyboard or gamepad; touch steering is not part
of the spec. Early snapshots may stall or crash because their policies are untrained.

M6 remains: multi-snapshot time-lapse playback, heatmap layers/slider, drift particles
and final presentation/documentation polish. Browser acceptance helpers under
`web/e2e/` are development-server-only and are not production build entry points.

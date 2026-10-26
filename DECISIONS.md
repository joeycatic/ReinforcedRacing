# M1 decisions

- Treat the existing lowercase `spec.md` as the master spec; `SPEC.md` in AGENTS.md refers to that file, so no duplicate specification is created.
- Stop at M1; defer the physics-parity skill's TypeScript port, golden state/observation fixtures, Vitest, and manifest checks to M4 as explicitly scheduled by the user's spec; Python is the versioned source of truth until then.
- Use Python floats and NumPy float64 in meters/radians with x-right/y-up and positive steering to the left, to give the future browser port a single clear coordinate convention.
- Store the rate-limited normalized steering input in `CarState` in addition to the six rigid-body fields, because actuator history is required for deterministic continuation.
- Regularize tire slip using a configurable minimum wheel longitudinal speed instead of a second kinematic model, to keep one continuous force model stable at rest and in reverse.
- Use static axle loads and independent lateral friction clamps, without load transfer or a friction-circle model, to keep M1's bicycle model simple; slightly lower rear friction permits recoverable oversteer.
- Negative throttle brakes while forward velocity exceeds the configured near-stop threshold, then engages speed-limited reverse; Space uses this same braking input because a separate handbrake is not required in M1.
- Interpret the spec's wall rule literally: contact occurs when the car center's distance from the road centerline exceeds `(track_width + car_width) / 2`, allowing a half-car-width excursion beyond the edge.
- Generate polar control points with uniform closed Catmull–Rom interpolation; use 1 m arc samples and a shorter closing segment, omitting duplicate endpoints to avoid degenerate segments.
- Validate centerline/boundary intersections (including touching and collinear overlap), local road folding, and curvature limited by both track config and the car's minimum steering radius; retries consume the same seeded RNG in a bounded loop.
- Permit per-showcase radius/noise/aspect overrides alongside fixed seeds, to produce broad, elongated, and compact routes with the same generator.
- Keep checkpoint ordinals monotonic and unwrapped; interpolate finish crossing time within its physics tick, and never award progress for respawn or start-line oscillation.
- Start the first lap timer at session creation/restart, include crash delays and manual respawns in lap time, and preserve completed laps on respawn; countdowns belong to M5.
- Use an accumulator driven by wall time only in pygame; simulation time is always integer physics ticks, with inputs held for two ticks; cap catch-up after a long UI stall to avoid a spiral of death.
- Provide `make tracks` for M1 JSON geometry export; required Make targets for future milestones fail explicitly instead of pretending to implement unavailable training, evaluation, model export, or web features.
- Use NumPy for geometry, PyYAML for typed config loading, pygame for manual driving, pytest for behavior checks, and Ruff for lint/format; pin resolved versions in uv.lock and use Python 3.12 by default (source supports 3.11+).

# M2 decisions

- Implement only the newly authorized M2 and continue deferring browser parity to M4; the existing scalar physics remains unchanged, with 600-tick regression checks against the native NumPy batch implementation.
- The listed observation components add up to 21 although the spec requires 22; append the existing rate-limited steering actuator as index 21 in `obs.v1` to expose otherwise hidden steering lag while preserving the listed component order.
- Normalize vx/vy/yaw by 80 m/s, 40 m/s, and 3 rad/s, and curvature by 0.1 m⁻¹; clip normalized values to [-1,1] so Gymnasium's declared observation bounds are reliable, including terminal states.
- Use the stored tangent at the projected segment index for heading error (tangent minus car heading), and linearly interpolate signed curvature at wrapped lookahead arcs so the layout and signs are explicit for M4.
- Precompute a uniform 8 m spatial grid of boundary segments and traverse crossed cells exactly with batched DDA; grid construction may loop over static geometry, but physics, projection, rays, checkpoints, and reset updates never loop over individual cars.
- Keep `RaceEnv` on the scalar M1 session as an independent reference, and subclass Gymnasium VectorEnv for `VecRaceEnv`; both expose float64 spaces, honor two physics ticks per action, and stop at the first tick that crashes or reaches the lap limit.
- Charge the time reward once per agent step (including an early crash) and log each weighted contribution plus episode totals; crash termination takes precedence over a simultaneous time/lap truncation.
- Default the episode cap to 6000 agent steps (200 simulation seconds) and two laps in `configs/env.yaml`; no extra stall or wrong-way termination is added beyond the spec.
- Single-car Gymnasium episodes require explicit reset; vector episodes reset during the finishing step, sampling tracks uniformly with replacement and preserving terminal observations/metrics in final_obs/final_info with boolean masks for correct future PPO bootstrapping.
- Root vector info describes the transition before autoreset and reset_track_index identifies the next episode's selected track; info/observations are independent snapshots rather than aliases to mutable environment buffers.
- Use a single seeded NumPy generator for vector track assignments; determinism is defined for the same seed, ordered pool, batch size and action sequence, not across different batch sizes or permutations.
- Reserve seeds 10000–10199 for the 200 training tracks and 20000–20049 for the 50 held-out tracks, disjoint from showcase seeds; construct pools explicitly once, with the three showcases as the quick default for environment constructors.
- Benchmark full transitions with 64 cars and 64 generated training tracks, pre-generated held random controls, 100 warmup batches and three 1000-batch samples; exclude one-time geometry/grid setup and include crash autoreset, sensor construction, rewards and info.
- Add Gymnasium as the only new dependency for M2, because its standard environment contract and checker are explicit milestone requirements; PPO remains unimplemented until M3.

# M3 decisions

- Implement PPO and the summary evaluator required by M3 acceptance; leave replay/heatmap generation, ONNX, TypeScript parity, and web assets for M4, without altering M1/M2 simulation behavior.
- Use PyTorch solely for neural networks, distributions, automatic differentiation and Adam, and TensorBoard for logging; training imports no reinforcement-learning framework.
- Keep network tensors float32 while physics and running-statistic accumulation remain float64; normalize observations in NumPy before converting to network tensors, and freeze statistics for evaluation.
- Compute Gaussian likelihoods on unclipped samples, clipping only environment inputs; advantage normalization is per minibatch, and the critic/actor use separate two-hidden-layer networks.
- Bootstrap a truncated transition from the critic's final-observation value, bootstrap no crash, and cut GAE recursion at both termination and truncation so rewards cannot leak across autoresets.
- Scale rewards using the running variance of discounted returns, without mean subtraction; clear each discounted-return accumulator after its episode ends, preserving it across rollout/checkpoint boundaries.
- Save scheduled snapshots at the first complete vector step at/after the requested count, recording actual and requested counts separately; policies update only at rollout boundaries and final training steps round up to a complete rollout.
- Keep policy-only ghost snapshots distinct from full resumable checkpoints; both save observation and return statistics, while resumable checkpoints also save Adam, CPU/CUDA/MPS RNG state, shuffle RNG, live environment arrays, environment RNG, track-geometry hash and simulation configuration.
- Resume at an update boundary and reject incompatible versions, geometry/order or settings; allow changing device/CPU threads or extending the total-step horizon, with exact same-device continuation regression tests.
- Require ≥95% first-lap success over eight seeded stochastic runs and at least 200k steps before moving from `sweep` to the configured 200-track pool; keep policy, optimizer and normalizers across the curriculum transition, resetting only live episodes/discounted returns.
- Define each M3 evaluation trial as one lap from rest (a one-lap truncation using the unchanged environment), with crashes/timeouts as failures; default to stochastic trials, offer deterministic policy-mean trials separately, and report success/crash rates plus times conditional on success.
- Use a separate seeded NumPy evaluation RNG for policy noise so evaluation cannot alter training's PyTorch RNG stream; never update normalization statistics or optimizer state during evaluation.
- Keep generated checkpoints/logs under ignored artifacts/ and commit source configs plus quantitative evaluation reports, rather than adding transient training output to Git.
- After checking metrics, normalization outliers, scripted rewards, action distributions, GAE/reset handling, and the successful single-track curriculum, extend only the total-step budget from 3M to 6M and compare two seeds; preserve the original baseline configs and checkpoint evidence, and use 6M as the new default without changing the specified PPO hyperparameters.
- Validate stochastic completion with 50 trials per showcase and five trials per held-out track, and evaluate deterministic means separately; select seed 2's final checkpoint as the reference because it passes both modes, explicitly retaining seed 1's late deterministic technical-track failure in the report.

# M4 decisions

- Implement only evaluation/export/parity for M4. Use the retained seed-2 run as the default export source; require every scheduled checkpoint and fail explicitly when one is missing. Keep generated bundles and evaluation caches ignored alongside training artifacts.
- Evaluate one lap from rest per trial. Record 50 stochastic trials per showcase/snapshot and five per held-out track, with one separate policy-mean showcase ghost. Include the initial ghost pose and every physics tick through its finish/crash/timeout; do not include subsequent autoreset episodes.
- Quantize absolute samples before delta encoding, using centimeters, milliradians and 1e-4 control units. Use sparse, sorted flat-grid counts for heatmaps with a shared per-track 0.5 m grid and 5 m safety margin; include crash terminal ticks in visitation and crash density.
- Export raw float64 observations with the frozen float64 normalizer embedded, then cast to the original float32 MLP. Return unclipped policy means and critic values; browser controls must clip the means just as Python evaluation does. Verify both outputs at the fixed 1e-5 tolerance, including single-car inference batches.
- Load canonical track JSON in TypeScript rather than reproduce NumPy RNG/track generation in JS. Port all runtime geometry queries, progress, physics, session and observation algorithms without changing Python simulation behavior or v1 versions, preserving trained checkpoint compatibility.
- Check the five manifest versions before dependent downloads, and include byte counts/SHA-256 for every listed asset. Stage and verify the entire bundle before replacing the prior export. Evaluation cache keys include checkpoint/source/config/geometry hashes and cache reuse verifies output hashes.
- Test 600-tick integrated play sessions on all showcases plus isolated unrestricted physics/observation kernels and progress/ray probes. A free-body stress car 44 m outside the road exposed a shared-vertex segment-selection tie amplified by tiny libm position differences; test that projection kernel with identical reference coordinates while keeping all integrated session state/progress/observation checks unchanged. Do not add an epsilon tie rule, alter the simulation, or loosen the 1e-6 tolerance.
- Use locked TypeScript/Vitest dependencies and npm's legacy-peer-deps resolver setting to avoid npm 10's optional-peer resolver crash. Keep race rendering and browser policy inference integration in M5.

# M5 decisions

- Implement M5's single-lap live race only, after the user's explicit continuation request. Keep time-lapse playback, heatmap controls and M6 polish separate. Work on `codex/m5-live-race`, based on the completed M4 commit.
- Run the unchanged float64 TypeScript simulation with a fixed 60 Hz accumulator and hold controls for two ticks. Use single-threaded ONNX Runtime WASM with raw float64 observations and the exported normalizer; sample no policy noise. Inference is awaited before advancing either car, and restart/pause invalidates a pending prediction.
- Use PixiJS for rendering, Vite for static bundling, ONNX Runtime Web for the required live opponent, Playwright for actual-browser acceptance, and self-hosted Fontsource fonts for legible timing/display type. No application framework or backend is needed. Bundle runtime WASM locally through the package's external-WASM condition to avoid CDN dependencies and duplicate binaries.
- Keep wall-clock deltas in the render accumulator only, cap catch-up at eight physics ticks, and pause on tab hiding. Countdown uses 180 fixed ticks without advancing cars or lap clocks. A long stall discards excess wall time rather than changing dt or skipping simulated ticks.
- Run both cars independently with play-mode checkpoint respawn. Stop each AI at its first completed lap; finish the race when the player crosses the line, comparing recorded first-lap times (or declaring the AI unfinished if the player finishes first). Cars share the same start and never collide.
- Record checkpoint splits using the existing within-tick crossing interpolation and compare the latest checkpoint both cars have passed. Preserve lap time during crash/manual respawn; pause stops the clock, and restart clears both cars, splits, history and normalizer inputs.
- Keep the optional reference ghost as an independent, quantized recording overlay. It never drives or replaces the live AI; the default overlay is off. Final-policy browser acceptance compares against the deterministic Python ghost on all three showcase tracks at the specified 2% lap-time tolerance.
- Render with WebGL multisample antialiasing disabled: the measured headless software renderer rose from about 40 FPS to 60 FPS. Keep all physics/observations unchanged; camera easing, speed zoom and render interpolation affect presentation only. Record the test environment and do not infer unmeasured laptop performance.
- Provide arrow/WASD driving, Space for the same brake/reverse axis as Python, R/Enter/Escape controls, and standard gamepad stick/triggers. Ignore input in form/link/help controls and clear held keys on blur. Small screens get a responsive view with an explicit keyboard/gamepad requirement, without adding touch-driving controls.

# M6 decisions

- Implement the explicitly authorized final milestone on `codex/m6-ghosts-heatmaps`. Preserve physics/track/observation v1 behavior and checkpoints; all changes are presentation, playback, packaging, tests or documentation.
- Use one seconds-based replay clock for all scheduled deterministic ghosts, sample each recording at its own exported dt, interpolate heading over the shortest arc, and hold terminal poses. Distinguish finished, crashed and time-limit outcomes; the 0-step recording can legitimately last 200 seconds. Add seeking alongside the specified play/pause, 0.5×–4× speed and restart controls.
- Color ghosts continuously from cool blue through green to warm orange, label actual training counts, and enlarge overview sprites only for readability. Deconflict labels with leader lines without altering replay positions. Pause playback on tab hiding; entering replay mode pauses the race and returning preserves it when track and opponent have not changed.
- Keep heatmap stage selection independent from the live opponent and replay clock. Load all heatmaps for a track once, verify them through the existing loader, and use a shared logarithmic peak across that track's stages (independent visitation/crash scales). Clear stale layers while loading and reject late responses after track/stage changes; a heatmap failure must not stop the race.
- Preserve exact 0.5 m cell coordinates while flipping world y into top-left texture coordinates. Precompose enabled visitation and crash grids into one cropped canvas texture, then draw raw crash points separately. Empty layers remain transparent. This avoids per-cell WebGL geometry and reduces software-renderer overdraw.
- Implement the inexpensive AI-vision stretch feature from the actual last inference input/state and mean controls: nine rays, steer/throttle bars, and critic output. Show the raw value numerically with a bounded tanh(value/10) meter; do not label it as lap time or win probability. Capturing these values never affects inference or simulation.
- Put presentation limits in `web/src/viewer/visual-config.ts`; use bounded drift particles and recycled batches of tire-mark geometry, stop their time on pause, honor reduced motion (including preference changes), and avoid redrawing unchanged paused/overview scenes. No new dependencies are needed.
- Keep Vite's relative base and locally bundled WASM/fonts; include `.nojekyll` and validate the exact production directory under `/racing-rl/` with a static file host and no SPA fallback. Generated model/data assets stay outside Git, so publish the prepared dist contents instead of pretending a source-only CI checkout can train/export them automatically.
- Make the README a complete entry point with architecture, actual result tables, training/export requirements, deployment instructions and explicitly requested GIF recording slots. Retain detailed M1–M5 instructions in `docs/DEVELOPMENT.md`; do not fabricate recordings or add broken GIF image links.

# Portfolio release decisions

- Create `main` at the M6 head; the milestone branches were already a linear stack, so no merge commit is needed.
- Commit the exported reference bundle (`web/public/assets/`, about 6 MB) instead of keeping it out of Git. A fresh clone can now play, and Pages can deploy from the repository, without a GPU training run. Checkpoints, evaluations and the copied ONNX runtime stay ignored; CI never trains or exports.
- Deploy through GitHub Actions (`pages.yml`) on pushes to `main`, and run `make test` plus a production build in `ci.yml`.
- Generate the README learning curve as a dependency-free SVG (`tools/plot_learning.py`) that adapts to light/dark mode, rather than adding matplotlib. It plots smoothed rollout return and the existing M4 snapshot completion rates; no new evaluation is run.
- License the project as MIT.
- Rewrite the README for first-time visitors (pitch, demo link, results, architecture) and move deployment and GIF-recording instructions to `docs/DEVELOPMENT.md`. Use the existing screenshot until real GIFs are recorded.

# M7 decisions

- Evaluate all 36 retained seed-2 resumable checkpoints without retraining; supply their numeric step as recorder identity because resumable files do not carry `requested_step`.
- Keep the required 3% gap on every shared track. Only Gold (step 2,785,280) qualifies; no Silver or Bronze qualifies behind it. Ship Gold + Author on every showcase, the explicitly accepted fewer-medals fallback.
- Gold deterministic laps: sweep 25.230622069s, ribbon 25.624031205s, technical 20.918658349s.
- Author final deterministic laps: sweep 24.246155310s, ribbon 24.591261626s, technical 19.812376420s.
- Keep the approved Stadium palette and Barlow Condensed heavy italic lettering; use world-space 10 m grass bands, 3 m kerb stripes, a gate at every fourth checkpoint, and camera zoom from 12 down to 6 px/m. Formula sprites are baked at 24 px/m, including wings, tyres, cockpit, number and layered shadow.
- Add training-pool size, held-out count, parallel-car count and the factual zero-RL-framework count to exported site stats so every visible intro number comes from site data. Omit a stale baked-in test count and use plain CI links; keep joeycatic's GitHub footer and omit the unsupplied LinkedIn URL.
- The parity fixture's `steps[].state` is the unrestricted physics trajectory, while `steps[].session.state` includes crash/respawn behavior. Replay the former with the unchanged physics kernel from `initial_state`, matching the required exported Python poses exactly, rather than wrongly comparing them to a DriveSession reset trajectory.
- Refuse silent reuse of an unrelated dev server in Playwright; allow E2E_PORT for testing an isolated checkout. This run uses port 5187 because the user's original dev server occupies 5173.
- The pose-only parity demo is exactly equal in this browser. Display zero as an exact match, while the stats tile retains the broader measured Python/TypeScript maximum from the committed validation report. The e2e assertion accepts a measured zero as well as scientific notation.
- Cache the static Stadium terrain at 3 px/m and reuse each livery texture per renderer. The new full-screen viewport exceeded the software renderer's pixel budget. Adapt only the render resolution after sustained slow frames (55 FPS target, 0.4 minimum); preserve physics dt, inference cadence, camera coordinates and simulation ticks. Headless Chromium 156 with heatmaps and AI vision measured 60.1 FPS / 16.8 ms p95 after adaptation; this is not a new laptop hardware measurement.
- Keep the intro's static fallback independent of model/WASM success, pause live rendering off screen or in a hidden tab, and respond to reduced-motion preference changes. The gate-HUD e2e uses a test-only live policy driver to supply controls to the real race tick; keyboard, gamepad input and race controls remain separately tested.
- Consolidate the coupled two-page, scene, cards, HUD and intro changes into one frontend commit after verification; the shared moved main.ts and race.html were implemented while export refreshed its caches. All M7 acceptance checks pass against the final committed bundle. M8 and M9 remain documentation-only future milestones.

# Racing RL — MVP Specification (Master Prompt)

You are building the MVP of **racing-rl**: a 2D top-down racing game with Trackmania-like arcade physics, where a car learns to drive via a **from-scratch PPO implementation**. The browser frontend lets users **race against the trained AI**, watch **ghost replays** of the AI at different training stages, and inspect **heatmaps** of crash locations and driving lines.

Work through the milestones below **in order**. After each milestone: run all tests, fix failures, then stop and write a short summary (what was built, how to run it, open issues). Do not start the next milestone until the current one meets its acceptance criteria.

---

## 0. Working rules

- Keep it simple. "Mildly complex, not overcomplex" is a hard requirement for the physics and the architecture.
- Python: 3.11+, type hints everywhere, `ruff` for lint/format, `pytest` for tests. Manage the environment with `uv`.
- Web: Vite + TypeScript (strict mode), PixiJS for rendering, `onnxruntime-web` for inference, `vitest` for tests.
- No backend server. The web app is fully static and loads exported files (ONNX models, JSON replays, JSON heatmaps).
- All tunable values (physics params, reward weights, PPO hyperparameters) live in typed config files (dataclasses + YAML), never hardcoded in logic.
- Everything must be deterministic given a seed: fixed physics timestep, seeded RNGs, no wall-clock dependence in the simulation.
- Do not add dependencies without a clear reason. Do not leave placeholder or TODO implementations in delivered features.
- When a decision is ambiguous, choose the simplest option that satisfies this spec and record it in `DECISIONS.md` (one line per decision, with the reason).
- Provide a `Makefile` with targets: `setup`, `test`, `drive`, `train`, `eval`, `export`, `web-dev`, `web-build`.

## 1. Non-goals for the MVP

3D rendering, online multiplayer, accounts, any backend, a track editor, neuroevolution, multiple car types, surfaces/boost pads. Design so these can be added later, but do not build them.

## 2. Repository structure

```
racing-rl/
├── sim/            # physics, track generation, geometry, observations, Gymnasium env, vectorized env
├── train/          # ppo.py (single file, from scratch), configs, checkpoint + snapshot logic
├── eval/           # evaluation runs, replay recording, heatmap aggregation
├── export/         # ONNX export, packaging of web assets
├── tools/          # debug_drive.py (pygame keyboard driving), benchmark.py
├── tests/          # pytest: physics, tracks, env, PPO smoke test, parity fixture generation
├── web/            # Vite + TS + PixiJS app (incl. TS port of the simulation)
├── configs/        # physics.yaml, reward.yaml, ppo.yaml, tracks.yaml
├── DECISIONS.md
└── README.md
```

## 3. Simulation (Python, the source of truth)

### 3.1 Timing
- Physics timestep `dt = 1/60 s`. The agent acts every 2 physics steps (30 Hz, action repeat = 2).
- Semi-implicit Euler integration. Use float64.

### 3.2 Car model (simplified dynamic bicycle model)
State: position `(x, y)`, heading `psi`, body-frame velocity `(vx, vy)`, yaw rate `r`.

- Inputs: `steer ∈ [-1, 1]`, `throttle ∈ [-1, 1]` (negative = brake; reverse only when nearly stopped, max reverse speed small).
- Steering angle = `steer * max_steer`, reduced with speed: `max_steer_eff = max_steer / (1 + speed / steer_speed_falloff)`. Steering input is rate-limited (no instant full lock).
- Front/rear lateral tire forces linear in slip angle up to a friction limit (`mu * normal_load`), then clamped. This clamp is what produces controllable drifting. Rear grip slightly lower than front so oversteer is possible but recoverable.
- Longitudinal: engine force (decreasing with speed), brake force, quadratic aerodynamic drag, rolling resistance.
- Yaw dynamics from front/rear lateral forces and the moment of inertia.
- Target feel: top speed ≈ 60–80 m/s, 0–100 km/h in ≈ 3 s, wide sweepers taken flat-out, hairpins require braking, drifting is possible but slower than a clean line.
- Wall contact (car center leaves the track by more than half the car width):
  - **Training mode:** episode terminates with crash penalty.
  - **Play mode:** car respawns at the last checkpoint with zero velocity after 0.5 s (Trackmania-style). A manual "respawn" key does the same instantly.

### 3.3 Tracks
- Closed loops, procedurally generated from a seed: 10–16 control points in polar coordinates around a center (radius noise + angle jitter), smoothed with a closed Catmull-Rom spline, resampled every 1 m along the centerline.
- Track width 10–14 m (constant per track for the MVP).
- Reject and regenerate tracks that self-intersect, whose boundaries overlap, or whose curvature exceeds what the car can physically drive.
- Representation: centerline polyline with cumulative arc length, tangents, curvature, plus left/right boundary polylines. Checkpoints every 25 m. Start/finish line at arc length 0.
- Serializable to a versioned JSON format (`track.v1.json`) so the web app loads exactly the same geometry. Custom editor tracks with crossings use `track.v2`, adding verified crossing arcs, boundary exclusions and rendering-only overpass choices; ordinary tracks retain `track.v1`.
- Define 3 fixed **showcase tracks** (fixed seeds, chosen to be fun and varied) in `configs/tracks.yaml`. These are used for evaluation, ghosts, heatmaps and racing in the web app. Training uses a pool of generated tracks; reserve a separate held-out pool for generalization tests.

### 3.4 Progress tracking
- Project the car onto the centerline, searching only a window around the last known index (no global search per step).
- Progress = arc length along the centerline, unwrapped across laps. Count laps; detect wrong-way driving.

### 3.5 Observation (version it as `obs.v1`)
All values normalized to roughly [-1, 1]:
1. 9 distance rays to the track boundary, evenly spread across ±120° relative to heading, max length 60 m, normalized by max length.
2. Longitudinal and lateral speed (body frame), yaw rate.
3. Heading error vs track tangent as `sin` and `cos`.
4. Lateral offset from the centerline divided by half-width.
5. Signed curvature of the centerline at lookahead distances 10, 25, 50, 80 m.
6. Previous action (steer, throttle).

Total: 22 dims. Document the exact order in code and in `sim/OBSERVATION.md`.

### 3.6 Reward (weights in `configs/reward.yaml`)
- `+0.1 × Δprogress_in_meters` per agent step (negative when driving backwards).
- `−0.001` per agent step (time pressure).
- `−10` and termination on crash.
- `+5` on lap completion.
- Episode truncates after 2 laps or a step limit.

Keep each reward term logged separately in `info` so ablations are easy later.

### 3.7 Environments
- `RaceEnv`: standard single-car Gymnasium environment (used for evaluation and debugging).
- `VecRaceEnv`: **natively vectorized** NumPy implementation simulating N cars at once (each on its own track from the pool), with batched physics and batched ray casting. Do not wrap N single envs in Python loops; this is where training speed comes from.
- Ray casting: precompute boundary segments per track and use a uniform grid / spatial hash so each ray only tests nearby segments.
- Auto-reset finished cars, assigning a new random track from the pool on reset.
- `tools/benchmark.py` reports steps/second. Target: **≥ 10,000 agent steps/s** with 64 cars on a laptop CPU.

## 4. PPO (from scratch, `train/ppo.py`)

Single readable file in the spirit of CleanRL. Do not import any RL library in the training code.

- Actor and critic: separate MLPs, 2 hidden layers × 256, tanh activations, orthogonal initialization (actor output layer gain 0.01, critic output gain 1.0).
- Gaussian policy with state-independent learnable `log_std`; actions clipped to [-1, 1] before being sent to the env.
- GAE with `γ = 0.99`, `λ = 0.95`. Clipped surrogate objective (`clip = 0.2`), value loss with optional clipping, entropy bonus (coef default 0.0, configurable), global grad-norm clip 0.5.
- Defaults: 64 envs × 256 steps per rollout, 10 epochs, 4 minibatches, Adam lr 3e-4 with linear annealing.
- Running mean/std normalization for observations (saved with every checkpoint), reward scaling by running return std, advantage normalization per minibatch.
- Device auto-selection: CUDA → Apple MPS → CPU, overridable via config. Note in the README that CPU is usually fastest for this network size.
- Logging to TensorBoard: episodic return, laps completed, best lap time, crash rate, each reward term, policy/value loss, entropy, approx KL, clip fraction, explained variance, steps/s.
- Checkpoints every N updates. Additionally save **ghost snapshots** at fixed global step counts: 0, 50k, 200k, 1M, 3M, final (configurable).
- Resumable training from a checkpoint.
- `tests/`: a PPO smoke test that trains for a few updates and asserts finite losses; a correctness test of GAE against a hand-computed example.
- Optional sanity script (dev dependency only): train Stable-Baselines3 PPO on the same env with matched hyperparameters and compare learning curves. Our PPO should reach comparable performance.

### Training curriculum for the MVP
1. Single showcase track until it reliably completes laps.
2. Pool of 200 generated tracks; evaluate on the 3 showcase tracks and on 50 held-out tracks.

## 5. Evaluation, ghosts, heatmaps (`eval/`)

- For every ghost snapshot and each showcase track:
  - Record **1 deterministic run** (policy mean action) as the ghost replay.
  - Run **K = 50 stochastic runs** for the heatmaps.
- **Replay format** (`replay.v1.json`): metadata (track id, snapshot step, lap time, finished/crashed) + per physics tick `x, y, psi, speed, steer, throttle`, quantized (e.g., cm / mrad) and delta-encoded to keep files small.
- **Heatmaps** (`heatmap.v1.json`) per snapshot and showcase track:
  - **Driving line:** visitation counts on a grid (0.5 m cells) covering the track bounding box.
  - **Crash locations:** list of crash positions plus a binned grid.
- `make eval` prints a summary table: lap completion rate, mean/best lap time, crash rate, per snapshot, for showcase and held-out tracks.

## 6. Export (`export/`)

- Export the policy to ONNX **including observation normalization inside the graph**, so the browser feeds raw observations. Outputs: deterministic action (mean) and the critic's value estimate.
- Verify with `onnxruntime` in Python that ONNX outputs match PyTorch outputs within 1e-5.
- `make export` assembles `web/public/assets/`: tracks, ONNX models per snapshot, replays, heatmaps, and a `manifest.json` listing everything with versions (`physics`, `obs`, `track`, `replay`, `heatmap`).

## 7. Physics parity (Python ↔ TypeScript)

Editor-builder fixtures and a crossing trajectory also compare Python and TypeScript.
Crossing ray exclusions preserve obs.v1; the original three parity fixtures regenerate
byte-identically.

- The web app contains a TypeScript port of: physics, track geometry/progress, observation builder. Same algorithms, same operation order, float64 (JS numbers).
- Python generates golden fixtures (`tests/fixtures/parity/*.json`): seed, track, a scripted action sequence of 600 physics steps, and the expected state + observation at every step.
- `vitest` replays the fixtures in TS and asserts max absolute error ≤ 1e-6 for state and observations.
- Both test suites run in `make test`. Parity failures block the milestone.

## 8. Web app (`web/`)

Static Vite + TS + PixiJS app, 60 FPS fixed-timestep loop with an accumulator and render interpolation. Load all data from `manifest.json`.

### 8.1 Race mode (core)
- Choose a showcase track and a medal opponent from track/medal cards: Bronze, Silver, Gold, Author (final policy). Unavailable medals are omitted; cards show deterministic lap times.
- Player drives with keyboard (arrows/WASD, space = brake/handbrake-light, R = respawn, Enter = restart). Gamepad support if trivial via the Gamepad API.
- **Live AI opponent:** runs the TS physics with the ONNX policy in the browser (`onnxruntime-web`, inference every 2 physics steps, same as training). It is a real car, not a replay; cars do not collide with each other (Trackmania-style).
- Optional ghost overlay of the AI's recorded best run.
- 3-2-1 countdown, lap timer, split times at checkpoints (green/red delta vs the AI), finish screen with result and "race again".
- Follow camera with smooth zoom by speed, plus a minimap.

### 8.2 Progressive Watch it learn (M8 delivered)
- `watch.html` plays improving deterministic attempts in training order, one car at a time,
  with faded cool-to-warm trails and data-derived captions.
- Play/pause, replay, skip to final, training-progress jumps and the showcase track picker
  control a roughly one-minute sequence. See §9 M8 for the complete requirements.

### 8.3 Heatmap overlay
- Toggleable layers on any track view: **driving line** (visitation intensity) and **crash locations** (red points / binned density).
- A slider over snapshots to watch both heatmaps evolve with training.

### 8.4 "What the AI sees" overlay (stretch, only if cheap)
Draw the 9 rays on the AI car, bars for current steer/throttle, and a meter for the value estimate.

### 8.5 Visual style
M7 uses the daylight Stadium identity: striped grass, grey tarmac, red/white kerbs,
blue checkpoint gates, and a checkered start. Code-drawn Formula cars have exposed tyres,
front/rear wings, a shaded body, cockpit and nose number. Player mint (#3fe0b0),
AI orange (#ff7a2f). A closer speed-eased camera and heavy italic white timing boxes
frame the race; gate split deltas are blue ahead and red behind.

The front page presents the live final policy, exported results, handwritten PPO,
interactive learning curves, live Python/TypeScript parity and static deployment.
The race lives at race.html; progressive Watch it learn lives at watch.html.
The legacy race.html#watch URL redirects to watch.html.

## 9. Milestones and acceptance criteria

**M1 — Simulation + manual driving**
Physics, track generation, progress/laps, JSON export of tracks. `make drive` opens a pygame window to drive a generated track with the keyboard (debug-quality rendering is fine).
✅ Car feel matches 3.2, drifting possible, lap timing correct, tests for track validity and progress pass.

**M2 — RL environment**
Observation, reward, `RaceEnv`, `VecRaceEnv`, benchmark.
✅ Gymnasium env checker passes, observation documented, benchmark ≥ 10k agent steps/s with 64 cars (report actual numbers).

**M3 — Own PPO**
`train/ppo.py`, configs, logging, checkpoints, ghost snapshots, tests.
✅ On a single showcase track the agent completes laps reliably. With the 200-track pool: ≥ 95 % lap completion on showcase tracks and ≥ 80 % on held-out tracks in `make eval`. Report the actual numbers and training time.

**M4 — Evaluation + export + parity**
Replays, heatmaps, ONNX export, manifest, TS physics port, parity fixtures and tests.
✅ ONNX matches PyTorch, parity tests pass, assets build with `make export`.

**M5 — Race against the AI**
Web race mode (8.1) with live ONNX opponent.
✅ The in-browser AI completes laps with lap times within 2 % of its Python evaluation on the same track (proves parity end to end). Stable 60 FPS on a mid-range laptop.

**M6 — Ghosts, heatmaps, polish**
Time-lapse mode, heatmap overlay with snapshot slider, visual style, README (what it is, how to run, architecture diagram, results table, GIF placeholders with instructions on how to record them).
✅ `make web-build` produces a static site that can be deployed to GitHub Pages as-is.

### M7: Stadium identity + intro page

Amends spec §8.5 (visual style) and the §8.1 opponent picker.

**Visual identity (owner's choices: Trackmania Stadium look, Formula car)**
- Environment: daylight; striped grass; grey tarmac; red/white kerbs on both edges; blue
  checkpoint gates across the road at each checkpoint; checkered start/finish.
- Cars: open-wheel Formula cars, top-down, drawn in code with body shading, soft drop shadow,
  front/rear wings, visible tyres, nose number. Player = mint (`#3fe0b0` family), AI = orange
  (`#ff7a2f` family). Reference: `cars.html` option B in the brainstorm mockups.
- HUD in Trackmania's language: heavy italic type in black boxes; checkpoint split shown at
  top-centre with the delta in **blue when ahead, red when behind**; race time bottom-centre;
  large speed readout bottom-right.
- Camera sits closer than today so the car detail reads in play; zoom still eases with speed.
  *(Builder's call within that: exact zoom levels.)*

**Race setup (owner approved)**
- Opponents are the training stages framed as Trackmania medals: **Bronze, Silver, Gold, Author**
  (Author = final policy).
- Today only two shipped snapshots (3M, final) can finish a lap, so the medals come from the
  **existing training checkpoints** (owner's decision): evaluate the deterministic mean policy of
  the 36 checkpoints in `artifacts/training/m3-seed2/checkpoints/` on each showcase track, pick
  three that finish **all three** showcase tracks with clearly separated lap times
  (Bronze slowest), and export ONNX for them through the existing export/validation path. No
  retraining. The same three checkpoints are used on every track.
- Fallback (owner's decision): if fewer than three checkpoints finish a given track, that track
  shows fewer medals. Never show an opponent that cannot finish.
- Medal cards show the medal's real lap time on that track, from the evaluation output.
  Chosen checkpoint steps and the selection evidence go in DECISIONS.md.
- Pre-race screen with track cards and medal-opponent cards replaces the sidebar of dropdowns.
  The race itself is full-screen.

**Intro page (owner approved the structure in `intro-page.html`)**
- It is the site's front page. Race, Watch it learn and the editor are separate screens reached
  from it.
- Hero: headline "A car that taught itself to drive.", one sentence on what it is, three
  buttons (Race the AI / Watch it learn / Draw a track). Background is the **real final policy
  lapping a Stadium track live** (ONNX in the browser), not a video.
- Stats strip: 98.7% showcase completion · 85.2% on 50 unseen tracks · 0 RL libraries ·
  5e-13 physics parity error. Values read from committed result files, not hand-typed.
- Engineer sections, in order:
  1. **PPO from scratch**: loop diagram (64 cars → rollout → GAE → clipped update), interactive
     learning curve (hover tooltips), the two-seed comparison including seed 1's regression,
     links into `train/ppo.py`.
  2. **Physics parity**: a Python-recorded trajectory and the live TypeScript simulation driven
     by the same inputs, drawn overlapping; error figures; one line on why it matters ("without
     this, the AI you race would drive worse than the one I trained").
  3. **Built to ship**: CI status, test counts, versioned/hashed assets, static deploy.
- Footer: owner's name, GitHub, LinkedIn. *(Owner to supply the name/LinkedIn URL; placeholder
  until then.)*
- Phones: full intro; buttons route to screens that work without a keyboard; Race shows the
  existing keyboard/gamepad notice.

**Acceptance**: every screen uses the Stadium look; no `<select>`-sidebar remains; intro renders
correctly at phone width; race behaviour, lap times and parity results unchanged; `make test`
and the e2e/static suites green.

### M8: Progressive "Watch it learn" — delivered

Replaces spec §8.2 (ghost time-lapse). §8.3 heatmaps stay, as an optional layer.

**Experience (owner's choices: one car at a time, improving each attempt, ~1 minute total)**
- About 10 attempts on one showcase track, played in training order, one car at a time.
- Each completed attempt leaves a faded trail, coloured cool (early) → warm (final), so the
  racing line visibly forms.
- Pacing to fit ~60 s: crashes play at 1× (short anyway); stalls are cut after ~3 s and labelled
  "stalled"; middle attempts fast-forward at 3×; the final clean lap gets the most time at 2× with
  a live lap timer. *(Builder's call: exact per-attempt speeds, as long as total ≈ 60 s.)*
- Bottom bar is **training progress**, one segment per attempt; play/pause, replay, skip to
  final, click any segment to jump.
- Captions are generated from data: attempt number, training steps, outcome ("crashed at
  312 m", "stalled", "lap 24.25 s"). A few milestone attempts get a short headline ("Found the
  throttle", "First full lap"), written only after inspecting what that checkpoint actually does.
- Crash heatmap for the current attempt's stage: optional toggle, off by default.
- Track picker for the three showcase tracks.

**Data**
- Attempts are deterministic mean-policy runs (as the current ghosts are) of selected
  checkpoints. Available: snapshots 0 / 50K / 200K / 1M / 3M / final plus 36 checkpoints every
  163,840 steps in `artifacts/training/m3-seed2/checkpoints/`.
- Selection: ~10 checkpoints, denser early in training, each showing a visible behaviour change
  on that track; selection may differ per track. *(Builder's call: the selection method; record
  the chosen steps in DECISIONS.md.)*
- Export adds replays for the selected checkpoints only. No ONNX models are needed for them,
  so the bundle grows only by replay size. Asset manifest/versioning extended accordingly.

**Acceptance**: a full playthrough on each showcase track takes 50–70 s; one moving car at a
time; captions match the replay outcomes; the six-ghost simultaneous view is gone.

### M9: Track editor

**Status: delivered; local acceptance verified.**
Drawn-track AI on drawings with crossings uses the final model with positive throttle
capped at 20%; crossing-free drawings retain original controls. Steering,
negative braking and value are unchanged. The owner approved this policy adapter
after the raw model crashed near 313 m on the figure-eight away from its crossing;
a capped native trial finished in 35.14 s. Applying the cap to all drawings regressed
a previously successful clockwise circle, so its scope is limited to overpass drawings.
Showcase, intro and Watch actions remain raw. The original clockwise-loop and
figure-eight lap assertions and complete local verification pass with this scoped pace.

New screen; adds a `track.v2` capability (crossings) to the simulation.

**Experience (owner's choices: draw freehand, crossings become over/underpasses, Clear to
start over, Submit when done, then the AI drives it, then you race it)**
- One continuous freehand stroke with mouse or finger. The loop closes automatically when the
  stroke ends near its start. The start line is the stroke's first point; driving direction is the
  drawing direction.
- The stroke is smoothed into a spline, resampled, and scaled to a track length in the range of
  the training tracks. Road width is fixed inside the training range (10–14 m).
- Crossings become overpasses: the later pass over a crossing is the bridge by default; clicking
  a crossing flips it. Bridge drawn on top with a shadow. Crossings shallower than a minimum angle
  are rejected (the roads would overlap too long). *(Builder's call: the exact minimum angle and
  minimum spacing between crossings / from the start line.)*
- Live validation while drawing:
  - **Blocks Submit**: geometry that cannot exist (road folds inside a corner, boundaries
    overlapping other than at a valid crossing, too short).
  - **Warns but allows**: corners sharper than the training limit (`max_curvature` 0.10) are
    painted red with a "too tight — the AI never trained on corners this sharp" note.
- **Clear** wipes the drawing.
- **Submit** → the final model, with a 20% positive-throttle cap only when the drawing has
  crossings, drives one lap with a follow camera and lap timer. The same rule applies when racing a drawn track; valid
  geometry does not guarantee completion. Result: lap
  time, or "crashed at N m" with the reason when known (e.g. the crash is inside a red section).
- **Race it** → the normal race screen on the drawn track, AI lap as the target, splits vs AI.
- **Share** (my addition, owner did not object): the track is encoded in the URL; opening the
  link loads it. No server.

**Simulation change: crossings**
- Crash detection already uses a local-window centreline projection, so it stays on the correct
  branch at a crossing.
- Raycasts must not see the other branch's boundaries inside a crossing zone (otherwise the car
  "sees" a wall across its road). Add per-crossing boundary-segment exclusion keyed on the car's
  current branch.
- Implemented in **both** Python (`sim/`) and TypeScript (`web/src/sim/`), with new golden
  fixtures on a crossing track. Tracks without crossings must produce identical observations; all
  existing fixtures must pass unchanged. This also means the editor's track builder is a
  TypeScript port of `build_track` / `validity_errors`, parity-tested against Python on the same
  control points.

**Acceptance**: a drawn simple loop can be driven by the AI and raced; a figure-eight with one
overpass is driven by the AI without phantom-wall crashes at the crossing; invalid drawings are
blocked with a reason; too-tight corners warn; shared URLs reproduce the same track;
Python/TS crossing parity fixtures pass; existing parity untouched.

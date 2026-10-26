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
- Serializable to a versioned JSON format (`track.v1.json`) so the web app loads exactly the same geometry.
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

- The web app contains a TypeScript port of: physics, track geometry/progress, observation builder. Same algorithms, same operation order, float64 (JS numbers).
- Python generates golden fixtures (`tests/fixtures/parity/*.json`): seed, track, a scripted action sequence of 600 physics steps, and the expected state + observation at every step.
- `vitest` replays the fixtures in TS and asserts max absolute error ≤ 1e-6 for state and observations.
- Both test suites run in `make test`. Parity failures block the milestone.

## 8. Web app (`web/`)

Static Vite + TS + PixiJS app, 60 FPS fixed-timestep loop with an accumulator and render interpolation. Load all data from `manifest.json`.

### 8.1 Race mode (core)
- Choose a showcase track and an opponent snapshot (e.g., "1M steps" or "final").
- Player drives with keyboard (arrows/WASD, space = brake/handbrake-light, R = respawn, Enter = restart). Gamepad support if trivial via the Gamepad API.
- **Live AI opponent:** runs the TS physics with the ONNX policy in the browser (`onnxruntime-web`, inference every 2 physics steps, same as training). It is a real car, not a replay; cars do not collide with each other (Trackmania-style).
- Optional ghost overlay of the AI's recorded best run.
- 3-2-1 countdown, lap timer, split times at checkpoints (green/red delta vs the AI), finish screen with result and "race again".
- Follow camera with smooth zoom by speed, plus a minimap.

### 8.2 Ghost time-lapse mode
- All ghost snapshots of one track start simultaneously as semi-transparent cars, colored on a gradient from early (cool) to final (warm), each labeled with its training step count.
- Playback controls: play/pause, speed (0.5×–4×), restart.

### 8.3 Heatmap overlay
- Toggleable layers on any track view: **driving line** (visitation intensity) and **crash locations** (red points / binned density).
- A slider over snapshots to watch both heatmaps evolve with training.

### 8.4 "What the AI sees" overlay (stretch, only if cheap)
Draw the 9 rays on the AI car, bars for current steer/throttle, and a meter for the value estimate.

### 8.5 Visual style
Dark asphalt track with light curbs, a clean neon accent palette (player and AI in clearly distinct colors), tire marks when lateral slip exceeds a threshold, light particles on drift. It should look intentional and readable, not like a debug view. No external art assets required.

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

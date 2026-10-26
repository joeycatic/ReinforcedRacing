# Racing RL

A deterministic 2D racing simulation with arcade bicycle physics and seeded tracks.
**M1 and M2 are implemented:** manual driving, seeded tracks, lap timing, track JSON,
and scalar/native batched RL environments with observations and rewards.
See [spec.md](spec.md) for the later PPO and browser milestones.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
make setup
make drive
```

`uv` manages the Python environment and dependencies. The project defaults to Python 3.12;
uv can download it if needed. A desktop display is required for interactive driving.

| Input | Action |
| --- | --- |
| W / Up | Accelerate |
| S / Down / Space | Brake; reverse when nearly stopped |
| A / Left, D / Right | Steer left/right |
| R | Instantly respawn at the last checkpoint |
| Enter | Restart the session and timer |
| Escape | Quit |

Steering ramps in rather than snapping to full lock. Brake for tight bends; use short
steering presses at speed and countersteer if the rear slides. Crashes freeze the car
for 0.5 simulation seconds, then respawn with zero velocity. Lap timing includes this delay.
The first timer starts when the session opens; press Enter for a fresh attempt.

```sh
make drive ARGS="--showcase ribbon"       # sweep (default), ribbon, technical
make drive ARGS="--seed 42"              # reproducible generated track
make drive ARGS="--export-track artifacts/my.track.v1.json"
make drive ARGS="--track artifacts/my.track.v1.json"
make tracks                              # export all three showcases
make tracks ARGS="--seed 42 --output artifacts/custom"
```

Exports are written to `artifacts/tracks/` by default and are intentionally ignored by Git.
No server or external art assets are needed.

## Verify

```sh
make test
make drive ARGS="--headless --steps 600 --screenshot artifacts/m1-drive.png"
```

`make test` checks Ruff lint/format and runs pytest, including track rejection and
serialization, progress in both directions, interpolated lap timing, crash/respawn,
determinism, tire saturation and recovery, complete driven laps on all showcases,
and a headless pygame smoke check. The headless CLI uses scripted controls and SDL's
dummy video driver; it is suitable for CI and does not need a display.

With the default physics, a straight-line simulation reaches **100 km/h in 2.98 s**
and settles at **74.38 m/s**. These are simulation measurements, not wall-clock benchmarks.
The tests enforce the specification's acceleration/top-speed ranges. M2 checks also run
Gymnasium's environment checker, compare scalar and batched physics over 600 ticks,
compare grid rays against full-boundary intersections, and cover episode limits,
reward terms, deterministic track assignment, autoreset, and terminal observations.

## RL environments (M2)

```python
import numpy as np
from sim.env import RaceEnv
from sim.vec_env import VecRaceEnv
from sim.pool import make_track_pool

# Quick defaults use the three showcase tracks.
env = RaceEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0]))
if terminated or truncated:
    observation, info = env.reset()
env.close()

# Build static geometry/grid once, then simulate 64 cars in NumPy batches.
pool = make_track_pool("training")  # 200 generated tracks; held_out has 50 separate seeds.
envs = VecRaceEnv(64, pool)
observations, info = envs.reset(seed=42)
observations, rewards, terminated, truncated, info = envs.step(np.zeros((64, 2)))
envs.close()
```

Actions are `[steer, throttle]`; shapes are `(2,)` and `(N, 2)`. Observations are
float64 with shapes `(22,)` and `(N, 22)`. The exact `obs.v1` layout, normalization,
signs, and terminal-observation contract are in [sim/OBSERVATION.md](sim/OBSERVATION.md).
The spec's listed components sum to 21; the 22nd value exposes the steering actuator.

Rewards sum four individually logged weighted terms from `configs/reward.yaml`:
`0.1 × delta_progress`, `−0.001` time pressure, `−10` crash, and `+5` per completed lap.
Crashes terminate; reaching two laps or 6000 agent steps truncates. Limits live in
`configs/env.yaml`. Info includes progress, lap count/timing, wrong-way status,
physics/agent step counts, episode return, and `reward_terms`; finished episodes also
include an `episode` summary and accumulated reward terms.

`RaceEnv` requires explicit reset after finishing. `VecRaceEnv` resets finished cars
immediately, drawing tracks uniformly from its pool. Returned observations for these
cars belong to the next episode. Use `info["final_obs"]` and `info["_final_obs"]` for
the previous episode's terminal observations; truncated transitions must bootstrap
from those terminal observations. Root info describes the transition just taken.
Always call `reset(seed=...)` before stepping. The same seed, ordered pool, batch size,
and actions reproduce track selection and transitions, including autoresets.

For fixed evaluation tracks, construct `TrackPool([track])`, or select the initial
track using `reset(options={"track_index": i})` on RaceEnv / `track_indices` on VecRaceEnv.
Environments are headless; use `make drive` for pygame keyboard driving.

## Throughput

```sh
make benchmark
make benchmark ARGS="--output artifacts/m2-benchmark.json"
```

The benchmark uses 64 cars, 64 generated training tracks, 100 warmup batches, and
three samples of 1000 batches. One agent step means **one car taking an action for
up to two physics ticks**, so a vector call contains 64 agent steps. Timing includes
physics, local projection, grid-based rays, observations, rewards, info, and same-step
autoreset. Pool generation/grid construction and action generation are excluded and
setup time is reported separately. Runtime controls are seeded and held for eight
agent steps, allowing cars to accelerate, crash, and exercise resets.

The recorded measurement on this host (Intel Core i5-11400F, Python 3.12.13, NumPy 2.5.4)
was **13,755 agent steps/s** (sample rates 13,819 / 13,755 / 13,709), above the 10,000
target. There were 711 / 714 / 729 autoresets; pool/grid setup took 11.11 s. Full results
are saved in [docs/m2-benchmark.json](docs/m2-benchmark.json). This is a desktop CPU
measurement; performance on a particular laptop still needs measuring.

## Code and data

| Location | Responsibility |
| --- | --- |
| `sim/config.py`, `configs/*.yaml` | Validated dataclasses and all simulation tuning |
| `sim/physics.py` | Tire forces, steering actuator, longitudinal forces, fixed-step integration |
| `sim/geometry.py`, `sim/track.py` | Seeded spline tracks, resampling, validation, JSON |
| `sim/progress.py` | Local projection, signed offset, unwrapped progress, checkpoints/laps |
| `sim/session.py` | Play/training crash behavior and deterministic session clock |
| `sim/pool.py`, `sim/raycast.py` | Shared static geometry and batched grid traversal |
| `sim/observation.py`, `sim/reward.py` | Versioned observations and weighted reward terms |
| `sim/env.py`, `sim/vec_env.py` | Scalar Gymnasium and native batched environments |
| `tools/debug_drive.py` | Keyboard controls and pygame rendering |
| `tools/export_tracks.py` | Track-only JSON export |
| `tools/benchmark.py` | Full-transition CPU throughput measurement |

Simulation uses SI units, positive-left steering and a y-up world. Physics advances at
1/60 s with semi-implicit Euler; controls update every two ticks. Rendering and wall
time never enter the simulation. The model uses front/rear linear slip forces clamped
at separate friction limits, speed-dependent engine/steering, drag, rolling resistance,
and low-speed reverse. `CarState.steering` holds the actuator's normalized input.
Current physics version: `physics.v1`.

Track files declare `version: "track.v1"` and contain `id`, `name`, `seed`, `width`,
`centerline`, `arc_length`, `tangents`, `curvature`, `left`, `right`, and `checkpoints`.
Closed polylines have N points with no duplicate final point; `arc_length` has N+1
entries including the closing segment. Checkpoints are arc lengths at 25 m intervals,
starting at zero. Sampling wraps across the finish. JSON loading checks the version,
finite geometry, derived arrays, and track validity, preserving exported numbers.
The future browser should load this geometry instead of regenerating it.

## Scope and open issues

M3–M6 are intentionally unimplemented: PPO, evaluation, ONNX, TypeScript parity,
and the browser game. M2 does not claim learning results.
`train`, `eval`, `export`, `web-dev`, and `web-build` exist as required Make targets
and exit with an explicit milestone message. M1 track export is `make tracks`.

Automated checks establish the numeric driving targets and recoverable sliding;
subjective keyboard feel still needs human playtesting. Rendering is deliberately
debug quality. See [DECISIONS.md](DECISIONS.md) for ambiguity resolutions, including
the literal half-car-width wall threshold and deferred browser parity work.

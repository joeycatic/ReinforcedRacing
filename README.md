# Racing RL

A deterministic 2D racing simulation with arcade bicycle physics and seeded tracks.
**M1 is implemented:** manual pygame driving, drifting, checkpoints, lap timing,
crash/respawn, and versioned track JSON. See [spec.md](spec.md) for the later RL and browser milestones.

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
The tests enforce the specification's acceleration/top-speed ranges.

## Code and data

| Location | Responsibility |
| --- | --- |
| `sim/config.py`, `configs/*.yaml` | Validated dataclasses and all simulation tuning |
| `sim/physics.py` | Tire forces, steering actuator, longitudinal forces, fixed-step integration |
| `sim/geometry.py`, `sim/track.py` | Seeded spline tracks, resampling, validation, JSON |
| `sim/progress.py` | Local projection, signed offset, unwrapped progress, checkpoints/laps |
| `sim/session.py` | Play/training crash behavior and deterministic session clock |
| `tools/debug_drive.py` | Keyboard controls and pygame rendering |
| `tools/export_tracks.py` | Track-only JSON export |

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

M2–M6 are intentionally unimplemented: observations/rewards, Gymnasium/vectorized
environments, PPO, evaluation, ONNX, TypeScript parity, and the browser game.
`train`, `eval`, `export`, `web-dev`, and `web-build` exist as required Make targets
and exit with an explicit milestone message. M1 track export is `make tracks`.

Automated checks establish the numeric driving targets and recoverable sliding;
subjective keyboard feel still needs human playtesting. Rendering is deliberately
debug quality. See [DECISIONS.md](DECISIONS.md) for ambiguity resolutions, including
the literal half-car-width wall threshold and deferred browser parity work.

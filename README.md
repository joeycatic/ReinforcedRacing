# Racing RL

A deterministic 2D racing simulation with arcade bicycle physics and seeded tracks.
**M1–M3 are implemented:** manual driving, seeded tracks, lap timing, track JSON,
scalar/native batched environments, and PPO from scratch with checkpoints and evaluation.
See [spec.md](spec.md) for the later export and browser milestones.

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

## Train PPO (M3)

```sh
make setup
make train
make train ARGS="--run-dir artifacts/training/my-run --device cpu"
make train ARGS="--run-dir artifacts/training/my-run --resume artifacts/training/my-run/latest.pt"
uv run tensorboard --logdir artifacts/training
```

`configs/ppo.yaml` contains all algorithm and curriculum settings. The default uses
64 cars × 256 steps, ten epochs/four minibatches, separate two-layer 256-unit tanh
actor/critic networks, Gaussian actions, a 6M-step budget, and Adam at 3e-4 with linear annealing.
Device auto-selection is CUDA → Apple MPS → CPU. Try `--device cpu`: CPU is usually
fastest for networks this small, though hardware and minibatch size affect results.

Training starts on the `sweep` showcase. Every ten updates an independent eight-run
stochastic evaluation checks the first lap. After at least 200k agent steps and ≥95%
success, training switches to the full 200-track pool. This transition preserves
the policy, optimizer, running statistics, and global step. A run that never passes
this gate remains on the single track; reaching its step budget does not mean M3's
learning criteria have been met.

`train/ppo.py` implements the clipped surrogate, optional clipped value loss, GAE,
minibatch advantage normalization, entropy bonus, and gradient norm clipping without
an RL framework. Actions are sampled from a Gaussian with learned state-independent
log standard deviation; likelihoods use the original samples and only the actions
sent to the environment are clipped. GAE bootstraps truncations from `final_obs`,
cuts recursion at every episode boundary, and never bootstraps a crash.

Running mean/variance normalizes observations. Rewards are divided by the standard
deviation of discounted returns, without subtracting a reward mean. Evaluation freezes
statistics and reports original, unscaled episode rewards. TensorBoard and JSONL
logs include episode returns/laps, best lap time, crash rate, reward terms, policy/value
losses, entropy/log_std, KL, clip fraction, explained variance, throughput, observation
ranges, and action clipping. The simulation remains float64; the networks use float32.

Checkpoints are atomic writes under the run directory. `latest.pt` and numbered
`checkpoints/` files contain model/Adam state, observation/return statistics, RNG states,
live vector-environment state, versions, and configuration. Resume rejects changed
geometry or simulation settings. Exact uninterrupted-vs-resumed training is tested
on CPU and on CUDA when available. To increase a run's budget, add `--total-timesteps`;
that extends the linear learning-rate horizon and is a deliberate continuation change.
`--updates 3` stops after three additional updates and writes a resumable checkpoint.

Scheduled policy snapshots are stored in `snapshots/` for steps 0, 50k, 200k, 1M, 3M,
and `final`. They include both normalizers and are intentionally separate from resumable
checkpoints. A threshold is captured at the first vector batch that reaches it (50k is
captured at 50,048 with 64 cars); metadata records both requested and actual step counts.
The policy stays fixed during a rollout, while its observation statistics continue to
update. Full training stops at a rollout boundary and can slightly exceed the budget.

## Evaluate policies

```sh
make eval
make eval ARGS="--checkpoint artifacts/training/my-run/latest.pt --runs 5"
make eval ARGS="--checkpoint artifacts/training/my-run/latest.pt --deterministic --runs 1"
make eval ARGS="--checkpoint artifacts/training/my-run/latest.pt --snapshots --runs 5"
```

The M3 evaluator prints lap completion, crash rate, and mean/best lap time per snapshot
for the three showcases and 50 held-out tracks, with per-track/per-episode JSON results
in `artifacts/evaluation/`. `--runs` is the number of trials **per track**. Default trials
sample the Gaussian policy; `--deterministic` uses its mean, as the eventual browser
opponent will. Each trial starts from rest at the start line and stops at its first
completed lap, crash, or the configured step limit. It never trains on held-out tracks
or updates running statistics. A timeout counts as an unsuccessful lap; lap times only
summarize successful trials. Replay/heatmap production and ONNX export remain M4 work.

The measured reference runs started with the saved 3M-step baseline configurations
and resumed to 6M after diagnosing insufficient training on the larger pool. To
reproduce that exact schedule (rather than a fresh run with a 6M annealing horizon):

```sh
make train ARGS="--config experiments/m3-baseline-seed1.yaml --run-dir artifacts/training/reproduce-seed1"
make train ARGS="--run-dir artifacts/training/reproduce-seed1 --resume artifacts/training/reproduce-seed1/latest.pt --total-timesteps 6000000"
```

Use `m3-baseline-seed2.yaml` and a separate directory for the second seed. See
[experiments/m3-diagnosis.json](experiments/m3-diagnosis.json) for the signal checks
and the one-setting budget change. The default budget was raised to 6M after this
comparison; learning results remain something to measure, not assume from finite losses.

The validated reference is `artifacts/training/m3-seed2/latest.pt` (6,012,928 steps):
**98.7% showcase completion** over 150 stochastic trials and **85.2% held-out completion**
over 250 trials, exceeding M3's 95% / 80% targets. Mean-action evaluation completes all
three showcases and 43/50 held-out tracks. Measured rollout/update time was **9.81 min**
on an RTX 3060 Ti; startup, pool generation and evaluation are excluded. Seed 1 also
passes the stochastic targets but has a deterministic technical-track regression,
so seed 2 is the reference for future browser work.

See [experiments/M3_RESULTS.md](experiments/M3_RESULTS.md) for both seeds, per-track
lap times, failure counts, experiment conditions, hashes, and raw results. To inspect
the retained reference directly:

```sh
make eval ARGS="--checkpoint artifacts/training/m3-seed2/latest.pt --showcase-only --runs 50"
make eval ARGS="--checkpoint artifacts/training/m3-seed2/latest.pt --runs 5"
```

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
| `train/config.py`, `configs/ppo.yaml` | Typed PPO, normalization, curriculum and snapshot settings |
| `train/ppo.py` | Single-file PPO, logging, checkpoint/resume and curriculum loop |
| `eval/evaluate.py` | Frozen-policy lap-completion and timing evaluation |

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

M4–M6 are intentionally unimplemented: replay/heatmap assets, ONNX, TypeScript parity,
and the browser game. `export`, `web-dev`, and `web-build` exit with an explicit milestone
message. M1 track export is `make tracks`.

Automated checks establish the numeric driving targets and recoverable sliding;
subjective keyboard feel still needs human playtesting. Rendering is deliberately
debug quality. See [DECISIONS.md](DECISIONS.md) for ambiguity resolutions, including
the literal half-car-width wall threshold and deferred browser parity work.

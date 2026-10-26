# Static assets and parity (M4)

`make eval` evaluates every scheduled snapshot in `configs/export.yaml`'s `run_dir`.
It records one deterministic policy-mean ghost per showcase, plus 50 stochastic
one-lap trials per showcase for heatmaps. Held-out evaluation uses five trials on
all 50 reserved tracks. A trial ends on its first lap, crash, or 6000-agent-step limit.
Timeouts count as unsuccessful laps. Mean/best times are conditional on completion.
The summary table reports the stochastic trials; the JSON also records deterministic
results. Statistics stay frozen and evaluation uses an independent seeded noise RNG.

`make export` runs this evaluation (reusing verified caches), exports each ONNX graph,
and stages a complete `web/public/assets/` bundle. It replaces the prior bundle only
after every model and source asset passes verification. Cache identities cover source
code, checkpoint hashes, configuration, and both pools' exact geometry. Corrupt cached
files trigger regeneration. Missing scheduled checkpoints are errors, never placeholders.

Generated checkpoints, evaluations, and assets are ignored by Git. Keep the training
run to regenerate exports. The current reference is `artifacts/training/m3-seed2`;
use `make export ARGS="--run-dir artifacts/training/my-run"` for another complete run.
Custom output paths, trial counts, quantization, and other export settings belong in a
YAML file passed through `--config`. The summary-only M3 evaluator remains available
with `make eval ARGS="--checkpoint path/to/latest.pt --runs 5"`.

## Manifest

`manifest.json` declares `manifest.v1`, plus physics/obs/track/replay/heatmap versions.
It lists the simulation configuration, three tracks, and all scheduled snapshots.
Each snapshot names its actual step count, source checkpoint SHA-256, ONNX model,
per-track replay and heatmap descriptors, and numeric ONNX validation results.
Every descriptor has a relative path, byte count and SHA-256. No asset depends on
files outside this directory. Serve it over HTTP; browser integrity checks require
localhost or HTTPS.

`web/src/assets/load.ts` checks all five versions before fetching any dependent file.
It verifies size/hash before parsing JSON or returning model bytes, rejects unsafe
paths and duplicate identities, and checks per-file versions and track/snapshot
identities. M5 will use this loader from the race UI; M4 supplies no game page yet.

## Models

ONNX input `observations` is **float64 [batch,22]**, the raw `obs.v1` vector produced
by the simulation (already clipped by its observation builder). Frozen running-mean
normalization, variance/epsilon scaling and observation clipping are inside the graph.
The MLP runs in float32, matching training. Outputs are float32 `action [batch,2]`
(the unsampled Gaussian mean, before control clipping) and `value [batch]`. Clamp
steer/throttle to [-1,1] and hold actions for two physics ticks, as in training.
No stochastic noise or reward normalization is required for browser inference.

Export compares the original PyTorch policy/critic using its independent NumPy
normalizer against ONNX Runtime CPU on batches of 1, 17, and 256 observations
(including zero and both observation extremes). Absolute error must be ≤1e-5 for
both outputs; non-finite values fail export. The manifest retains each result.
The explicit TorchScript ONNX exporter is used for this feed-forward graph; PyTorch
currently emits a deprecation notice for it. Browser inference integration is M5.

## Replays

`replay.v1` stores `dt`, columns `[x,y,psi,speed,steer,throttle]`, six scales,
metadata, and integer `deltas`. Tick zero is the exact start pose quantized to the
specified units, with zero speed/controls. The first delta row is absolute;
subsequent rows are differences of **independently quantized absolute samples**.
Cumulatively sum integers, then divide each column by its scale. This prevents
rounding drift. Defaults are 100 units/m for position/speed, 1000/rad for heading,
and 10000/unit for controls. Heading wraps at ±π; decode integers before applying
any shortest-angle rendering interpolation. Every physics tick is recorded through
the terminal tick; frame count is `metadata.physics_ticks + 1`.

Metadata includes track id/seed, requested snapshot identity and actual global step,
checkpoint hash, lap time (null on failure), finished/crashed/timeout status, return,
progress, simulation ticks, and evaluation seed. A crashed ghost ends at its crash.

## Heatmaps

`heatmap.v1` has a common 0.5 m grid for each track across all snapshots. The grid
covers both boundaries' bounding box plus a configured 5 m margin for crash overshoot.
`origin` is the lower-left world coordinate; cell `(x,y)` is at flat index
`y * width + x`, using floor((position-origin)/cell_size). Counts use
`sparse.flat.v1`: sorted `[index,count]` pairs, with omitted cells equal to zero.
`visits` counts live-car positions after each physics tick, including the terminal
tick. `crash_bins` counts terminal crash ticks, and `crash_positions` retains their
unquantized coordinates. `visit_count` and `crash_count` are conservation checks.
Finished rows add no subsequent visits or crashes. Start poses are not counted.

## Simulation parity

Python is authoritative. TypeScript uses JS numbers (float64), the same force and
integration order, local projection, sequential checkpoints, lap interpolation,
crash/respawn clock, spatial-grid DDA rays, and the complete 22-field observation
builder. It loads the exact exported track geometry; it does not rerun NumPy's
seeded generator. Track generation and rendering remain separate concerns.

`make parity` regenerates the three committed showcase fixtures. Each contains a
seed, configuration, geometry, 600 scripted controls, expected states/observations,
and full play-session results, including crashes and manual respawn/restart. Tests
compare integrated play sessions using TS-generated state/progress/observations at
every tick. Additional unrestricted free-body trajectories stress sliding and reverse:
state and observation kernels are compared independently, while projection uses the
identical Python coordinates. This isolates discrete nearest-segment ties far off the
road (observed at 44 m distance), where a tiny libm position difference can choose
an adjacent segment. No simulation rule or tolerance was changed to hide that tie.
Separate 600-position probes cover multiple laps, wrong-way motion, and checkpoints;
ray probes cover collinearity, axis-parallel rays and origins outside the grid.

`make test` checks Python fixture freshness, pytest, TypeScript types and Vitest.
State/observation error must be ≤1e-6 and failures report the first tick/field.
If an exported bundle is present, Vitest also loads every production asset through
the browser loader with integrity and identity checks. This integration test is
skipped on clean checkouts until `make export`; the synthetic codec/loader tests
always run. Browser rendering and live AI integration are the next milestone.

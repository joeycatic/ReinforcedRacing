# M3 training and evaluation results

Date: 2026-10-26. Reference checkpoint: `artifacts/training/m3-seed2/latest.pt`; policy snapshot: `artifacts/training/m3-seed2/snapshots/final.pt`. These local artifacts are retained outside Git. Their hashes, configs, versions and individual trial results are recorded in [m3-results.json](m3-results.json) and [evaluation/](evaluation/).

Both seeds use the specified 64 × 256 rollout, separate 256 × 256 tanh networks, ten epochs/four minibatches, γ=.99, λ=.95, clip=.2, Adam 3e-4 with linear annealing, entropy coefficient 0, and gradient cap .5. They first train on sweep, then the same 200 generated training tracks. All 50 held-out seeds remain excluded from training.

Hardware/software: Intel Core i5-11400F, NVIDIA RTX 3060 Ti, PyTorch 2.14.1+cu130, Python 3.12.13, NumPy 2.5.4. Network calculations use float32; simulation and running statistics use float64.

## Budget comparison (stochastic policy)

Every row uses evaluation seed 42 and five trials per track: 15 showcase and 250 held-out trials. Each trial starts at rest, uses frozen normalizers, and ends at its first lap, crash, or 6000-agent-step timeout. Means/bests are conditional on completing a lap; timeouts count as failures but are separate from crashes.

| Training seed | Budget | Showcase laps | Held-out laps | Rollout/update time |
| --- | --- | --- | --- | --- |
| 1 | 3M | 86.7% | 76.4% | 4.57 min |
| 1 | 6M | 100.0% | 93.2% | 9.60 min |
| 2 | 3M | 93.3% | 82.0% | 4.98 min |
| 2 | 6M | 100.0% | 85.2% | 9.81 min |

Time is cumulative measured wall time inside rollout collection and PPO updates. It excludes geometry/grid construction, curriculum evaluation, checkpoint writes and the pause between the 3M baseline and its continuation. The two seeded runs overlapped on the same host; this is a learning-time report, not a hardware throughput comparison.

Both single-track gates passed at step 1,310,720 with 8/8 stochastic laps. Final runs stop at step 6,012,928 (367 full rollouts), not exactly 6M. Baselines stop at 3,014,656. The only treatment change is extending total_timesteps from 3M to 6M and therefore extending the linear annealing horizon; the continuation keeps Adam, normalization, live environment and RNG state. The saved baseline YAMLs reproduce the original 3M stage. A fresh 6M run has a different learning-rate history and is not claimed to reproduce this exact table.

## Reference validation (seed 2)

Fifty stochastic trials on each showcase provide 150 trials total. Held-out validation uses five stochastic trials on each of 50 tracks (250 trials). Both requested M3 thresholds are met.

| Track/pool | Trials | Lap completion | Crash rate | Mean lap | Best lap |
| --- | --- | --- | --- | --- | --- |
| sweep | 50 | 100.0% | 0.0% | 24.59 s | 24.37 s |
| ribbon | 50 | 96.0% | 4.0% | 24.97 s | 24.74 s |
| technical | 50 | 100.0% | 0.0% | 20.15 s | 19.86 s |
| held-out | 250 | 85.2% | 12.8% | 22.77 s | 19.57 s |

The reference achieves **148/150 showcase laps (98.7%)** and **213/250 held-out laps (85.2%)**. Held-out failures include 32 crashes and five timeouts; failures are retained in the raw reports. Seed 1 separately reaches 147/150 showcase laps (98.0%) and 233/250 held-out laps (93.2%).

## Deterministic mean-action check

The browser opponent will use policy means, so this is measured separately from stochastic validation. Seed 2 completes all three showcases and 43/50 held-out tracks (86.0%). Repeating an identical deterministic run is not treated as an independent trial.

| Showcase | Mean-action lap time |
| --- | --- |
| sweep | 24.25 s |
| ribbon | 24.59 s |
| technical | 19.81 s |

Seed 1 has a late mean-action regression on technical (2/3 showcase finishes), even though its stochastic completion passes. The diagnostic checkpoint scan is preserved in [m3-seed1-mean-diagnostic.json](m3-seed1-mean-diagnostic.json). This failure is why seed 2, which passes both modes, is the reference for M4; the failed seed-1 deterministic run is not omitted from the evidence.

## Verification and reproduction

`make test`: 80 tests pass on this host, including hand-computed GAE, rollout-level terminal-observation bootstrapping, finite PPO updates, normalizer correctness, scheduled snapshots, TensorBoard tags, and bit-exact CPU/CUDA checkpoint resume through autoresets.

```sh
make train ARGS="--config experiments/m3-baseline-seed2.yaml --run-dir artifacts/training/reproduce-seed2"
make train ARGS="--run-dir artifacts/training/reproduce-seed2 --resume artifacts/training/reproduce-seed2/latest.pt --total-timesteps 6000000"
make eval ARGS="--checkpoint artifacts/training/m3-seed2/latest.pt --showcase-only --runs 50"
make eval ARGS="--checkpoint artifacts/training/m3-seed2/latest.pt --runs 5"
make eval ARGS="--checkpoint artifacts/training/m3-seed2/latest.pt --deterministic --runs 1"
```

Checkpoints include the model, optimizer, observation/return normalizers, RNG/environment state and version/config guards. Snapshot steps 0, 50k, 200k, 1M, 3M, final exist for both seeds. 50k is captured at 50,048; all other scheduled numerical thresholds align with a 64-car vector step.

The command-line snapshot evaluator was also checked with one stochastic trial per showcase: 0/3 completions at 0, 50k and 200k; 1/3 at 1M; 3/3 at 3M and final. These small diagnostic samples demonstrate the full snapshot path, not acceptance rates. See [the snapshot report](evaluation/seed2-snapshots-showcase.json).

M3 stops here. Replay/heatmap assets, ONNX export, TypeScript parity and browser validation remain M4+. Remaining model limitations are the observed held-out failures and seed sensitivity; the measured reference satisfies the specified acceptance rates without claiming universal safe driving.

**Diagnosis:** 3M underperformance → metrics, observation normalization/outliers, scripted rewards, action distributions, GAE/autoreset and easy-track learning checked → no signal bug found; too little pool training before annealing ended → extend only the budget to 6M and compare two seeds → both seeds pass stochastic M3 targets, with seed 2 also passing the mean-action showcase check.

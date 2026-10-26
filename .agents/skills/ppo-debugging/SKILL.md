---
name: ppo-debugging
description: Use when PPO training in train/ppo.py is not learning, learns then collapses, plateaus, or behaves strangely (circling, crashing at the start, standing still). Systematic diagnosis before touching hyperparameters.
---

# PPO debugging

When training goes wrong, do not start by tuning hyperparameters. Most failures are bugs or broken signals. Work through these steps in order and stop at the first one that finds a problem. Report what you checked and what you found.

## 1. Read the metrics first

Open the TensorBoard logs for the run and check:

| Metric | Healthy | Suspicious |
|---|---|---|
| approx KL | ~0.005–0.03 | > 0.05 (updates too large) or ~0 (nothing learned) |
| clip fraction | ~0.05–0.3 | > 0.4 or 0 |
| explained variance | rises toward 0.5–0.9 | stays ≤ 0 → critic is broken or rewards are noise |
| entropy / log_std | decreases slowly | collapses early (premature determinism) or never decreases |
| per-term rewards | progress term grows | crash term dominates, or time penalty outweighs progress |
| steps/s | stable | sudden drops → environment bug or memory leak |

## 2. Check the signals, not the algorithm

- **Observations:** log min/max/mean of each observation dimension over a rollout. Any NaN, inf, constant value, or values far outside [-5, 5] after normalization is a bug. Confirm the running normalizer is updated during training and frozen during evaluation.
- **Rewards:** for one hand-driven or scripted episode, print every reward term per step. A car driving forward cleanly must clearly earn more than one standing still or wiggling in place.
- **Actions:** histogram the actions sent to the env. Constant ±1 everywhere means saturation (log_std too high or clipping issue); all near 0 means the policy never explores.
- **Done/truncation handling:** truncation (step limit, lap limit) must bootstrap from the value estimate; only real terminations (crash) cut the return. Getting this wrong is one of the most common PPO bugs.
- **Auto-reset:** verify that after a reset the stored next observation belongs to the new episode and that GAE does not leak across episode boundaries.

## 3. Make the problem trivial

- Train on **one easy showcase track** only, with a fixed seed.
- If it still fails, shorten the problem: lower top speed, wider track, or a straight-ish track. A correct PPO must solve the trivial case quickly.
- Run the GAE unit test and the PPO smoke test.
- If available, run the Stable-Baselines3 sanity script on the same env and config. If SB3 learns and ours does not, the bug is in `train/ppo.py`. If neither learns, the bug is in the environment or reward.

## 4. Typical behaviors and their usual causes

- **Drives in circles:** reward pays for speed or distance instead of progress along the centerline, or progress unwrapping across the start line is broken.
- **Crashes immediately, never improves:** crash penalty too large relative to progress reward, or the episode ends before any positive reward is reachable.
- **Stands still:** time penalty or crash fear outweighs progress; check reward scale and that throttle actually moves the car.
- **Learns, then collapses:** learning rate too high, KL spikes, or observation normalizer statistics shifting heavily; check KL and clip fraction around the collapse.
- **Good on training tracks, bad on held-out tracks:** overfitting; increase track pool size before changing the network.

## 5. Only then tune hyperparameters

Change **one** setting at a time, run at least 2 seeds, and compare against the baseline run. Write the comparison into `experiments/` (config, seeds, result). Candidate knobs in order: learning rate, entropy coefficient, rollout length, number of epochs, network size.

## Report format

End with: symptom → what was checked → root cause found (or ruled out) → fix applied → evidence that the fix worked (metric or eval result).

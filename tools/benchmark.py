"""Measure complete agent transitions, including sensors, reward, info and autoreset."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from sim.pool import make_track_pool
from sim.vec_env import VecRaceEnv


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def benchmark(
    num_envs: int,
    track_count: int,
    steps: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    pool_started = perf_counter()
    pool = make_track_pool("training", count=track_count)
    setup_seconds = perf_counter() - pool_started
    env = VecRaceEnv(num_envs, pool)
    rng = np.random.default_rng(seed)
    # Hold each random input for eight agent steps so cars build speed and crash.
    total = warmup + steps
    controls = rng.uniform(-1, 1, size=((total + 7) // 8, num_envs, 2))
    controls[..., 1] = rng.uniform(0.25, 1.0, size=controls.shape[:2])
    controls = np.repeat(controls, 8, axis=0)[:total]
    rates, reset_counts = [], []
    for repeat in range(repeats):
        env.reset(seed=seed + repeat)
        for action in controls[:warmup]:
            env.step(action)
        resets = 0
        started = perf_counter()
        for action in controls[warmup:]:
            _, _, terminated, truncated, _ = env.step(action)
            resets += int(np.count_nonzero(terminated | truncated))
        elapsed = perf_counter() - started
        rates.append(num_envs * steps / elapsed)
        reset_counts.append(resets)
    env.close()
    return {
        "num_envs": num_envs,
        "track_count": track_count,
        "track_seeds": [track.seed for track in pool.tracks],
        "seed": seed,
        "batch_steps_per_repeat": steps,
        "warmup_batch_steps": warmup,
        "repeats": repeats,
        "agent_steps_per_second": median(rates),
        "rates": rates,
        "resets_per_repeat": reset_counts,
        "setup_seconds": setup_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "timing": "physics + progress + observations + reward + info + same-step autoreset",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--tracks", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optionally save the full result as JSON")
    args = parser.parse_args()
    if min(args.envs, args.tracks, args.steps, args.repeats) <= 0 or args.warmup < 0:
        parser.error("counts must be positive and warmup must be nonnegative")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    result = benchmark(args.envs, args.tracks, args.steps, args.warmup, args.repeats, args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Weighted reward terms, shared by scalar and batched environments."""

import numpy as np

from sim.config import RewardConfig
from sim.geometry import FloatArray

REWARD_NAMES = ("progress", "time", "crash", "lap")


def reward_terms(
    progress: FloatArray, crashes: FloatArray, laps: FloatArray, config: RewardConfig
) -> FloatArray:
    """Return each weighted contribution, not just its combined reward."""
    return np.column_stack(
        (
            progress * config.progress,
            np.full(len(progress), config.time),
            crashes * config.crash,
            laps * config.lap,
        )
    )

from dataclasses import astuple

import numpy as np

from sim.config import PhysicsConfig
from sim.physics import Action, CarState, step, step_batch


def test_batch_matches_scalar_for_600_ticks(physics: PhysicsConfig) -> None:
    rng = np.random.default_rng(604)
    initial = np.zeros((12, 7))
    initial[:, :2] = rng.normal(0, 10, (12, 2))
    initial[:, 2] = rng.uniform(-np.pi, np.pi, 12)
    initial[:, 3] = np.linspace(-5, 60, 12)
    initial[:, 4:6] = rng.normal(0, 0.5, (12, 2))
    scalar = [CarState(*row) for row in initial]
    batch = initial.copy()
    for tick in range(600):
        if tick % 30 == 0:
            actions = rng.uniform(-1, 1, (12, 2))
        untouched = batch.copy()
        next_batch = step_batch(batch, actions, physics)
        np.testing.assert_array_equal(batch, untouched)
        scalar = [
            step(state, Action(*action), physics)
            for state, action in zip(scalar, actions, strict=True)
        ]
        batch = next_batch
        np.testing.assert_allclose(batch, [astuple(state) for state in scalar], rtol=0, atol=1e-9)

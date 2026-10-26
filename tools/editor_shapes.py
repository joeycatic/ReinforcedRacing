"""Deterministic world-space drawings used by both builder test suites."""

import numpy as np

from sim.geometry import FloatArray


def shapes() -> dict[str, tuple[FloatArray, list[str]]]:
    t = np.arange(24) * (2 * np.pi / 24)
    loop = np.column_stack((110 * np.cos(t), 110 * np.sin(t)))
    u = t - 0.1 * np.sin(4 * t)
    figure8 = np.column_stack(
        (150 * np.cos(u) / (1 + np.sin(u) ** 2), 150 * np.cos(u) * np.sin(u) / (1 + np.sin(u) ** 2))
    )
    # Long parallel sections separated by ~7.9 m after scaling, with wide lobes.
    x = np.array([150, 140, 120, 100, 70, 40, 0, -40, -70, -100, -120, -140])
    y = np.array([0, 35, 50, 35, 4.5, 4.5, 4.5, 4.5, 4.5, 35, 50, 35])
    parallel_straights = np.vstack((np.column_stack((x, y)), np.column_stack((-x, -y))))
    return {
        "loop": (loop, []),
        "figure8": (figure8, []),
        "pretzel": (np.column_stack((120 * np.cos(t), 30 * np.sin(3 * t))), []),
        "shallow": (
            figure8 * np.column_stack((np.ones(24), 0.2 + 0.5 * np.cos(u) ** 2)),
            ["shallow-crossing"],
        ),
        "near_start": (np.roll(figure8, -6, axis=0), ["crossing-start"]),
        "overlap": (parallel_straights, ["overlap"]),
        "fold": (loop * [1, 0.18], ["fold"]),
        "tight": (loop * [1, 0.22], []),
    }

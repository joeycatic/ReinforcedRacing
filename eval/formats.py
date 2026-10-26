"""Versioned replay and heatmap codecs, independent of policy inference."""

from typing import Any

import numpy as np

from eval.config import ExportConfig
from sim.geometry import FloatArray
from sim.track import Track

REPLAY_VERSION = "replay.v1"
HEATMAP_VERSION = "heatmap.v1"
REPLAY_COLUMNS = ["x", "y", "psi", "speed", "steer", "throttle"]


def encode_replay(
    frames: FloatArray, metadata: dict[str, Any], dt: float, config: ExportConfig
) -> dict:
    scales = [
        config.position_scale,
        config.position_scale,
        config.heading_scale,
        config.speed_scale,
        config.action_scale,
        config.action_scale,
    ]
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Replay frames must be a nonempty finite (ticks,6) array")
    # Quantize each absolute sample first, so rounding error cannot accumulate.
    quantized = np.rint(values * np.array(scales)).astype(np.int64)
    deltas = np.vstack((quantized[0], np.diff(quantized, axis=0)))
    return {
        "version": REPLAY_VERSION,
        "dt": dt,
        "columns": REPLAY_COLUMNS,
        "scales": scales,
        "metadata": metadata,
        "deltas": deltas.tolist(),
    }


def decode_replay(data: dict) -> FloatArray:
    if data.get("version") != REPLAY_VERSION or data.get("columns") != REPLAY_COLUMNS:
        raise ValueError("Unsupported replay version or columns")
    scales = np.asarray(data["scales"], dtype=np.float64)
    deltas = np.asarray(data["deltas"])
    if (
        scales.shape != (6,)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or deltas.ndim != 2
        or deltas.shape[1] != 6
        or not len(deltas)
        or not np.issubdtype(deltas.dtype, np.integer)
    ):
        raise ValueError("Invalid replay scales or integer deltas")
    return np.cumsum(deltas, axis=0, dtype=np.int64) / scales


class Heatmap:
    def __init__(self, track: Track, config: ExportConfig) -> None:
        points = np.vstack((track.left, track.right))
        self.cell_size = config.heatmap_cell_size
        self.origin = (
            np.floor((points.min(axis=0) - config.heatmap_margin) / self.cell_size) * self.cell_size
        )
        self.size = (
            np.ceil(
                (points.max(axis=0) + config.heatmap_margin - self.origin) / self.cell_size
            ).astype(int)
            + 1
        )
        self.visits = np.zeros(int(np.prod(self.size)), dtype=np.int64)
        self.crashes = np.zeros_like(self.visits)
        self.positions: list[list[float]] = []

    def add(self, positions: FloatArray, crashed: np.ndarray) -> None:
        cells = np.floor((positions - self.origin) / self.cell_size).astype(np.int64)
        if not np.all((cells >= 0) & (cells < self.size)):
            raise ValueError("Heatmap position outside configured bounds; increase heatmap_margin")
        bins = cells[:, 1] * self.size[0] + cells[:, 0]
        np.add.at(self.visits, bins, 1)
        np.add.at(self.crashes, bins[crashed], 1)
        self.positions.extend(positions[crashed].tolist())

    def to_dict(self, metadata: dict[str, Any]) -> dict:
        def sparse(counts: np.ndarray) -> list[list[int]]:
            nonzero = np.flatnonzero(counts)
            return np.column_stack((nonzero, counts[nonzero])).tolist()

        return {
            "version": HEATMAP_VERSION,
            "metadata": metadata,
            "grid": {
                "encoding": "sparse.flat.v1",
                "origin": self.origin.tolist(),
                "cell_size": self.cell_size,
                "width": int(self.size[0]),
                "height": int(self.size[1]),
            },
            "visits": sparse(self.visits),
            "crash_bins": sparse(self.crashes),
            "crash_positions": self.positions,
            "visit_count": int(self.visits.sum()),
            "crash_count": len(self.positions),
        }

"""Windowed projection, unwrapped progress, sequential checkpoints and lap timing."""

from dataclasses import dataclass
from math import floor

import numpy as np

from sim.config import TrackConfig
from sim.geometry import cross
from sim.track import Track


@dataclass(frozen=True)
class Projection:
    index: int
    arc: float
    offset: float
    distance: float


def project(track: Track, x: float, y: float, index: int | None, window: int) -> Projection:
    """Global search only at initialization; each tick examines 2*window+1 segments."""
    count = len(track.centerline)
    indices = (
        np.arange(count) if index is None else (index + np.arange(-window, window + 1)) % count
    )
    start = track.centerline[indices]
    vector = track.centerline[(indices + 1) % count] - start
    length_sq = np.sum(vector * vector, axis=1)
    relative = np.array([x, y], dtype=np.float64) - start
    fraction = np.clip(np.sum(relative * vector, axis=1) / length_sq, 0.0, 1.0)
    residual = relative - fraction[:, None] * vector
    winner = int(np.argmin(np.sum(residual * residual, axis=1)))
    chosen = int(indices[winner])
    length = float(np.sqrt(length_sq[winner]))
    arc = float(track.arc_length[chosen] + fraction[winner] * length) % track.length
    offset = float(cross(vector[winner], residual[winner]) / length)
    return Projection(chosen, arc, offset, float(np.linalg.norm(residual[winner])))


class ProgressTracker:
    def __init__(self, track: Track, config: TrackConfig) -> None:
        self.track = track
        self.config = config
        self.projection = project(track, *track.centerline[0], None, config.projection_window)
        self.unwrapped = 0.0
        self.delta = 0.0
        self.wrong_way = False
        self.laps = 0
        self.lap_times: list[float] = []
        self.lap_started_at = 0.0
        self.last_checkpoint = 0.0  # Unwrapped arc, including lap number.
        self.next_checkpoint = 1  # Global ordered checkpoint ordinal; start is ordinal zero.

    def _checkpoint_arc(self, ordinal: int) -> float:
        lap, index = divmod(ordinal, len(self.track.checkpoints))
        return lap * self.track.length + float(self.track.checkpoints[index])

    def update(self, x: float, y: float, tick_end_time: float, dt: float) -> None:
        projection = project(self.track, x, y, self.projection.index, self.config.projection_window)
        delta = projection.arc - self.projection.arc
        delta = (delta + self.track.length / 2.0) % self.track.length - self.track.length / 2.0
        previous = self.unwrapped
        self.unwrapped += delta
        self.delta = delta
        self.wrong_way = delta < -self.config.wrong_way_min_delta
        self.projection = projection
        while delta > 0.0 and self.unwrapped >= self._checkpoint_arc(self.next_checkpoint):
            checkpoint = self._checkpoint_arc(self.next_checkpoint)
            self.last_checkpoint = checkpoint
            if self.next_checkpoint % len(self.track.checkpoints) == 0:
                fraction = (checkpoint - previous) / delta
                crossed_at = tick_end_time - dt + fraction * dt
                self.lap_times.append(crossed_at - self.lap_started_at)
                self.lap_started_at = crossed_at
                self.laps += 1
            self.next_checkpoint += 1

    def respawn(self) -> tuple[float, float, float]:
        position, heading = self.track.sample(self.last_checkpoint)
        self.unwrapped = self.last_checkpoint
        self.projection = project(
            self.track,
            float(position[0]),
            float(position[1]),
            int(np.searchsorted(self.track.arc_length, self.last_checkpoint % self.track.length))
            % len(self.track.centerline),
            self.config.projection_window,
        )
        self.delta = 0.0
        self.wrong_way = False
        # Checkpoint ordinal remains monotonic: respawn never awards progress/laps.
        return float(position[0]), float(position[1]), heading

    @property
    def checkpoint_index(self) -> int:
        return (self.next_checkpoint - 1) % len(self.track.checkpoints)

    @property
    def lap_fraction(self) -> float:
        return self.unwrapped / self.track.length - floor(self.unwrapped / self.track.length)

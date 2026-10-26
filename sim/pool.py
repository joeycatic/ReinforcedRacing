"""Static flattened geometry shared by all cars; generation/precomputation happen once."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from sim.config import (
    load_observation,
    load_physics,
    load_pool_config,
    load_tracks,
)
from sim.geometry import FloatArray
from sim.raycast import BoundaryGrid, IntArray
from sim.track import Track, generate_track, showcase_track


class TrackPool:
    def __init__(self, tracks: Sequence[Track], *, cell_size: float | None = None) -> None:
        if not tracks:
            raise ValueError("A track pool must not be empty")
        self.tracks = tuple(tracks)
        self.counts = np.array([len(track.centerline) for track in tracks], dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.lengths = np.array([track.length for track in tracks])
        self.arc_offsets = np.r_[0.0, np.cumsum(self.lengths)[:-1]]
        self.widths = np.array([track.width for track in tracks])
        self.points = np.concatenate([track.centerline for track in tracks])
        self.vectors = np.concatenate(
            [np.roll(track.centerline, -1, axis=0) - track.centerline for track in tracks]
        )
        self.segment_lengths = np.concatenate([np.diff(track.arc_length) for track in tracks])
        self.arc = np.concatenate([track.arc_length[:-1] for track in tracks])
        self.arc_keys = np.concatenate(
            [
                track.arc_length[:-1] + offset
                for track, offset in zip(tracks, self.arc_offsets, strict=True)
            ]
        )
        self.tangents = np.concatenate([track.tangents for track in tracks])
        self.curvature = np.concatenate([track.curvature for track in tracks])
        self.checkpoint_counts = np.array([len(track.checkpoints) for track in tracks])
        max_checkpoints = int(self.checkpoint_counts.max())
        self.checkpoints = np.zeros((len(tracks), max_checkpoints))
        for index, track in enumerate(tracks):
            self.checkpoints[index, : len(track.checkpoints)] = track.checkpoints
        self.starts = np.zeros((len(tracks), 7))
        for index, track in enumerate(tracks):
            position, heading = track.sample(0)
            self.starts[index, :3] = [position[0], position[1], heading]
        resolved_cell_size = load_observation().grid_cell_size if cell_size is None else cell_size
        self.grid = BoundaryGrid(tracks, resolved_cell_size)

    def hidden_tags(self, track_ids: IntArray, arcs: FloatArray) -> np.ndarray | None:
        if not self.grid.tag_count:
            return None
        from sim.crossings import branch

        hidden = np.zeros((len(track_ids), self.grid.tag_count + 1), dtype=np.bool_)
        tag = 1
        for index, track in enumerate(self.tracks):
            cars = np.flatnonzero(track_ids == index)
            for crossing in track.crossings:
                for car in cars:
                    hidden[car, tag + 1 - branch(crossing, float(arcs[car]), track.length)] = True
                tag += 2
        return hidden

    def curvature_at(self, track_ids: IntArray, arcs: FloatArray) -> FloatArray:
        """Linear interpolation across nodes, including the closing segment."""
        wrapped = arcs % self.lengths[track_ids, None]
        keys = wrapped + self.arc_offsets[track_ids, None]
        flat = np.searchsorted(self.arc_keys, keys, side="right") - 1
        local = flat - self.offsets[track_ids, None]
        following = self.offsets[track_ids, None] + (local + 1) % self.counts[track_ids, None]
        fraction = (wrapped - self.arc[flat]) / self.segment_lengths[flat]
        return self.curvature[flat] + fraction * (self.curvature[following] - self.curvature[flat])


def make_track_pool(
    split: Literal["showcase", "training", "held_out"] = "showcase",
    *,
    count: int | None = None,
) -> TrackPool:
    config, showcases = load_tracks()
    physics = load_physics()
    if split == "showcase":
        if count is not None and not 1 <= count <= len(showcases):
            raise ValueError("Requested showcase count exceeds the configured pool")
        return TrackPool([showcase_track(item, config, physics) for item in showcases[:count]])
    pools = load_pool_config()
    if split == "training":
        start, available = pools.training_seed_start, pools.training_count
    elif split == "held_out":
        start, available = pools.held_out_seed_start, pools.held_out_count
    else:
        raise ValueError(f"Unknown pool split: {split}")
    requested = available if count is None else count
    if not 1 <= requested <= available:
        raise ValueError(f"Requested pool count must be between 1 and {available}")
    return TrackPool([generate_track(start + index, config, physics) for index in range(requested)])

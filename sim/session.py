"""Manual-driving session: fixed ticks, crashes, delayed/instant checkpoint respawn.

Training mode only supplies crash termination here. Gymnasium/rewards are M2.
"""

from math import ceil
from typing import Literal

from sim.config import PhysicsConfig, TrackConfig
from sim.physics import Action, CarState, step
from sim.progress import ProgressTracker
from sim.track import Track


class DriveSession:
    def __init__(
        self,
        track: Track,
        physics: PhysicsConfig,
        tracks: TrackConfig,
        mode: Literal["play", "training"] = "play",
    ) -> None:
        if mode not in ("play", "training"):
            raise ValueError(f"Unknown driving mode: {mode}")
        self.track = track
        self.physics = physics
        self.tracks = tracks
        self.mode = mode
        self.restart()

    def restart(self) -> None:
        self.progress = ProgressTracker(self.track, self.tracks)
        self.state = CarState(*self.progress.respawn())
        self.ticks = 0
        self.respawn_ticks = 0
        self.crashed = False
        self.crashes = 0
        self.terminated = False

    @property
    def time(self) -> float:
        return self.ticks * self.physics.dt

    @property
    def lap_time(self) -> float:
        return self.time - self.progress.lap_started_at

    def respawn(self) -> None:
        self.state = CarState(*self.progress.respawn())
        self.respawn_ticks = 0
        self.crashed = False

    def tick(self, action: Action) -> None:
        if self.terminated:
            return
        self.ticks += 1
        if self.respawn_ticks:
            self.respawn_ticks -= 1
            if not self.respawn_ticks:
                self.respawn()
            return
        self.state = step(self.state, action, self.physics)
        self.progress.update(self.state.x, self.state.y, self.time, self.physics.dt)
        # Literal spec threshold: center beyond the road edge by half a car width.
        wall_limit = (self.track.width + self.physics.car_width) / 2.0
        if self.progress.projection.distance > wall_limit:
            self.crashed = True
            self.crashes += 1
            if self.mode == "training":
                self.terminated = True
            else:
                self.respawn_ticks = ceil(self.physics.respawn_delay / self.physics.dt)

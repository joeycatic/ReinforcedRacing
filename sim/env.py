"""Standard single-car Gymnasium environment, using the scalar M1 simulation."""

from dataclasses import astuple
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.config import (
    EnvConfig,
    ObservationConfig,
    PhysicsConfig,
    RewardConfig,
    TrackConfig,
    load_env,
    load_observation,
    load_physics,
    load_reward,
    load_tracks,
)
from sim.geometry import FloatArray
from sim.observation import OBS_SIZE, ObservationBuilder
from sim.physics import Action
from sim.pool import TrackPool, make_track_pool
from sim.reward import REWARD_NAMES, reward_terms
from sim.session import DriveSession


def validate_actions(actions: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Actions must be finite with shape {shape}")
    return np.clip(values, -1.0, 1.0)


class RaceEnv(gym.Env[FloatArray, FloatArray]):
    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        pool: TrackPool | None = None,
        *,
        physics: PhysicsConfig | None = None,
        tracks: TrackConfig | None = None,
        observation: ObservationConfig | None = None,
        reward: RewardConfig | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        self.pool = pool if pool is not None else make_track_pool()
        self.physics = physics or load_physics()
        self.tracks = tracks or load_tracks()[0]
        self.config = config or load_env()
        self.reward_config = reward or load_reward()
        self.builder = ObservationBuilder(self.pool, observation or load_observation())
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float64)
        self._needs_reset = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "track_index" in options:
            index = options["track_index"]
            if type(index) is not int or not 0 <= index < len(self.pool.tracks):
                raise ValueError("track_index is outside the pool")
            self.track_index = index
        else:
            self.track_index = int(self.np_random.integers(len(self.pool.tracks)))
        self.session = DriveSession(
            self.pool.tracks[self.track_index], self.physics, self.tracks, mode="training"
        )
        self.previous_action = np.zeros(2)
        self.agent_steps = 0
        self.episode_return = 0.0
        self.reward_totals = np.zeros(len(REWARD_NAMES))
        self._needs_reset = False
        return self._observation(), self._info(np.zeros(len(REWARD_NAMES)), 0.0)

    def _observation(self) -> FloatArray:
        progress = self.session.progress
        return self.builder.build(
            np.array([astuple(self.session.state)], dtype=np.float64),
            np.array([self.track_index]),
            np.array([progress.projection.index]),
            np.array([progress.projection.arc]),
            np.array([progress.projection.offset]),
            self.previous_action[None, :],
        )[0]

    def _info(self, terms: FloatArray, delta: float) -> dict[str, Any]:
        progress = self.session.progress
        return {
            "track_index": self.track_index,
            "track_id": self.session.track.id,
            "progress": progress.unwrapped,
            "delta_progress": delta,
            "laps": progress.laps,
            "lap_time": self.session.lap_time,
            "best_lap_time": min(progress.lap_times, default=0.0),
            "wrong_way": progress.wrong_way,
            "crashed": self.session.crashed,
            "physics_ticks": self.session.ticks,
            "agent_steps": self.agent_steps,
            "episode_return": self.episode_return,
            "reward_terms": dict(zip(REWARD_NAMES, map(float, terms), strict=True)),
        }

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before stepping a new/finished RaceEnv")
        values = validate_actions(action, (2,))
        control = Action(*values)
        progress = self.session.progress
        previous_progress, previous_laps = progress.unwrapped, progress.laps
        for _ in range(self.physics.action_repeat):
            self.session.tick(control)
            if self.session.terminated or progress.laps >= self.config.max_laps:
                break
        self.agent_steps += 1
        delta = progress.unwrapped - previous_progress
        terms = reward_terms(
            np.array([delta]),
            np.array([self.session.crashed]),
            np.array([progress.laps - previous_laps]),
            self.reward_config,
        )[0]
        reward = float(terms.sum())
        self.episode_return += reward
        self.reward_totals += terms
        self.previous_action = values
        terminated = self.session.terminated
        truncated = not terminated and (
            progress.laps >= self.config.max_laps or self.agent_steps >= self.config.max_agent_steps
        )
        info = self._info(terms, delta)
        self._needs_reset = terminated or truncated
        if self._needs_reset:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.agent_steps,
                "laps": progress.laps,
                "best_lap_time": info["best_lap_time"],
                "crashed": terminated,
                "reward_terms": dict(
                    zip(REWARD_NAMES, map(float, self.reward_totals), strict=True)
                ),
            }
        return self._observation(), reward, terminated, truncated, info

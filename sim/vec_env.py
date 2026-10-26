"""Native batched NumPy environment. No collection of single-car environments.

Gymnasium same-step autoreset: returned observations for finished rows are reset
observations; final_obs/_final_obs preserve terminal observations for bootstrapping.
Info metrics describe the transition; reset_track_index identifies the new track.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

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
from sim.env import validate_actions
from sim.geometry import FloatArray
from sim.observation import OBS_SIZE, ObservationBuilder
from sim.physics import step_batch
from sim.pool import TrackPool, make_track_pool
from sim.raycast import IntArray
from sim.reward import REWARD_NAMES, reward_terms

NDArrayBool = NDArray[np.bool_]


class VecRaceEnv(VectorEnv):
    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs: int = 64,
        pool: TrackPool | None = None,
        *,
        physics: PhysicsConfig | None = None,
        tracks: TrackConfig | None = None,
        observation: ObservationConfig | None = None,
        reward: RewardConfig | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.pool = pool if pool is not None else make_track_pool()
        self.physics = physics or load_physics()
        self.tracks = tracks or load_tracks()[0]
        self.config = config or load_env()
        self.reward_config = reward or load_reward()
        self.builder = ObservationBuilder(self.pool, observation or load_observation())
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self.single_observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float64)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.autoreset_mode = AutoresetMode.SAME_STEP
        self.states = np.zeros((num_envs, 7))
        self.track_ids = np.zeros(num_envs, dtype=np.int64)
        self.indices = np.zeros(num_envs, dtype=np.int64)
        self.arcs = np.zeros(num_envs)
        self.offsets = np.zeros(num_envs)
        self.progress_meters = np.zeros(num_envs)
        self.wrong_way = np.zeros(num_envs, dtype=bool)
        self.ticks = np.zeros(num_envs, dtype=np.int64)
        self.agent_steps = np.zeros(num_envs, dtype=np.int64)
        self.laps = np.zeros(num_envs, dtype=np.int64)
        self.next_checkpoint = np.ones(num_envs, dtype=np.int64)
        self.lap_started_at = np.zeros(num_envs)
        self.best_lap_time = np.full(num_envs, np.inf)
        self.previous_actions = np.zeros((num_envs, 2))
        self.episode_returns = np.zeros(num_envs)
        self.reward_totals = np.zeros((num_envs, len(REWARD_NAMES)))
        self._window = np.arange(-self.tracks.projection_window, self.tracks.projection_window + 1)
        self._rows = np.arange(num_envs)
        self._needs_reset = True

    def _project(self) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
        pool = self.pool
        local = (self.indices[:, None] + self._window) % pool.counts[self.track_ids, None]
        flat = pool.offsets[self.track_ids, None] + local
        vector = pool.vectors[flat]
        relative = self.states[:, None, :2] - pool.points[flat]
        length_sq = np.sum(vector * vector, axis=2)
        fraction = np.clip(np.sum(relative * vector, axis=2) / length_sq, 0.0, 1.0)
        residual = relative - fraction[..., None] * vector
        squared_distance = np.sum(residual * residual, axis=2)
        winner = np.argmin(squared_distance, axis=1)
        chosen = flat[self._rows, winner]
        length = np.sqrt(length_sq[self._rows, winner])
        arc = (pool.arc[chosen] + fraction[self._rows, winner] * length) % pool.lengths[
            self.track_ids
        ]
        selected_residual = residual[self._rows, winner]
        selected_vector = vector[self._rows, winner]
        offset = (
            selected_vector[:, 0] * selected_residual[:, 1]
            - selected_vector[:, 1] * selected_residual[:, 0]
        ) / length
        distance = np.sqrt(squared_distance[self._rows, winner])
        return local[self._rows, winner], arc, offset, distance

    def _reset_rows(self, rows: IntArray, choices: IntArray | None = None) -> None:
        if choices is None:
            choices = self.np_random.integers(len(self.pool.tracks), size=len(rows))
        self.track_ids[rows] = choices
        self.states[rows] = self.pool.starts[choices]
        self.indices[rows] = 0
        indices, arcs, offsets, _ = self._project()
        self.indices[rows], self.arcs[rows], self.offsets[rows] = (
            indices[rows],
            arcs[rows],
            offsets[rows],
        )
        self.progress_meters[rows] = 0.0
        self.wrong_way[rows] = False
        self.ticks[rows] = 0
        self.agent_steps[rows] = 0
        self.laps[rows] = 0
        self.next_checkpoint[rows] = 1
        self.lap_started_at[rows] = 0.0
        self.best_lap_time[rows] = np.inf
        self.previous_actions[rows] = 0.0
        self.episode_returns[rows] = 0.0
        self.reward_totals[rows] = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        choices = None
        if options is not None and "track_indices" in options:
            choices = np.asarray(options["track_indices"])
            if (
                choices.shape != (self.num_envs,)
                or not np.issubdtype(choices.dtype, np.integer)
                or np.any((choices < 0) | (choices >= len(self.pool.tracks)))
            ):
                raise ValueError("track_indices must contain one valid integer pool index per car")
        self._reset_rows(self._rows, choices)
        self._needs_reset = False
        return self._observation(), self._info(
            np.zeros_like(self.reward_totals), np.zeros(self.num_envs)
        )

    def _observation(self) -> FloatArray:
        return self.builder.build(
            self.states,
            self.track_ids,
            self.indices,
            self.arcs,
            self.offsets,
            self.previous_actions,
        )

    def _tick(self, actions: FloatArray, active: NDArrayBool) -> NDArrayBool:
        self.states[active] = step_batch(self.states, actions, self.physics)[active]
        self.ticks[active] += 1
        indices, arcs, offsets, distance = self._project()
        length = self.pool.lengths[self.track_ids]
        delta = (arcs - self.arcs + length / 2.0) % length - length / 2.0
        previous = self.progress_meters.copy()
        self.progress_meters[active] += delta[active]
        self.wrong_way[active] = delta[active] < -self.tracks.wrong_way_min_delta
        self.indices[active], self.arcs[active], self.offsets[active] = (
            indices[active],
            arcs[active],
            offsets[active],
        )
        counts = self.pool.checkpoint_counts[self.track_ids]
        while True:
            checkpoint_index = self.next_checkpoint % counts
            checkpoint_arc = (
                self.next_checkpoint // counts * length
                + self.pool.checkpoints[self.track_ids, checkpoint_index]
            )
            passed = active & (delta > 0.0) & (self.progress_meters >= checkpoint_arc)
            if not np.any(passed):
                break
            finish = passed & (checkpoint_index == 0)
            if np.any(finish):
                fraction = (checkpoint_arc[finish] - previous[finish]) / delta[finish]
                crossed_at = (
                    self.ticks[finish] * self.physics.dt
                    - self.physics.dt
                    + fraction * self.physics.dt
                )
                lap_time = crossed_at - self.lap_started_at[finish]
                self.best_lap_time[finish] = np.minimum(self.best_lap_time[finish], lap_time)
                self.lap_started_at[finish] = crossed_at
                self.laps[finish] += 1
            self.next_checkpoint[passed] += 1
        return active & (
            distance > (self.pool.widths[self.track_ids] + self.physics.car_width) / 2.0
        )

    def _info(self, terms: FloatArray, delta: FloatArray) -> dict[str, Any]:
        return {
            "track_index": self.track_ids.copy(),
            "progress": self.progress_meters.copy(),
            "delta_progress": delta.copy(),
            "laps": self.laps.copy(),
            "lap_time": self.ticks * self.physics.dt - self.lap_started_at,
            "best_lap_time": np.where(np.isfinite(self.best_lap_time), self.best_lap_time, 0.0),
            "wrong_way": self.wrong_way.copy(),
            "physics_ticks": self.ticks.copy(),
            "agent_steps": self.agent_steps.copy(),
            "episode_return": self.episode_returns.copy(),
            "reward_terms": {name: terms[:, i].copy() for i, name in enumerate(REWARD_NAMES)},
        }

    def step(
        self, actions: FloatArray
    ) -> tuple[FloatArray, FloatArray, NDArrayBool, NDArrayBool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before stepping VecRaceEnv")
        values = validate_actions(actions, (self.num_envs, 2))
        before, previous_laps = self.progress_meters.copy(), self.laps.copy()
        terminated = np.zeros(self.num_envs, dtype=bool)
        active = np.ones(self.num_envs, dtype=bool)
        for _ in range(self.physics.action_repeat):
            terminated |= self._tick(values, active)
            active &= ~terminated & (self.laps < self.config.max_laps)
        self.agent_steps += 1
        delta = self.progress_meters - before
        terms = reward_terms(delta, terminated, self.laps - previous_laps, self.reward_config)
        rewards = terms.sum(axis=1)
        self.episode_returns += rewards
        self.reward_totals += terms
        self.previous_actions[:] = values
        truncated = ~terminated & (
            (self.laps >= self.config.max_laps) | (self.agent_steps >= self.config.max_agent_steps)
        )
        observations = self._observation()
        info = self._info(terms, delta)
        info["crashed"] = terminated.copy()
        done = terminated | truncated
        if np.any(done):
            info["episode"] = {
                "r": self.episode_returns.copy(),
                "l": self.agent_steps.copy(),
                "laps": self.laps.copy(),
                "best_lap_time": info["best_lap_time"].copy(),
                "crashed": terminated.copy(),
                "reward_terms": {
                    name: self.reward_totals[:, i].copy() for i, name in enumerate(REWARD_NAMES)
                },
            }
            info["_episode"] = done.copy()
            final_info = info.copy()
            info["final_info"] = final_info
            info["_final_info"] = done.copy()
            info["final_obs"] = observations.copy()
            info["_final_obs"] = done.copy()
            self._reset_rows(np.flatnonzero(done))
            # Build only reset rows' sensors, preserving terminal observations in info.
            observations[done] = self.builder.build(
                self.states[done],
                self.track_ids[done],
                self.indices[done],
                self.arcs[done],
                self.offsets[done],
                self.previous_actions[done],
            )
            info["reset_track_index"] = self.track_ids.copy()
        return observations, rewards, terminated, truncated, info

"""Per-physics-tick recording on the unchanged native batched simulation."""

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from eval.config import ExportConfig
from eval.evaluate import summarize
from eval.formats import Heatmap, encode_replay
from sim.config import load_env
from sim.pool import TrackPool
from sim.vec_env import VecRaceEnv
from train.config import PPOConfig
from train.ppo import ActorCritic, RunningMeanStd


class RecordingEnv(VecRaceEnv):
    def __init__(self, pool: TrackPool, config: ExportConfig, runs: int, replay: bool) -> None:
        self.assignments = np.repeat(np.arange(len(pool.tracks)), runs)
        super().__init__(len(self.assignments), pool, config=replace(load_env(), max_laps=1))
        self.recording = np.ones(self.num_envs, dtype=bool)
        self.maps = [Heatmap(track, config) for track in pool.tracks]
        self.frames: list[list[list[float]]] | None = None
        if replay:
            self.frames = [[[*pool.starts[index, :3], 0.0, 0.0, 0.0]] for index in self.assignments]

    def _tick(self, actions: np.ndarray, active: np.ndarray) -> np.ndarray:
        eligible = active & self.recording
        crashed = super()._tick(actions, eligible)
        for index, heatmap in enumerate(self.maps):
            selected = eligible & (self.assignments == index)
            heatmap.add(self.states[selected, :2], crashed[selected])
        if self.frames is not None:
            for row in np.flatnonzero(eligible):
                state = self.states[row]
                self.frames[row].append(
                    [*state[:3], float(np.hypot(state[3], state[4])), *actions[row]]
                )
        return crashed


@torch.no_grad()
def record_runs(
    checkpoint: dict, pool: TrackPool, config: ExportConfig, *, deterministic: bool
) -> tuple[dict, list[dict], list[dict]]:
    ppo = PPOConfig(**checkpoint["config"])
    torch.set_num_threads(ppo.cpu_threads)
    model = ActorCritic(ppo)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = RunningMeanStd((22,), ppo.initial_stats_count)
    stats.load_state_dict(checkpoint["obs_stats"])
    runs = 1 if deterministic else config.stochastic_runs
    env = RecordingEnv(pool, config, runs, replay=deterministic)
    observations, _ = env.reset(seed=config.seed, options={"track_indices": env.assignments})
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    row_records: dict[int, dict] = {}
    try:
        for _ in range(env.config.max_agent_steps):
            normalized = stats.normalize(
                observations, ppo.observation_clip, ppo.normalization_epsilon
            )
            actions = model.actor(torch.from_numpy(normalized).float()).numpy()
            if not deterministic:
                actions += model.log_std.exp().numpy() * rng.standard_normal(actions.shape)
            actions[~env.recording] = 0.0
            observations, _, terminated, truncated, info = env.step(actions.clip(-1, 1))
            completed = (terminated | truncated) & env.recording
            for row in np.flatnonzero(completed):
                track = pool.tracks[env.assignments[row]]
                success = bool(info["laps"][row] >= 1)
                record = {
                    "track_id": track.id,
                    "track_seed": track.seed,
                    "run": int(row % runs),
                    "finished": success,
                    "crashed": bool(terminated[row]),
                    "truncated": bool(truncated[row] and not success),
                    "lap_time": float(info["best_lap_time"][row]) if success else None,
                    "return": float(info["episode_return"][row]),
                    "progress": float(info["progress"][row]),
                    "agent_steps": int(info["agent_steps"][row]),
                    "physics_ticks": int(info["physics_ticks"][row]),
                }
                records.append(record)
                row_records[int(row)] = record
            env.recording[completed] = False
            if not env.recording.any():
                break
        if env.recording.any():
            raise RuntimeError("Missing recording results at episode limit")
        common = {
            "snapshot_step": checkpoint["global_step"],
            "snapshot_id": str(checkpoint["requested_step"]),
            "checkpoint_sha256": checkpoint["source_sha256"],
            "versions": checkpoint["versions"],
            "seed": config.seed,
            "runs": runs,
            "deterministic": deterministic,
        }
        replays = []
        if env.frames is not None:
            for row, frames in enumerate(env.frames):
                if len(frames) != row_records[row]["physics_ticks"] + 1:
                    raise RuntimeError("Replay lost physics ticks")
                replays.append(
                    encode_replay(
                        np.asarray(frames), {**common, **row_records[row]}, env.physics.dt, config
                    )
                )
        heatmaps = [
            heatmap.to_dict({**common, "track_id": track.id})
            for heatmap, track in zip(env.maps, pool.tracks, strict=True)
        ]
        summary = {
            "summary": summarize(records),
            "tracks": {
                track.id: summarize([r for r in records if r["track_id"] == track.id])
                for track in pool.tracks
            },
            "episodes": records,
        }
        return summary, replays, heatmaps
    finally:
        env.close()

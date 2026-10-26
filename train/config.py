"""Typed PPO/curriculum settings, separate from the single-file algorithm."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml

from sim.config import ROOT


@dataclass(frozen=True)
class PPOConfig:
    seed: int
    device: str
    cpu_threads: int
    deterministic: bool
    total_timesteps: int
    num_envs: int
    rollout_steps: int
    update_epochs: int
    num_minibatches: int
    hidden_size: int
    hidden_gain: float
    actor_output_gain: float
    critic_output_gain: float
    initial_log_std: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_value_loss: bool
    value_clip_coef: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    learning_rate: float
    anneal_lr: bool
    adam_eps: float
    target_kl: float | None
    normalize_advantage: bool
    observation_clip: float
    reward_clip: float
    normalization_epsilon: float
    initial_stats_count: float
    checkpoint_every_updates: int
    snapshot_steps: tuple[int, ...]
    eval_every_updates: int
    curriculum_eval_runs: int
    curriculum_success_rate: float
    curriculum_min_steps: int
    single_showcase: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and (
                type(value) is not int or value < (0 if field.name == "seed" else 1)
            ):
                raise ValueError(f"Invalid positive integer: {field.name}")
            if field.type is bool and type(value) is not bool:
                raise ValueError(f"{field.name} must be boolean")
            if field.type is float and (isinstance(value, bool) or not isfinite(value)):
                raise ValueError(f"{field.name} must be finite")
        for name in ("gamma", "gae_lambda", "curriculum_success_rate"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in (
            "hidden_gain",
            "actor_output_gain",
            "critic_output_gain",
            "clip_coef",
            "value_clip_coef",
            "max_grad_norm",
            "learning_rate",
            "adam_eps",
            "observation_clip",
            "reward_clip",
            "normalization_epsilon",
            "initial_stats_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.entropy_coef < 0 or self.value_coef < 0:
            raise ValueError("Loss coefficients cannot be negative")
        if self.num_envs * self.rollout_steps % self.num_minibatches:
            raise ValueError("Rollout batch must be divisible by num_minibatches")
        if self.num_envs * self.rollout_steps // self.num_minibatches < 2:
            raise ValueError("Minibatches require at least two samples")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto/cpu/cuda/mps")
        if self.target_kl is not None and (not isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be positive or null")
        if any(type(step) is not int or step < 0 for step in self.snapshot_steps):
            raise ValueError("Snapshot steps must be nonnegative integers")
        if tuple(sorted(set(self.snapshot_steps))) != self.snapshot_steps:
            raise ValueError("Snapshot steps must be unique and increasing")


def load_ppo(path: Path = ROOT / "configs/ppo.yaml") -> PPOConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["snapshot_steps"] = tuple(data["snapshot_steps"])
    return PPOConfig(**data)

"""PPO from scratch: Gaussian policy, GAE, clipping, normalization and checkpoints.

No RL library is used. Physics remains NumPy float64; network tensors are float32.
References: Schulman et al., arXiv:1707.06347 (PPO), arXiv:1506.02438 (GAE).
"""

import argparse
import hashlib
import io
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.distributions import Normal  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from sim import OBS_VERSION, PHYSICS_VERSION, TRACK_VERSION  # noqa: E402
from sim.geometry import FloatArray  # noqa: E402
from sim.observation import OBS_SIZE  # noqa: E402
from sim.pool import TrackPool, make_track_pool  # noqa: E402
from sim.reward import REWARD_NAMES  # noqa: E402
from sim.vec_env import VecRaceEnv  # noqa: E402
from train.config import PPOConfig, load_ppo  # noqa: E402

VERSIONS = {"physics": PHYSICS_VERSION, "obs": OBS_VERSION, "track": TRACK_VERSION}
ENV_ARRAYS = (
    "states",
    "track_ids",
    "indices",
    "arcs",
    "offsets",
    "progress_meters",
    "wrong_way",
    "ticks",
    "agent_steps",
    "laps",
    "next_checkpoint",
    "lap_started_at",
    "best_lap_time",
    "previous_actions",
    "episode_returns",
    "reward_totals",
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS was requested but is unavailable")
    return device


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], initial_count: float) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = initial_count

    def update(self, samples: FloatArray) -> None:
        values = np.asarray(samples, dtype=np.float64)
        if len(values) == 0:
            return
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite normalizer input")
        count = len(values)
        delta = values.mean(axis=0) - self.mean
        total = self.count + count
        moment = self.var * self.count + values.var(axis=0) * count
        moment += delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.var = moment / total
        self.count = total

    def normalize(self, values: FloatArray, clip: float, epsilon: float) -> FloatArray:
        return np.clip((values - self.mean) / np.sqrt(self.var + epsilon), -clip, clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": torch.as_tensor(self.mean.copy()),
            "var": torch.as_tensor(self.var.copy()),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = state["mean"].cpu().numpy().copy()
        self.var = state["var"].cpu().numpy().copy()
        self.count = float(state["count"])


def initialize(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            initialize(nn.Linear(OBS_SIZE, config.hidden_size), config.hidden_gain),
            nn.Tanh(),
            initialize(nn.Linear(config.hidden_size, config.hidden_size), config.hidden_gain),
            nn.Tanh(),
            initialize(nn.Linear(config.hidden_size, 2), config.actor_output_gain),
        )
        self.critic = nn.Sequential(
            initialize(nn.Linear(OBS_SIZE, config.hidden_size), config.hidden_gain),
            nn.Tanh(),
            initialize(nn.Linear(config.hidden_size, config.hidden_size), config.hidden_gain),
            nn.Tanh(),
            initialize(nn.Linear(config.hidden_size, 1), config.critic_output_gain),
        )
        self.log_std = nn.Parameter(torch.full((2,), config.initial_log_std))

    def value(self, observations: Tensor) -> Tensor:
        return self.critic(observations).squeeze(-1)

    def distribution(self, observations: Tensor) -> Normal:
        mean = self.actor(observations)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def evaluate(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(observations)
        return (
            distribution.log_prob(actions).sum(-1),
            distribution.entropy().sum(-1),
            self.value(observations),
        )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Bootstrap truncations from final_obs, but cut recursion across every reset.

    All inputs are (time, env). next_values must already contain the terminal
    observation's value for truncated transitions, never their reset state's value.
    """
    if not all(
        array.shape == rewards.shape for array in (values, next_values, terminated, truncated)
    ):
        raise ValueError("GAE arrays must have identical (time, env) shapes")
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * (~terminated[step]) * next_values[step] - values[step]
        following = delta + gamma * gae_lambda * (~(terminated[step] | truncated[step])) * following
        advantages[step] = following
    return advantages, advantages + values


def single_pool(track_id: str) -> TrackPool:
    pool = make_track_pool("showcase")
    selected = [track for track in pool.tracks if track.id == track_id]
    if not selected:
        raise ValueError(f"Unknown showcase track {track_id}")
    return TrackPool(selected)


def pool_signature(pool: TrackPool) -> str:
    digest = hashlib.sha256()
    for track in pool.tracks:
        digest.update(json.dumps(track.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


class Trainer:
    def __init__(
        self,
        config: PPOConfig,
        pool: TrackPool,
        run_dir: Path,
        *,
        stage: str = "single",
        tensorboard: bool = True,
    ) -> None:
        self.config = config
        self.device = choose_device(config.device)
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(config.deterministic)
        self.rng = np.random.default_rng(config.seed)
        self.model = ActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=config.adam_eps
        )
        self.obs_stats = RunningMeanStd((OBS_SIZE,), config.initial_stats_count)
        self.return_stats = RunningMeanStd((), config.initial_stats_count)
        self.env = VecRaceEnv(config.num_envs, pool)
        self.observations, _ = self.env.reset(seed=config.seed)
        self.obs_stats.update(self.observations)
        self.discounted_returns = np.zeros(config.num_envs)
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(run_dir / "tensorboard")) if tensorboard else None
        self.stage = stage
        self.global_step = 0
        self.updates = 0
        self.saved_snapshots: set[int] = set()
        self.training_seconds = 0.0
        self.last_metrics: dict[str, float] = {}
        self.pool_hash = pool_signature(pool)
        self._write_config()

    def _write_config(self) -> None:
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(asdict(self.config)), encoding="utf-8"
        )

    def normalized(self, observations: FloatArray) -> Tensor:
        values = self.obs_stats.normalize(
            observations, self.config.observation_clip, self.config.normalization_epsilon
        )
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _policy_payload(self) -> dict[str, Any]:
        return {
            "format": "ppo.v1",
            "versions": VERSIONS,
            "config": asdict(self.config),
            "model": {
                name: value.detach().cpu() for name, value in self.model.state_dict().items()
            },
            "obs_stats": self.obs_stats.state_dict(),
            "return_stats": self.return_stats.state_dict(),
            "global_step": self.global_step,
            "updates": self.updates,
            "stage": self.stage,
            "training_seconds": self.training_seconds,
        }

    def _save_file(self, data: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, temporary)
        temporary.replace(path)

    def snapshot(self, requested_step: int | str) -> None:
        data = self._policy_payload()
        data["requested_step"] = requested_step
        self._save_file(data, self.run_dir / "snapshots" / f"{requested_step}.pt")

    def checkpoint(self, path: Path) -> None:
        data = self._policy_payload()
        data.update(
            {
                "optimizer": self.optimizer.state_dict(),
                "saved_snapshots": sorted(self.saved_snapshots),
                "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "mps_rng": torch.mps.get_rng_state() if self.device.type == "mps" else None,
                "env_rng": self.env.np_random.bit_generator.state,
                "env_arrays": {
                    name: torch.as_tensor(getattr(self.env, name).copy()) for name in ENV_ARRAYS
                },
                "observations": torch.as_tensor(self.observations.copy()),
                "discounted_returns": torch.as_tensor(self.discounted_returns.copy()),
                "pool_signature": self.pool_hash,
                "sim_config": {
                    "physics": asdict(self.env.physics),
                    "tracks": asdict(self.env.tracks),
                    "observation": asdict(self.env.builder.config),
                    "reward": asdict(self.env.reward_config),
                    "env": asdict(self.env.config),
                },
            }
        )
        self._save_file(data, path)

    def restore(self, path: Path) -> None:
        data = load_checkpoint(path)
        if "optimizer" not in data:
            raise ValueError("Policy snapshots are not resumable training checkpoints")
        if data["pool_signature"] != self.pool_hash:
            raise ValueError("Checkpoint track geometry/order does not match this training pool")
        current = asdict(self.config)
        prior = data["config"].copy()
        for name in ("device", "cpu_threads", "total_timesteps"):
            current.pop(name)
            prior.pop(name)
        if current != prior or self.config.total_timesteps < data["config"]["total_timesteps"]:
            raise ValueError(
                "Resume settings must match, except device/threads or an extended horizon"
            )
        expected_sim = {
            "physics": asdict(self.env.physics),
            "tracks": asdict(self.env.tracks),
            "observation": asdict(self.env.builder.config),
            "reward": asdict(self.env.reward_config),
            "env": asdict(self.env.config),
        }
        if data["sim_config"] != expected_sim:
            raise ValueError("Simulation config differs from checkpoint")
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.obs_stats.load_state_dict(data["obs_stats"])
        self.return_stats.load_state_dict(data["return_stats"])
        self.global_step, self.updates = data["global_step"], data["updates"]
        self.stage, self.training_seconds = data["stage"], data["training_seconds"]
        self.saved_snapshots = set(data["saved_snapshots"])
        for name in ENV_ARRAYS:
            array = data["env_arrays"][name].cpu().numpy()
            if array.shape != getattr(self.env, name).shape:
                raise ValueError(f"Checkpoint environment shape mismatch: {name}")
            getattr(self.env, name)[:] = array
        self.env.np_random.bit_generator.state = data["env_rng"]
        self.observations = data["observations"].cpu().numpy().copy()
        self.discounted_returns = data["discounted_returns"].cpu().numpy().copy()
        self.rng.bit_generator.state = data["rng"]
        torch.set_rng_state(data["torch_rng"])
        if self.device.type == "cuda" and data["cuda_rng"]:
            torch.cuda.set_rng_state_all(data["cuda_rng"])
        if self.device.type == "mps" and data["mps_rng"] is not None:
            torch.mps.set_rng_state(data["mps_rng"])

    def _due_snapshots(self) -> None:
        for threshold in self.config.snapshot_steps:
            if self.global_step >= threshold and threshold not in self.saved_snapshots:
                self.snapshot(threshold)
                self.saved_snapshots.add(threshold)

    @torch.no_grad()
    def collect(self) -> tuple[dict[str, Tensor], dict[str, float]]:
        c = self.config
        observations, actions, log_probs, values = [], [], [], []
        rewards, terminated, truncated, terminal_values = [], [], [], []
        episode_returns, episode_laps, crashes, lap_times = [], [], [], []
        contributions = np.zeros(len(REWARD_NAMES))
        self._due_snapshots()
        for _ in range(c.rollout_steps):
            normalized = self.normalized(self.observations)
            distribution = self.model.distribution(normalized)
            sampled = distribution.sample()
            observations.append(normalized)
            actions.append(sampled)
            log_probs.append(distribution.log_prob(sampled).sum(-1))
            values.append(self.model.value(normalized))
            self.observations, raw_reward, term, trunc, info = self.env.step(
                sampled.cpu().numpy().clip(-1, 1)
            )
            self.obs_stats.update(self.observations)
            done = term | trunc
            self.discounted_returns = self.discounted_returns * c.gamma + raw_reward
            self.return_stats.update(self.discounted_returns)
            scaled = raw_reward / np.sqrt(self.return_stats.var + c.normalization_epsilon)
            self.discounted_returns[done] = 0.0
            rewards.append(
                torch.as_tensor(
                    scaled.clip(-c.reward_clip, c.reward_clip),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            terminated.append(torch.as_tensor(term, device=self.device))
            truncated.append(torch.as_tensor(trunc, device=self.device))
            bootstrap = torch.zeros(c.num_envs, device=self.device)
            if np.any(trunc):
                bootstrap[torch.as_tensor(trunc, device=self.device)] = self.model.value(
                    self.normalized(info["final_obs"][trunc])
                )
            terminal_values.append(bootstrap)
            for name_index, name in enumerate(REWARD_NAMES):
                contributions[name_index] += info["reward_terms"][name].sum()
            if np.any(done):
                episode = info["episode"]
                episode_returns.extend(episode["r"][done].tolist())
                episode_laps.extend(episode["laps"][done].tolist())
                crashes.extend(episode["crashed"][done].tolist())
                lap_times.extend(
                    episode["best_lap_time"][done & (episode["best_lap_time"] > 0)].tolist()
                )
                if self.writer is not None:
                    for row in np.flatnonzero(done):
                        self.writer.add_scalar(
                            "episodes/return", episode["r"][row], self.global_step
                        )
                        self.writer.add_scalar(
                            "episodes/laps", episode["laps"][row], self.global_step
                        )
            self.global_step += c.num_envs
            self._due_snapshots()
        value_tensor = torch.stack(values)
        trunc_tensor, term_tensor = torch.stack(truncated), torch.stack(terminated)
        last_value = self.model.value(self.normalized(self.observations))
        next_values = torch.cat((value_tensor[1:], last_value[None]), dim=0)
        next_values = torch.where(trunc_tensor, torch.stack(terminal_values), next_values)
        advantages, returns = compute_gae(
            torch.stack(rewards),
            value_tensor,
            next_values,
            term_tensor,
            trunc_tensor,
            c.gamma,
            c.gae_lambda,
        )
        batch = {
            "observations": torch.stack(observations).flatten(0, 1),
            "actions": torch.stack(actions).flatten(0, 1),
            "log_probs": torch.stack(log_probs).flatten(),
            "values": value_tensor.flatten(),
            "advantages": advantages.flatten(),
            "returns": returns.flatten(),
        }
        metrics = {
            "rollout/episodic_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "rollout/laps_completed": float(np.mean(episode_laps)) if episode_laps else 0.0,
            "rollout/crash_rate": float(np.mean(crashes)) if crashes else 0.0,
            "rollout/best_lap_time": min(lap_times, default=0.0),
            "rollout/episodes": float(len(episode_returns)),
            "actions/clipped_fraction": float((batch["actions"].abs() > 1).float().mean()),
        }
        for index, name in enumerate(REWARD_NAMES):
            metrics[f"rewards/{name}"] = float(
                contributions[index] / (c.num_envs * c.rollout_steps)
            )
        for index in range(OBS_SIZE):
            column = batch["observations"][:, index]
            metrics[f"observations/{index}/min"] = float(column.min())
            metrics[f"observations/{index}/max"] = float(column.max())
            metrics[f"observations/{index}/mean"] = float(column.mean())
        return batch, metrics

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        c = self.config
        size = len(batch["returns"])
        minibatch_size = size // c.num_minibatches
        totals = np.zeros(6)
        batches = 0
        for _ in range(c.update_epochs):
            order = self.rng.permutation(size)
            for start in range(0, size, minibatch_size):
                indices = torch.as_tensor(order[start : start + minibatch_size], device=self.device)
                new_log_prob, entropy, new_value = self.model.evaluate(
                    batch["observations"][indices], batch["actions"][indices]
                )
                log_ratio = new_log_prob - batch["log_probs"][indices]
                ratio = log_ratio.exp()
                advantages = batch["advantages"][indices]
                if c.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + c.normalization_epsilon
                    )
                policy_loss = torch.maximum(
                    -advantages * ratio, -advantages * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)
                ).mean()
                value_error = (new_value - batch["returns"][indices]) ** 2
                if c.clip_value_loss:
                    clipped = batch["values"][indices] + (
                        new_value - batch["values"][indices]
                    ).clamp(-c.value_clip_coef, c.value_clip_coef)
                    value_error = torch.maximum(
                        value_error, (clipped - batch["returns"][indices]) ** 2
                    )
                value_loss, entropy_mean = 0.5 * value_error.mean(), entropy.mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), c.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > c.clip_coef).float().mean()
                totals += np.array(
                    [
                        policy_loss.item(),
                        value_loss.item(),
                        entropy_mean.item(),
                        kl.item(),
                        clip_fraction.item(),
                        grad_norm.item(),
                    ]
                )
                batches += 1
            if c.target_kl is not None and kl.item() > c.target_kl:
                break
        variance = batch["returns"].var(unbiased=False)
        explained = 1.0 - (batch["returns"] - batch["values"]).var(unbiased=False) / (
            variance + c.normalization_epsilon
        )
        metrics = dict(
            zip(
                (
                    "loss/policy",
                    "loss/value",
                    "policy/entropy",
                    "policy/approx_kl",
                    "policy/clip_fraction",
                    "policy/grad_norm",
                ),
                (totals / batches).tolist(),
                strict=True,
            )
        )
        metrics["critic/explained_variance"] = float(explained)
        metrics["policy/steer_log_std"] = self.model.log_std[0].item()
        metrics["policy/throttle_log_std"] = self.model.log_std[1].item()
        return metrics

    def train_update(self) -> dict[str, float]:
        started = perf_counter()
        if self.config.anneal_lr:
            fraction = max(0.0, 1.0 - self.global_step / self.config.total_timesteps)
            self.optimizer.param_groups[0]["lr"] = fraction * self.config.learning_rate
        batch, metrics = self.collect()
        metrics.update(self.update(batch))
        elapsed = perf_counter() - started
        self.training_seconds += elapsed
        self.updates += 1
        metrics["time/steps_per_second"] = (
            self.config.num_envs * self.config.rollout_steps / elapsed
        )
        metrics["time/training_seconds"] = self.training_seconds
        metrics["optimizer/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.last_metrics = metrics
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, self.global_step)
            self.writer.flush()
        record = {"step": self.global_step, "update": self.updates, "stage": self.stage, **metrics}
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "step",
                        "update",
                        "stage",
                        "rollout/episodic_return",
                        "rollout/laps_completed",
                        "policy/approx_kl",
                        "critic/explained_variance",
                        "time/steps_per_second",
                    )
                }
            ),
            flush=True,
        )
        if self.updates % self.config.checkpoint_every_updates == 0:
            self.checkpoint(self.run_dir / "checkpoints" / f"{self.global_step}.pt")
            self.checkpoint(self.run_dir / "latest.pt")
        return metrics

    def switch_to_pool(self, pool: TrackPool) -> None:
        self.env.close()
        self.env = VecRaceEnv(self.config.num_envs, pool)
        self.observations, _ = self.env.reset(seed=self.config.seed)
        self.obs_stats.update(self.observations)
        self.discounted_returns[:] = 0.0
        self.stage = "pool"
        self.pool_hash = pool_signature(pool)

    def close(self) -> None:
        self.env.close()
        if self.writer is not None:
            self.writer.close()


def load_checkpoint(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    data = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if data.get("format") != "ppo.v1" or data.get("versions") != VERSIONS:
        raise ValueError("Unsupported checkpoint format or simulation/observation versions")
    data["source_sha256"] = hashlib.sha256(contents).hexdigest()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/training/default"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument(
        "--updates", type=int, help="Stop after this many additional updates (resumable)"
    )
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = PPOConfig(**checkpoint["config"]) if checkpoint else load_ppo(args.config)
    if args.device:
        config = replace(config, device=args.device)
    if args.total_timesteps is not None:
        config = replace(config, total_timesteps=args.total_timesteps)
    if args.updates is not None and args.updates <= 0:
        parser.error("--updates must be positive")
    if not args.resume and (args.run_dir / "latest.pt").exists():
        parser.error("Run already exists; use --resume or choose a new --run-dir")
    stage = checkpoint["stage"] if checkpoint else "single"
    pool = make_track_pool("training") if stage == "pool" else single_pool(config.single_showcase)
    trainer = Trainer(config, pool, args.run_dir, stage=stage)
    if args.resume:
        trainer.restore(args.resume)
    else:
        trainer.checkpoint(args.run_dir / "latest.pt")
    print(f"device={trainer.device} stage={stage} run_dir={args.run_dir}", flush=True)
    first_update = trainer.updates
    try:
        while trainer.global_step < config.total_timesteps:
            if args.updates is not None and trainer.updates - first_update >= args.updates:
                break
            trainer.train_update()
            if trainer.stage == "single" and trainer.updates % config.eval_every_updates == 0:
                from eval.evaluate import evaluate_policy

                # Evaluation sampling must not perturb the training RNG stream.
                with torch.random.fork_rng(
                    devices=[trainer.device.index or 0] if trainer.device.type == "cuda" else []
                ):
                    result = evaluate_policy(
                        trainer.model,
                        trainer.obs_stats,
                        trainer.env.pool,
                        config,
                        runs=config.curriculum_eval_runs,
                        seed=config.seed,
                        deterministic=False,
                    )
                rate = result["summary"]["lap_completion_rate"]
                print(f"curriculum success={rate:.1%} step={trainer.global_step}", flush=True)
                (args.run_dir / "curriculum.json").write_text(json.dumps(result, indent=2) + "\n")
                if (
                    rate >= config.curriculum_success_rate
                    and trainer.global_step >= config.curriculum_min_steps
                ):
                    trainer.checkpoint(args.run_dir / "single-solved.pt")
                    trainer.switch_to_pool(make_track_pool("training"))
                    trainer.checkpoint(args.run_dir / "latest.pt")
        trainer.checkpoint(args.run_dir / "latest.pt")
        if trainer.global_step >= config.total_timesteps:
            trainer.snapshot("final")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

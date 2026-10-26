from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from eval.evaluate import evaluate_policy
from sim.config import EnvConfig
from sim.observation import OBS_SIZE
from sim.pool import TrackPool
from train.config import PPOConfig, load_ppo
from train.ppo import (
    ActorCritic,
    RunningMeanStd,
    Trainer,
    choose_device,
    compute_gae,
    load_checkpoint,
)


def small_config() -> PPOConfig:
    return replace(
        load_ppo(),
        device="cpu",
        num_envs=4,
        rollout_steps=8,
        update_epochs=2,
        num_minibatches=2,
        hidden_size=32,
        total_timesteps=96,
        snapshot_steps=(0, 31, 64),
        checkpoint_every_updates=1,
    )


def test_gae_matches_hand_computed_values_and_cuts_resets() -> None:
    # gamma=.9, lambda=.8. Env 0 truncates at t=1 and resets at t=2;
    # env 1 terminates at t=1 (so its final value of 99 must be ignored).
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]])
    values = torch.tensor([[0.5, 0.5], [1.0, 1.0], [3.0, 3.0]])
    next_values = torch.tensor([[1.0, 1.0], [4.0, 99.0], [5.0, 5.0]])
    terminated = torch.tensor([[False, False], [False, True], [False, False]])
    truncated = torch.tensor([[False, False], [True, False], [False, False]])
    advantage, returns = compute_gae(rewards, values, next_values, terminated, truncated, 0.9, 0.8)
    # deltas at t1: 2+.9*4-1=4.6 vs 2-1=1. No t2 leakage into t1.
    expected = torch.tensor([[1.4 + 0.72 * 4.6, 1.4 + 0.72], [4.6, 1.0], [101.5, 101.5]])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected + values)


def test_running_stats_matches_pooled_samples_and_roundtrips() -> None:
    rng = np.random.default_rng(321)
    samples = rng.normal(2, 3, (100, OBS_SIZE))
    first = RunningMeanStd((OBS_SIZE,), 0.0001)
    whole = RunningMeanStd((OBS_SIZE,), 0.0001)
    first.update(samples[:40])
    first.update(samples[40:])
    whole.update(samples)
    np.testing.assert_allclose(first.mean, whole.mean, atol=1e-12)
    np.testing.assert_allclose(first.var, whole.var, atol=1e-12)
    restored = RunningMeanStd((OBS_SIZE,), 0.0001)
    restored.load_state_dict(first.state_dict())
    np.testing.assert_array_equal(restored.mean, first.mean)
    np.testing.assert_array_equal(restored.var, first.var)
    assert restored.count == first.count
    with pytest.raises(FloatingPointError):
        restored.update(np.full((1, OBS_SIZE), np.nan))


def test_architecture_initialization_and_gaussian_log_prob() -> None:
    config = load_ppo()
    torch.set_num_threads(1)
    model = ActorCritic(config)
    assert model.actor[0].out_features == model.actor[2].out_features == 256
    assert model.critic[0].out_features == model.critic[2].out_features == 256
    assert set(map(id, model.actor.parameters())).isdisjoint(map(id, model.critic.parameters()))
    observations = torch.zeros((8, OBS_SIZE))
    actions = torch.tensor([[2.0, -3.0]]).expand(8, 2)
    probabilities, entropy, values = model.evaluate(observations, actions)
    # Log likelihood is evaluated on the original Gaussian samples, not clipped env actions.
    reference = torch.distributions.Normal(torch.zeros_like(actions), torch.ones_like(actions))
    torch.testing.assert_close(probabilities, reference.log_prob(actions).sum(-1))
    assert values.shape == entropy.shape == (8,)
    torch.testing.assert_close(
        model.actor[-1].weight @ model.actor[-1].weight.T, torch.eye(2) * 0.01**2
    )


def test_ppo_smoke_finite_losses_snapshots_and_tensorboard(pool: TrackPool, tmp_path: Path) -> None:
    config = small_config()
    trainer = Trainer(config, pool, tmp_path)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    try:
        for _ in range(3):
            metrics = trainer.train_update()
            assert all(np.isfinite(value) for value in metrics.values())
            assert metrics["policy/approx_kl"] >= -1e-7
            assert 0 <= metrics["policy/clip_fraction"] <= 1
        trainer.snapshot("final")
        assert any(
            not torch.equal(before[name], value)
            for name, value in trainer.model.state_dict().items()
        )
        assert trainer.obs_stats.count > config.num_envs
        assert trainer.return_stats.count > config.num_envs
        for requested in (0, 31, 64):
            snapshot = load_checkpoint(tmp_path / "snapshots" / f"{requested}.pt")
            assert snapshot["requested_step"] == requested
            assert requested <= snapshot["global_step"] < requested + config.num_envs
            assert "obs_stats" in snapshot and "return_stats" in snapshot
    finally:
        trainer.close()
    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    tags = events.Tags()["scalars"]
    for tag in (
        "loss/policy",
        "loss/value",
        "policy/entropy",
        "policy/approx_kl",
        "policy/clip_fraction",
        "critic/explained_variance",
        "time/steps_per_second",
        "rewards/progress",
        "rewards/time",
        "rewards/crash",
        "rewards/lap",
    ):
        assert tag in tags


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_resume_matches_uninterrupted_training(
    pool: TrackPool, tmp_path: Path, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = replace(small_config(), device=device)
    uninterrupted = Trainer(config, pool, tmp_path / "full", tensorboard=False)
    uninterrupted.env.config = EnvConfig(2, 3)
    for _ in range(3):
        uninterrupted.train_update()
    expected = {name: value.clone() for name, value in uninterrupted.model.state_dict().items()}
    expected_states = uninterrupted.env.states.copy()
    expected_obs = uninterrupted.obs_stats.state_dict()
    expected_return = uninterrupted.return_stats.state_dict()
    uninterrupted.close()
    partial = Trainer(config, pool, tmp_path / "partial", tensorboard=False)
    partial.env.config = EnvConfig(2, 3)
    partial.train_update()
    checkpoint = tmp_path / "partial/latest.pt"
    partial.checkpoint(checkpoint)
    partial.close()
    resumed = Trainer(config, pool, tmp_path / "resumed", tensorboard=False)
    resumed.env.config = EnvConfig(2, 3)
    try:
        resumed.restore(checkpoint)
        resumed.train_update()
        resumed.train_update()
        assert resumed.global_step == 96 and resumed.updates == 3
        for name, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        np.testing.assert_array_equal(resumed.env.states, expected_states)
        for name in ("mean", "var"):
            torch.testing.assert_close(
                resumed.obs_stats.state_dict()[name], expected_obs[name], rtol=0, atol=0
            )
            torch.testing.assert_close(
                resumed.return_stats.state_dict()[name], expected_return[name], rtol=0, atol=0
            )
    finally:
        resumed.close()


def test_rollout_bootstraps_from_terminal_observation_at_truncation(
    pool: TrackPool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(small_config(), pool, tmp_path, tensorboard=False)
    trainer.env.config = EnvConfig(2, 1)  # Every transition truncates and autoresets.
    monkeypatch.setattr(trainer.obs_stats, "normalize", lambda values, clip, epsilon: values)
    monkeypatch.setattr(trainer.return_stats, "update", lambda samples: None)
    monkeypatch.setattr(trainer.model, "value", lambda observation: 2 + 10 * observation[:, 9])
    with torch.no_grad():
        trainer.model.actor[-1].weight.zero_()
        trainer.model.actor[-1].bias[:] = torch.tensor([0.0, 1.0])
        trainer.model.log_std.fill_(-20)
    expected = []
    original_step = trainer.env.step

    def capture(actions: np.ndarray) -> tuple:
        result = original_step(actions)
        _, reward, _, truncated, info = result
        assert truncated.all()
        value = 2 + 10 * info["final_obs"][:, 9]
        assert (value > 2).all()  # Reset observations would incorrectly give value=2.
        scaled = reward / np.sqrt(1 + trainer.config.normalization_epsilon)
        expected.extend((scaled + trainer.config.gamma * value).tolist())
        return result

    monkeypatch.setattr(trainer.env, "step", capture)
    try:
        batch, _ = trainer.collect()
        np.testing.assert_allclose(batch["returns"].cpu().numpy(), expected, atol=1e-6)
    finally:
        trainer.close()


def test_evaluation_freezes_normalizer_and_does_not_change_training_rng(
    pool: TrackPool, tmp_path: Path
) -> None:
    trainer = Trainer(small_config(), pool, tmp_path, tensorboard=False)
    before = trainer.obs_stats.state_dict()
    rng = torch.get_rng_state().clone()
    # Force a short evaluation with reverse throttle, which consistently leaves the course.
    with torch.no_grad():
        trainer.model.actor[-1].bias[:] = torch.tensor([1.0, 1.0])
    result = evaluate_policy(
        trainer.model, trainer.obs_stats, pool, trainer.config, runs=2, seed=42, deterministic=True
    )
    assert result["summary"]["episodes"] == 2
    assert result["summary"]["crash_rate"] == 1
    assert trainer.obs_stats.count == before["count"]
    torch.testing.assert_close(
        trainer.obs_stats.state_dict()["mean"], before["mean"], rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    trainer.close()


def test_device_selection_and_invalid_config() -> None:
    assert choose_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        replace(load_ppo(), gamma=1.5)
    with pytest.raises(ValueError):
        replace(load_ppo(), num_minibatches=3)

from dataclasses import astuple, replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sim.config import EnvConfig, PhysicsConfig, TrackConfig, load_env, load_pool_config
from sim.env import RaceEnv
from sim.physics import CarState
from sim.pool import TrackPool, make_track_pool
from sim.progress import project
from sim.track import Track, build_track
from sim.vec_env import VecRaceEnv


def test_gymnasium_checker(pool: TrackPool) -> None:
    env = RaceEnv(pool)
    check_env(env, skip_render_check=True)
    env.close()


def test_reward_terms_progress_backwards_and_tick_rate(pool: TrackPool) -> None:
    env = RaceEnv(pool)
    env.reset(seed=4)
    env.session.state = replace(env.session.state, vx=10)
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert not terminated and not truncated
    assert info["physics_ticks"] == 2
    assert info["delta_progress"] > 0
    terms = info["reward_terms"]
    assert terms["progress"] == pytest.approx(0.1 * info["delta_progress"])
    assert terms["time"] == -0.001
    assert terms["crash"] == terms["lap"] == 0
    assert reward == pytest.approx(sum(terms.values()))
    env.reset(seed=4)
    env.session.state = replace(env.session.state, vx=-3)
    _, reward, _, _, info = env.step(np.zeros(2))
    assert info["wrong_way"]
    assert info["reward_terms"]["progress"] < 0 and reward < 0


def test_crash_reward_and_termination_on_first_tick(pool: TrackPool) -> None:
    env = RaceEnv(pool)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))
    env.reset(seed=4)
    env.session.state = replace(env.session.state, x=80)
    observation, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated
    assert observation.shape == (22,)
    assert info["physics_ticks"] == 1
    assert info["reward_terms"]["crash"] == -10
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert info["episode"]["r"] == reward
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))


def test_step_limit_is_truncation_and_preserves_terminal_observation(pool: TrackPool) -> None:
    env = RaceEnv(pool, config=EnvConfig(max_laps=2, max_agent_steps=1))
    env.reset(seed=4)
    observation, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0]))
    assert truncated and not terminated
    assert observation[9] > 0
    assert info["episode"]["l"] == 1
    assert info["episode"]["reward_terms"] == info["reward_terms"]
    assert info["episode"]["r"] == reward


def position_near_finish(env: RaceEnv, lap: int) -> None:
    track = env.session.track
    arc = track.length - 0.1
    point, heading = track.sample(arc)
    env.session.state = CarState(*point, heading, vx=12)
    tracker = env.session.progress
    tracker.unwrapped = lap * track.length - 0.1
    tracker.projection = project(track, *point, None, env.tracks.projection_window)
    tracker.next_checkpoint = lap * len(track.checkpoints)
    tracker.laps = lap - 1
    tracker.lap_started_at = env.session.time - 20


def test_lap_bonus_and_two_lap_limit_are_separate_from_crashes(pool: TrackPool) -> None:
    env = RaceEnv(pool)
    env.reset(seed=4)
    position_near_finish(env, 1)
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert not terminated and not truncated
    assert info["laps"] == 1
    assert info["reward_terms"]["lap"] == 5
    assert info["best_lap_time"] > 20
    position_near_finish(env, 2)
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert not terminated and truncated
    assert info["laps"] == 2 and info["reward_terms"]["lap"] == 5


def test_batched_reset_and_spaces(pool: TrackPool) -> None:
    env = VecRaceEnv(64, pool)
    observation, info = env.reset(seed=20)
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (64, 2)
    assert env.single_observation_space.shape == (22,)
    assert observation.dtype == np.float64
    assert info["track_index"].shape == (64,)
    assert not hasattr(env, "envs")  # This is not N single environments in a wrapper.


def test_scalar_and_native_batch_match_through_crashes_resets_and_timeouts(
    circle: Track, physics: PhysicsConfig, tracks: TrackConfig
) -> None:
    pool = TrackPool([circle, build_track(circle.centerline * [1.3, 0.9], 10, 25)])
    config = replace(load_env(), max_agent_steps=151)
    batch = VecRaceEnv(4, pool, physics=physics, tracks=tracks, config=config)
    scalar = [RaceEnv(pool, physics=physics, tracks=tracks, config=config) for _ in range(4)]
    observation, _ = batch.reset(seed=20, options={"track_indices": [0, 1, 0, 1]})
    for index, env in enumerate(scalar):
        initial, _ = env.reset(seed=20, options={"track_index": index % 2})
        np.testing.assert_array_equal(observation[index], initial)
    rng = np.random.default_rng(192)
    crash_count, timeout_count = 0, 0
    for tick in range(600):
        if tick % 8 == 0:
            actions = rng.uniform(-1, 1, (4, 2))
            actions[:, 1] = 0.9
        observation, rewards, terminated, truncated, infos = batch.step(actions)
        for index, env in enumerate(scalar):
            expected, reward, term, trunc, info = env.step(actions[index])
            assert terminated[index] == term and truncated[index] == trunc
            assert rewards[index] == pytest.approx(reward, abs=1e-9)
            assert infos["physics_ticks"][index] == info["physics_ticks"]
            np.testing.assert_allclose(infos["progress"][index], info["progress"], atol=1e-9)
            final = infos["final_obs"][index] if term or trunc else observation[index]
            np.testing.assert_allclose(final, expected, atol=1e-8, rtol=0)
            for name, value in info["reward_terms"].items():
                assert infos["reward_terms"][name][index] == pytest.approx(value, abs=1e-9)
            if term or trunc:
                crash_count += term
                timeout_count += trunc
                assert infos["_final_obs"][index]
                assert infos["final_info"]["episode"]["l"][index] == info["episode"]["l"]
                expected, _ = env.reset(
                    options={"track_index": int(infos["reset_track_index"][index])}
                )
                np.testing.assert_allclose(observation[index], expected, atol=1e-10)
            np.testing.assert_allclose(batch.states[index], astuple(env.session.state), atol=1e-9)
    assert crash_count > 0 and timeout_count > 0


def test_vector_autoreset_only_resets_finished_rows_and_keeps_snapshot(pool: TrackPool) -> None:
    env = VecRaceEnv(3, pool)
    env.reset(seed=42)
    env.states[1, 0] = 80
    observation, reward, terminated, truncated, info = env.step(np.ones((3, 2)))
    np.testing.assert_array_equal(terminated, [False, True, False])
    assert not truncated.any()
    assert env.ticks.tolist() == [2, 0, 2]
    assert reward[1] < -9
    assert info["final_obs"][1, 19] == 1
    assert observation[1, 19] == 0
    assert info["episode"]["l"][1] == 1
    snapshot = observation.copy(), info["progress"].copy(), info["final_obs"].copy()
    env.step(np.ones((3, 2)))
    for expected, previous in zip(
        snapshot, (observation, info["progress"], info["final_obs"]), strict=True
    ):
        np.testing.assert_array_equal(expected, previous)


def test_vector_lap_rewards_timing_and_final_lap_truncation(pool: TrackPool) -> None:
    scalar = RaceEnv(pool)
    scalar.reset(seed=42)
    position_near_finish(scalar, 2)
    tracker = scalar.session.progress
    batch = VecRaceEnv(2, pool)
    batch.reset(seed=42)
    batch.states[0] = astuple(scalar.session.state)
    batch.progress_meters[0] = tracker.unwrapped
    batch.indices[0] = tracker.projection.index
    batch.arcs[0] = tracker.projection.arc
    batch.offsets[0] = tracker.projection.offset
    batch.next_checkpoint[0] = tracker.next_checkpoint
    batch.laps[0] = tracker.laps
    batch.lap_started_at[0] = tracker.lap_started_at
    expected, reward, term, trunc, info = scalar.step(np.zeros(2))
    observation, rewards, terminated, truncated, infos = batch.step(np.zeros((2, 2)))
    assert term == terminated[0] == False  # noqa: E712
    assert trunc == truncated[0] == True  # noqa: E712
    assert not terminated[1] and not truncated[1]
    assert infos["reward_terms"]["lap"].tolist() == [5, 0]
    assert infos["physics_ticks"][0] == info["physics_ticks"] == 1
    assert infos["best_lap_time"][0] == pytest.approx(info["best_lap_time"], abs=1e-12)
    assert rewards[0] == pytest.approx(reward, abs=1e-9)
    np.testing.assert_allclose(infos["final_obs"][0], expected, atol=1e-8)
    assert observation[0, 9] == 0 and batch.laps[0] == 0


def test_seeded_autoresets_choose_tracks_reproducibly(circle: Track) -> None:
    pool = TrackPool([circle, build_track(circle.centerline * 1.5, 12, 25)])
    config = EnvConfig(2, 2)
    a, b = VecRaceEnv(8, pool, config=config), VecRaceEnv(8, pool, config=config)
    x, _ = a.reset(seed=42)
    y, _ = b.reset(seed=42)
    np.testing.assert_array_equal(x, y)
    selected = set(a.track_ids.tolist())
    for _ in range(20):
        x, r1, t1, tr1, _ = a.step(np.ones((8, 2)))
        y, r2, t2, tr2, _ = b.step(np.ones((8, 2)))
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(a.track_ids, b.track_ids)
        selected.update(a.track_ids.tolist())
    assert selected == {0, 1}


@pytest.mark.parametrize("action", [np.zeros(3), np.array([np.nan, 1]), np.array([np.inf, 1])])
def test_invalid_actions_fail_before_mutating_state(pool: TrackPool, action: np.ndarray) -> None:
    env = RaceEnv(pool)
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.session.ticks == 0
    batch = VecRaceEnv(2, pool)
    batch.reset(seed=42)
    with pytest.raises(ValueError):
        batch.step(np.broadcast_to(action, (2, len(action))))
    assert not batch.ticks.any()


def test_pool_splits_and_held_out_seeds_are_disjoint() -> None:
    config = load_pool_config()
    training = range(config.training_seed_start, config.training_seed_start + config.training_count)
    held_out = range(config.held_out_seed_start, config.held_out_seed_start + config.held_out_count)
    assert len(training) == 200 and len(held_out) == 50
    assert set(training).isdisjoint(held_out)
    pool = make_track_pool("held_out", count=2)
    assert [track.seed for track in pool.tracks] == list(held_out)[:2]

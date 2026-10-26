from dataclasses import replace

import pytest

from sim.config import (
    EnvConfig,
    load_env,
    load_observation,
    load_pool_config,
    load_reward,
)


def test_defaults_match_observation_reward_and_episode_spec() -> None:
    observation = load_observation()
    assert observation.ray_count == 9 and observation.ray_half_angle_degrees == 120
    assert observation.ray_max_distance == 60
    assert observation.lookahead_meters == (10, 25, 50, 80)
    assert load_env().max_laps == 2
    assert load_reward().progress == 0.1
    assert load_reward().time == -0.001
    assert load_reward().crash == -10
    assert load_reward().lap == 5


def test_invalid_dimensions_scales_limits_and_pool_overlap_fail_fast() -> None:
    for values in (
        {"ray_count": 8},
        {"ray_count": 9.0},
        {"yaw_rate_scale": 0},
        {"grid_cell_size": float("nan")},
        {"lookahead_meters": (10, -25, 50, 80)},
    ):
        with pytest.raises(ValueError):
            replace(load_observation(), **values)
    with pytest.raises(ValueError):
        EnvConfig(2, 0)
    with pytest.raises(ValueError):
        EnvConfig(2, 1.5)
    with pytest.raises(ValueError):
        replace(load_reward(), progress=float("inf"))
    with pytest.raises(ValueError):
        replace(load_pool_config(), held_out_seed_start=10050)

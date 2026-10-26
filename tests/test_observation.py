from dataclasses import astuple, replace

import numpy as np
import pytest

from sim import OBS_VERSION
from sim.config import load_observation
from sim.env import RaceEnv
from sim.observation import ObservationBuilder
from sim.physics import CarState
from sim.pool import TrackPool
from sim.raycast import BoundaryGrid
from sim.track import Track, build_track


def brute_ray(track: Track, origin: np.ndarray, direction: np.ndarray, maximum: float) -> float:
    """Independent full-boundary oracle: deliberately does not use the spatial grid."""
    start = np.vstack((track.left, track.right))
    end = np.vstack((np.roll(track.left, -1, axis=0), np.roll(track.right, -1, axis=0)))
    vector = end - start
    relative = start - origin
    denominator = direction[0] * vector[:, 1] - direction[1] * vector[:, 0]
    valid = np.abs(denominator) > 1e-10
    numerator = relative[:, 0] * vector[:, 1] - relative[:, 1] * vector[:, 0]
    distances = numerator[valid] / denominator[valid]
    along = (relative[valid, 0] * direction[1] - relative[valid, 1] * direction[0]) / denominator[
        valid
    ]
    hits = distances[(distances >= -1e-10) & (along >= -1e-10) & (along <= 1 + 1e-10)]
    return min(maximum, max(0, float(hits.min()))) if len(hits) else maximum


@pytest.mark.parametrize("cell_size", [2.0, 8.0, 23.0])
def test_spatial_rays_match_full_boundary_oracle(circle: Track, cell_size: float) -> None:
    # Include reversed geometry, different track widths and origins outside the road.
    other = build_track(circle.centerline[::-1] * [1.4, 0.8], 10, 25)
    tracks = [circle, other]
    grid = BoundaryGrid(tracks, cell_size)
    rng = np.random.default_rng(45)
    ids = np.arange(50) % 2
    origins = []
    for index in ids:
        point, heading = tracks[index].sample(rng.uniform(0, tracks[index].length))
        point += rng.uniform(-18, 18) * np.array([-np.sin(heading), np.cos(heading)])
        origins.append(point)
    origins = np.array(origins)
    angles = rng.uniform(-np.pi, np.pi, (50, 9))
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    distances = grid.cast(ids, origins, directions, 60)
    expected = np.array(
        [
            [brute_ray(tracks[index], origin, direction, 60) for direction in rays]
            for index, origin, rays in zip(ids, origins, directions, strict=True)
        ]
    )
    np.testing.assert_allclose(distances, expected, rtol=0, atol=1e-8)


def test_rays_enter_grid_from_outside_and_hit_vertices(circle: Track) -> None:
    grid = BoundaryGrid([circle], 8)
    result = grid.cast(
        np.array([0, 0]),
        np.array([[300.0, 0], [60.0, 0]]),
        np.array([[[-1.0, 0], [1.0, 0]], [[1.0, 0], [-1.0, 0]]]),
        500,
    )
    np.testing.assert_allclose(result, [[234, 500], [6, 6]], atol=1e-8)


def test_collinear_ray_and_boundary_origin_return_zero(circle: Track) -> None:
    grid = BoundaryGrid([circle], 8)
    origin = circle.left[0]
    vector = circle.left[1] - origin
    direction = vector / np.linalg.norm(vector)
    distances = grid.cast(np.array([0]), origin[None], direction[None, None], 60)
    assert distances[0, 0] == pytest.approx(0, abs=1e-8)


def test_exact_observation_order_normalization_and_action_history(pool: TrackPool) -> None:
    config = load_observation()
    builder = ObservationBuilder(pool, config)
    state = CarState(x=60, psi=np.pi / 2, vx=20, vy=-4, r=0.3, steering=0.4)
    observation = builder.build(
        np.array([astuple(state)]),
        np.array([0]),
        np.array([0]),
        np.array([0.0]),
        np.array([3.0]),
        np.array([[0.8, -0.5]]),
    )[0]
    assert OBS_VERSION == "obs.v1"
    assert observation.shape == (22,) and observation.dtype == np.float64
    assert np.all((observation[:9] >= 0) & (observation[:9] <= 1))
    np.testing.assert_allclose(observation[9:15], [0.25, -0.1, 0.1, 0, 1, 0.5], atol=1e-12)
    np.testing.assert_allclose(observation[15:19], [1 / 6] * 4, atol=1e-4)
    np.testing.assert_allclose(observation[19:], [0.8, -0.5, 0.4])
    # Across +/-90 degrees, the two road edges are six meters away.
    np.testing.assert_allclose(observation[[1, 7]], [0.1, 0.1], atol=1e-8)


def test_previous_action_and_steering_reset_and_clipping(pool: TrackPool) -> None:
    env = RaceEnv(pool)
    reset, _ = env.reset(seed=4)
    np.testing.assert_array_equal(reset[19:], 0)
    observation, _, _, _, _ = env.step(np.array([4.0, -3.0]))
    np.testing.assert_array_equal(observation[19:21], [1, -1])
    assert observation[21] == pytest.approx(2 * env.physics.steer_rate * env.physics.dt)
    env.session.state = replace(env.session.state, vx=1000, vy=-1000, r=100)
    clipped = env._observation()
    np.testing.assert_array_equal(clipped[9:12], [1, -1, 1])
    reset, _ = env.reset(seed=4)
    np.testing.assert_array_equal(reset[19:], 0)


def test_lookahead_wrap_and_signed_curvature(circle: Track) -> None:
    clockwise = build_track(circle.centerline[::-1], 12, 25)
    pool = TrackPool([circle, clockwise])
    samples = pool.curvature_at(np.array([0, 1]), np.array([[0, circle.length + 4], [-4, 10]]))
    np.testing.assert_allclose(samples[0], 1 / 60, atol=1e-5)
    np.testing.assert_allclose(samples[1], -1 / 60, atol=1e-5)

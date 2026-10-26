import numpy as np
import pytest

from sim.config import PhysicsConfig, TrackConfig, load_physics, load_tracks
from sim.track import Track, build_track


@pytest.fixture
def physics() -> PhysicsConfig:
    return load_physics()


@pytest.fixture
def tracks() -> TrackConfig:
    return load_tracks()[0]


@pytest.fixture
def circle() -> Track:
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    return build_track(np.column_stack((60 * np.cos(angles), 60 * np.sin(angles))), 12, 25)

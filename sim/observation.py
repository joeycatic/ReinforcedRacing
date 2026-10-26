"""obs.v1, 22 float64 values. Exact order is also specified in OBSERVATION.md.

0:9 rays (-120 to +120 degrees); 9 vx; 10 vy; 11 yaw rate;
12 sin(tangent-heading); 13 cos(tangent-heading); 14 lateral offset;
15:19 curvature at 10/25/50/80m; 19:21 previous steer/throttle;
21 rate-limited steering actuator. Normalized values are clipped to [-1,1].
"""

import numpy as np

from sim.config import ObservationConfig
from sim.geometry import FloatArray
from sim.pool import TrackPool
from sim.raycast import IntArray

OBS_SIZE = 22


class ObservationBuilder:
    def __init__(self, pool: TrackPool, config: ObservationConfig) -> None:
        self.pool = pool
        self.config = config
        half_angle = np.deg2rad(config.ray_half_angle_degrees)
        self.ray_angles = np.linspace(-half_angle, half_angle, config.ray_count)
        self.lookahead = np.asarray(config.lookahead_meters)

    def build(
        self,
        states: FloatArray,
        track_ids: IntArray,
        indices: IntArray,
        arcs: FloatArray,
        offsets: FloatArray,
        previous_actions: FloatArray,
    ) -> FloatArray:
        c = self.config
        observations = np.empty((len(states), OBS_SIZE), dtype=np.float64)
        psi = states[:, 2]
        angles = psi[:, None] + self.ray_angles
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        observations[:, :9] = (
            self.pool.grid.cast(
                track_ids,
                states[:, :2],
                directions,
                c.ray_max_distance,
                self.pool.hidden_tags(track_ids, arcs),
            )
            / c.ray_max_distance
        )
        observations[:, 9] = states[:, 3] / c.longitudinal_speed_scale
        observations[:, 10] = states[:, 4] / c.lateral_speed_scale
        observations[:, 11] = states[:, 5] / c.yaw_rate_scale
        tangent = self.pool.tangents[self.pool.offsets[track_ids] + indices]
        sine, cosine = np.sin(psi), np.cos(psi)
        observations[:, 12] = tangent[:, 1] * cosine - tangent[:, 0] * sine
        observations[:, 13] = tangent[:, 0] * cosine + tangent[:, 1] * sine
        observations[:, 14] = offsets / (self.pool.widths[track_ids] / 2.0)
        observations[:, 15:19] = (
            self.pool.curvature_at(track_ids, arcs[:, None] + self.lookahead) / c.curvature_scale
        )
        observations[:, 19:21] = previous_actions
        observations[:, 21] = states[:, 6]
        return np.clip(observations, -1.0, 1.0)

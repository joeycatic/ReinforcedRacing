import type { Action, CarState, ObservationConfig } from "./types";
import { Track, type Projection } from "./track";
import { BoundaryGrid } from "./raycast";
import { clamp } from "./physics";

export const OBS_SIZE = 22;
export class ObservationBuilder {
  readonly grid: BoundaryGrid;
  readonly rayAngles: number[];
  constructor(
    readonly track: Track,
    readonly config: ObservationConfig,
  ) {
    if (config.ray_count !== 9 || config.lookahead_meters.length !== 4)
      throw new Error("obs.v1 requires 9 rays and 4 lookaheads");
    this.grid = new BoundaryGrid(track, config.grid_cell_size);
    const halfAngle = config.ray_half_angle_degrees * (Math.PI / 180),
      increment = (2 * halfAngle) / (config.ray_count - 1);
    this.rayAngles = Array.from({ length: config.ray_count }, (_, i) =>
      i === config.ray_count - 1 ? halfAngle : -halfAngle + i * increment,
    );
  }
  build(s: CarState, p: Projection, previousAction: Action): number[] {
    const c = this.config;
    const rays = this.rayAngles.map(
      (angle) =>
        this.grid.cast(
          [s.x, s.y],
          [Math.cos(s.psi + angle), Math.sin(s.psi + angle)],
          c.ray_max_distance,
        ) / c.ray_max_distance,
    );
    const tangent = this.track.data.tangents[p.index],
      sine = Math.sin(s.psi),
      cosine = Math.cos(s.psi);
    return [
      ...rays,
      s.vx / c.longitudinal_speed_scale,
      s.vy / c.lateral_speed_scale,
      s.r / c.yaw_rate_scale,
      tangent[1] * cosine - tangent[0] * sine,
      tangent[0] * cosine + tangent[1] * sine,
      p.offset / (this.track.data.width / 2),
      ...c.lookahead_meters.map(
        (a) => this.track.curvatureAt(p.arc + a) / c.curvature_scale,
      ),
      ...previousAction,
      s.steering,
    ].map((v) => clamp(v, -1, 1));
  }
}

// Configuration mirrors sim/config.py; all simulation values are float64 JS numbers.
export interface PhysicsConfig {
  dt: number;
  action_repeat: number;
  mass: number;
  inertia: number;
  front_axle: number;
  rear_axle: number;
  car_length: number;
  car_width: number;
  gravity: number;
  front_stiffness: number;
  rear_stiffness: number;
  front_mu: number;
  rear_mu: number;
  max_steer: number;
  steer_speed_falloff: number;
  steer_rate: number;
  slip_speed_floor: number;
  engine_force: number;
  engine_speed_falloff: number;
  brake_force: number;
  drag: number;
  rolling_resistance: number;
  reverse_force: number;
  reverse_threshold: number;
  max_reverse_speed: number;
  respawn_delay: number;
}
export interface TrackConfig {
  min_control_points: number;
  max_control_points: number;
  radius: number;
  radius_noise: number;
  angle_jitter: number;
  aspect_ratio: number;
  min_width: number;
  max_width: number;
  spline_samples_per_span: number;
  sample_spacing: number;
  checkpoint_spacing: number;
  max_curvature: number;
  max_attempts: number;
  projection_window: number;
  wrong_way_min_delta: number;
}
export interface ObservationConfig {
  ray_count: number;
  ray_half_angle_degrees: number;
  ray_max_distance: number;
  grid_cell_size: number;
  longitudinal_speed_scale: number;
  lateral_speed_scale: number;
  yaw_rate_scale: number;
  curvature_scale: number;
  lookahead_meters: number[];
}
export type Vec2 = [number, number];
export type Action = [number, number];
export interface CarState {
  x: number;
  y: number;
  psi: number;
  vx: number;
  vy: number;
  r: number;
  steering: number;
}
export interface TrackData {
  version: string;
  id: string;
  name: string;
  seed: number;
  width: number;
  centerline: Vec2[];
  arc_length: number[];
  tangents: Vec2[];
  curvature: number[];
  left: Vec2[];
  right: Vec2[];
  checkpoints: number[];
}
export const stateValues = (s: CarState): number[] => [
  s.x,
  s.y,
  s.psi,
  s.vx,
  s.vy,
  s.r,
  s.steering,
];

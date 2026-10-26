import type { Action, CarState, PhysicsConfig } from "./types";

export const clamp = (value: number, low: number, high: number): number =>
  Math.max(low, Math.min(value, high));
export const speed = (s: CarState): number => Math.hypot(s.vx, s.vy);
export function step(s: CarState, action: Action, c: PhysicsConfig): CarState {
  if (!action.every(Number.isFinite)) throw new Error("Actions must be finite");
  const requested = clamp(action[0], -1, 1);
  const steering =
    s.steering +
    clamp(requested - s.steering, -c.steer_rate * c.dt, c.steer_rate * c.dt);
  const angle =
    (steering * c.max_steer) / (1 + speed(s) / c.steer_speed_falloff);
  const frontLateral = s.vy + c.front_axle * s.r;
  const frontWheelX = s.vx * Math.cos(angle) + frontLateral * Math.sin(angle);
  const frontWheelY = frontLateral * Math.cos(angle) - s.vx * Math.sin(angle);
  const frontSlip = Math.atan2(
    frontWheelY,
    Math.max(Math.abs(frontWheelX), c.slip_speed_floor),
  );
  const rearSlip = Math.atan2(
    s.vy - c.rear_axle * s.r,
    Math.max(Math.abs(s.vx), c.slip_speed_floor),
  );
  const wheelbase = c.front_axle + c.rear_axle;
  const frontLimit =
    (c.front_mu * c.mass * c.gravity * c.rear_axle) / wheelbase;
  const rearLimit = (c.rear_mu * c.mass * c.gravity * c.front_axle) / wheelbase;
  const front = clamp(-c.front_stiffness * frontSlip, -frontLimit, frontLimit);
  const rear = clamp(-c.rear_stiffness * rearSlip, -rearLimit, rearLimit);
  const throttle = clamp(action[1], -1, 1);
  const braking = throttle < 0 && s.vx > c.reverse_threshold;
  const drive =
    throttle >= 0
      ? (throttle * c.engine_force) /
        (1 + Math.max(s.vx, 0) / c.engine_speed_falloff)
      : braking
        ? throttle * c.brake_force
        : throttle * c.reverse_force;
  let resistance = c.drag * speed(s) * s.vx;
  if (s.vx !== 0) resistance += Math.sign(s.vx) * c.rolling_resistance;
  const forceX = drive - resistance - front * Math.sin(angle);
  const forceY = front * Math.cos(angle) + rear;
  let vx = s.vx + (forceX / c.mass + s.r * s.vy) * c.dt;
  const vy = s.vy + (forceY / c.mass - s.r * s.vx) * c.dt;
  const r =
    s.r +
    ((c.front_axle * front * Math.cos(angle) - c.rear_axle * rear) /
      c.inertia) *
      c.dt;
  if (s.vx > 0 && vx < 0 && (braking || throttle === 0)) vx = 0;
  else if (s.vx < 0 && vx > 0 && throttle === 0) vx = 0;
  vx = Math.max(vx, -c.max_reverse_speed);
  const psi = Math.atan2(
    Math.sin(s.psi + r * c.dt),
    Math.cos(s.psi + r * c.dt),
  );
  return {
    x: s.x + (vx * Math.cos(psi) - vy * Math.sin(psi)) * c.dt,
    y: s.y + (vx * Math.sin(psi) + vy * Math.cos(psi)) * c.dt,
    psi,
    vx,
    vy,
    r,
    steering,
  };
}

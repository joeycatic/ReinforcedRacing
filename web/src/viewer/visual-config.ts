/** Presentation-only limits; none of these values enter physics or inference. */
export const visualConfig = {
  drift: {
    slipThreshold: 3,
    maxMarkPairs: 1200,
    maxParticles: 64,
    particleLifetime: 0.45,
    particleVelocity: 1.5,
    markWidth: 0.25,
  },
  ghost: { minimumPixelsPerMeter: 3.2, labelSize: 11 },
  criticValueScale: 10,
} as const;

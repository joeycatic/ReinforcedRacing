# M1 decisions

- Treat the existing lowercase `spec.md` as the master spec; `SPEC.md` in AGENTS.md refers to that file, so no duplicate specification is created.
- Stop at M1; defer the physics-parity skill's TypeScript port, golden state/observation fixtures, Vitest, and manifest checks to M4 as explicitly scheduled by the user's spec; Python is the versioned source of truth until then.
- Use Python floats and NumPy float64 in meters/radians with x-right/y-up and positive steering to the left, to give the future browser port a single clear coordinate convention.
- Store the rate-limited normalized steering input in `CarState` in addition to the six rigid-body fields, because actuator history is required for deterministic continuation.
- Regularize tire slip using a configurable minimum wheel longitudinal speed instead of a second kinematic model, to keep one continuous force model stable at rest and in reverse.
- Use static axle loads and independent lateral friction clamps, without load transfer or a friction-circle model, to keep M1's bicycle model simple; slightly lower rear friction permits recoverable oversteer.
- Negative throttle brakes while forward velocity exceeds the configured near-stop threshold, then engages speed-limited reverse; Space uses this same braking input because a separate handbrake is not required in M1.
- Interpret the spec's wall rule literally: contact occurs when the car center's distance from the road centerline exceeds `(track_width + car_width) / 2`, allowing a half-car-width excursion beyond the edge.
- Generate polar control points with uniform closed Catmull–Rom interpolation; use 1 m arc samples and a shorter closing segment, omitting duplicate endpoints to avoid degenerate segments.
- Validate centerline/boundary intersections (including touching and collinear overlap), local road folding, and curvature limited by both track config and the car's minimum steering radius; retries consume the same seeded RNG in a bounded loop.
- Permit per-showcase radius/noise/aspect overrides alongside fixed seeds, to produce broad, elongated, and compact routes with the same generator.
- Keep checkpoint ordinals monotonic and unwrapped; interpolate finish crossing time within its physics tick, and never award progress for respawn or start-line oscillation.
- Start the first lap timer at session creation/restart, include crash delays and manual respawns in lap time, and preserve completed laps on respawn; countdowns belong to M5.
- Use an accumulator driven by wall time only in pygame; simulation time is always integer physics ticks, with inputs held for two ticks; cap catch-up after a long UI stall to avoid a spiral of death.
- Provide `make tracks` for M1 JSON geometry export; required Make targets for future milestones fail explicitly instead of pretending to implement unavailable training, evaluation, model export, or web features.
- Use NumPy for geometry, PyYAML for typed config loading, pygame for manual driving, pytest for behavior checks, and Ruff for lint/format; pin resolved versions in uv.lock and use Python 3.12 by default (source supports 3.11+).

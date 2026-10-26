# obs.v1

Both environments return **22 float64 values** per car, in the order below.
`RaceEnv` has shape `(22,)`; `VecRaceEnv` has shape `(N, 22)`.
All components are clipped to `[-1, 1]` after normalization; ray distances lie in `[0, 1]`.
Scales, ray settings, and lookaheads are typed in `ObservationConfig` and stored in
`configs/observation.yaml`. Altering their meaning requires a new observation version
and makes earlier policy checkpoints incompatible.

| Zero-based index | Field | Normalization / convention |
| --- | --- | --- |
| 0 | Boundary ray at −120° | distance / 60 m |
| 1 | Boundary ray at −90° | distance / 60 m |
| 2 | Boundary ray at −60° | distance / 60 m |
| 3 | Boundary ray at −30° | distance / 60 m |
| 4 | Boundary ray at 0° (forward) | distance / 60 m |
| 5 | Boundary ray at +30° | distance / 60 m |
| 6 | Boundary ray at +60° | distance / 60 m |
| 7 | Boundary ray at +90° | distance / 60 m |
| 8 | Boundary ray at +120° | distance / 60 m |
| 9 | Body longitudinal velocity `vx` | / 80 m/s; negative in reverse |
| 10 | Body lateral velocity `vy` | / 40 m/s; positive to the car's left |
| 11 | Yaw rate `r` | / 3 rad/s; positive counterclockwise |
| 12 | Sine of heading error | `sin(track_tangent_heading − psi)` |
| 13 | Cosine of heading error | `cos(track_tangent_heading − psi)` |
| 14 | Signed lateral centerline offset | / (track width / 2); positive left of track direction |
| 15 | Signed centerline curvature 10 m ahead | / 0.1 m⁻¹ |
| 16 | Signed centerline curvature 25 m ahead | / 0.1 m⁻¹ |
| 17 | Signed centerline curvature 50 m ahead | / 0.1 m⁻¹ |
| 18 | Signed centerline curvature 80 m ahead | / 0.1 m⁻¹ |
| 19 | Previous requested steering input | clipped to `[-1, 1]` |
| 20 | Previous requested throttle input | clipped to `[-1, 1]` |
| 21 | Current steering actuator input | `CarState.steering`, rate-limited normalized input |

The specification's enumerated fields total **21**, despite specifying 22 dimensions.
Index 21 resolves that ambiguity with the existing steering actuator state, which is
needed to observe steering lag. The first 21 components retain the spec's listed order.
This decision is recorded in `DECISIONS.md`.

Positive ray angles rotate left/counterclockwise relative to car heading in the y-up
world. Rays intersect the **actual exported left/right boundary polylines**, select
the nearest nonnegative intersection, and return 60 m if no intersection is in range.
The uniform grid contains all boundary segments overlapping each cell; exact DDA cell
traversal avoids skipping narrow corner crossings. Parallel/collinear contacts and
origins outside the grid are handled. Grid resolution changes speed, not sensor meaning.

The heading tangent is the track's stored vertex tangent at the current projected
segment index, matching its direction of increasing arc length. Curvature is linearly
interpolated between centerline nodes at wrapped lookahead arc lengths, including the
closing segment. Positive curvature bends left; negative curvature bends right.

The previous action is the action used to reach this observation, held over up to two
physics ticks. On reset the previous action, velocity, yaw rate, and actuator are zero.
There is no observation RNG, running normalization, or reward information in this builder.

## Terminal observations and resets

`RaceEnv.step()` returns the terminal observation and requires an explicit `reset()`.
`VecRaceEnv` uses **same-step autoreset**: finished rows in the returned observation
already describe the new episode, with zero previous action and actuator. The full
terminal-observation batch is saved in `info["final_obs"]`; use only rows selected by
`info["_final_obs"]`. The `final_info` dictionary and `_final_info` mask preserve the
finished transitions' metrics.

For a time/lap truncation, value bootstrapping must use the terminal observation, not
the reset observation. For a crash termination there is no bootstrap. The next policy
action uses the returned reset observation. Reward terms and root info metrics always
describe the transition just taken, including for rows that reset. `reset_track_index`
identifies the newly selected pool track whenever any row resets.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { Track } from "../src/sim/track";
import { ProgressTracker } from "../src/sim/progress";
import { DriveSession } from "../src/sim/session";
import { step, clamp } from "../src/sim/physics";
import { ObservationBuilder } from "../src/sim/observation";
import { requireVersions } from "../src/assets/versions";
import type {
  CarState,
  Action,
  TrackData,
  PhysicsConfig,
  TrackConfig,
  ObservationConfig,
} from "../src/sim/types";

interface Progress {
  projection: { index: number; arc: number; offset: number; distance: number };
  unwrapped: number;
  delta: number;
  wrong_way: boolean;
  laps: number;
  lap_times: number[];
  lap_started_at: number;
  last_checkpoint: number;
  next_checkpoint: number;
}
interface Fixture {
  crossing_probes?: {
    state: CarState;
    index: number;
    arc: number;
    offset: number;
    previous_action: Action;
    observation: number[];
  }[];
  seed: number;
  versions: Record<string, string>;
  track: TrackData;
  initial_state: CarState;
  config: {
    physics: PhysicsConfig;
    tracks: TrackConfig;
    observation: ObservationConfig;
  };
  steps: {
    action: Action;
    event: string | null;
    state: CarState;
    observation: number[];
    progress: Progress;
    session: {
      state: CarState;
      progress: Progress;
      observation: number[];
      ticks: number;
      respawn_ticks: number;
      crashed: boolean;
      crashes: number;
      terminated: boolean;
      time: number;
    };
  }[];
  progress_probes: { position: [number, number]; progress: Progress }[];
  ray_probes: {
    origin: [number, number];
    direction: [number, number];
    distance: number;
  }[];
}
const directory = fileURLToPath(
  new URL("../../tests/fixtures/parity/", import.meta.url),
);
let maxError = 0;
function compare(actual: unknown, expected: unknown, label: string): void {
  if (typeof expected === "number") {
    const error = Math.abs((actual as number) - expected);
    maxError = Math.max(maxError, error);
    if (!Number.isFinite(error) || error > 1e-6)
      throw new Error(
        `${label}: actual=${actual}, expected=${expected}, error=${error}`,
      );
  } else if (expected && typeof expected === "object") {
    if (Array.isArray(expected)) {
      expect(Array.isArray(actual), label).toBe(true);
      expect((actual as unknown[]).length, label + ".length").toBe(
        expected.length,
      );
    }
    for (const key of Object.keys(expected))
      compare(
        (actual as Record<string, unknown>)[key],
        (expected as Record<string, unknown>)[key],
        `${label}.${key}`,
      );
  } else expect(actual, label).toEqual(expected);
}
for (const dir of [
  directory,
  fileURLToPath(new URL("../../tests/fixtures/crossing/", import.meta.url)),
])
  for (const name of readdirSync(dir).filter((n) => n.endsWith(".json")))
    describe(name, () => {
      const f = JSON.parse(readFileSync(dir + "/" + name, "utf8")) as Fixture;
      it("matches state, observations, progress, and play sessions over 600 physics ticks", () => {
        requireVersions(f.versions);
        expect(f.steps.length).toBe(600);
        const track = new Track(f.track),
          tracker = new ProgressTracker(track, f.config.tracks);
        tracker.respawn();
        const session = new DriveSession(
          track,
          f.config.physics,
          f.config.tracks,
        );
        const builder = new ObservationBuilder(track, f.config.observation);
        let state = f.initial_state;
        f.steps.forEach((record, i) => {
          state = step(state, record.action, f.config.physics);
          tracker.update(
            record.state.x,
            record.state.y,
            (i + 1) * f.config.physics.dt,
            f.config.physics.dt,
          );
          const previous: Action = [
            clamp(record.action[0], -1, 1),
            clamp(record.action[1], -1, 1),
          ];
          compare(state, record.state, `tick ${i + 1}.state`);
          // Stress trajectories deliberately continue far beyond a crash. Test the
          // projection kernel with identical input coordinates: tiny libm differences
          // can select either segment at an off-road shared-vertex distance tie.
          compare(tracker, record.progress, `tick ${i + 1}.progress`);
          compare(
            builder.build(state, tracker.projection, previous),
            record.observation,
            `tick ${i + 1}.observation`,
          );
          if (record.event === "respawn") session.respawn();
          if (record.event === "restart") session.restart();
          session.tick(record.action);
          compare(
            {
              ...session,
              time: session.time,
              observation: builder.build(
                session.state,
                session.progress.projection,
                previous,
              ),
            },
            record.session,
            `tick ${i + 1}.session`,
          );
          compare(
            builder.build(session.state, session.progress.projection, previous),
            record.session.observation,
            `tick ${i + 1}.session.observation`,
          );
        });
        console.log(`${name}: maximum absolute error ${maxError}`);
      });
      it("matches ordered checkpoints, reverse, and interpolated lap times", () => {
        const tracker = new ProgressTracker(
          new Track(f.track),
          f.config.tracks,
        );
        tracker.respawn();
        f.progress_probes.forEach((probe, i) => {
          tracker.update(
            ...probe.position,
            (i + 1) * f.config.physics.dt,
            f.config.physics.dt,
          );
          compare(tracker, probe.progress, `probe ${i + 1}`);
        });
        expect(tracker.laps).toBeGreaterThanOrEqual(2);
      });
      if (f.crossing_probes)
        it("matches observations approaching both crossing branches", () => {
          const builder = new ObservationBuilder(
            new Track(f.track),
            f.config.observation,
          );
          expect(f.crossing_probes).toHaveLength(40);
          for (const probe of f.crossing_probes!)
            compare(
              builder.build(
                probe.state,
                {
                  index: probe.index,
                  arc: probe.arc,
                  offset: probe.offset,
                  distance: 0,
                },
                probe.previous_action,
              ),
              probe.observation,
              "crossing probe",
            );
        });
      it("matches grid traversal at boundaries, parallel edges and outside origins", () => {
        const builder = new ObservationBuilder(
          new Track(f.track),
          f.config.observation,
        );
        f.ray_probes.forEach((probe, i) =>
          compare(
            builder.grid.cast(probe.origin, probe.direction, 100),
            probe.distance,
            `ray ${i}`,
          ),
        );
      });
    });

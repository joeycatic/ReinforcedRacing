"""Generate Python truth; never edit fixtures to conceal a TypeScript divergence."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval.artifacts import simulation_config, write_json
from export.assets import VERSIONS
from sim.config import ROOT, load_observation, load_physics, load_tracks
from sim.observation import ObservationBuilder
from sim.physics import Action, CarState, step
from sim.pool import TrackPool, make_track_pool
from sim.progress import ProgressTracker
from sim.session import DriveSession


def progress_values(progress: ProgressTracker) -> dict:
    return {
        "projection": asdict(progress.projection),
        "unwrapped": progress.unwrapped,
        "delta": progress.delta,
        "wrong_way": progress.wrong_way,
        "laps": progress.laps,
        "lap_times": list(progress.lap_times),
        "lap_started_at": progress.lap_started_at,
        "last_checkpoint": progress.last_checkpoint,
        "next_checkpoint": progress.next_checkpoint,
    }


def observe(
    s: CarState, p: ProgressTracker, builder: ObservationBuilder, action: Action
) -> list[float]:
    return builder.build(
        np.array([list(asdict(s).values())]),
        np.array([0]),
        np.array([p.projection.index]),
        np.array([p.projection.arc]),
        np.array([p.projection.offset]),
        np.array([[np.clip(action.steer, -1, 1), action.throttle]]),
    )[0].tolist()


def fixtures() -> dict[str, dict]:
    physics, tracks = load_physics(), load_tracks()[0]
    result = {}
    for number, track in enumerate(make_track_pool("showcase").tracks):
        pool = TrackPool([track])
        builder = ObservationBuilder(pool, load_observation())
        rng = np.random.default_rng(track.seed)
        tracker = ProgressTracker(track, tracks)
        x, y, psi = tracker.respawn()
        state = CarState(x, y, psi, 27.0 if number else -3.0, 5.0, 0.3, -0.2)
        session = DriveSession(track, physics, tracks)
        steps = []
        for tick in range(600):
            if tick % 2 == 0:
                action = Action(
                    float(rng.uniform(-1.4, 1.4)),
                    1.0 if tick < 180 or tick >= 440 else -1.0 if tick < 360 else 0.0,
                )
            state = step(state, action, physics)
            tracker.update(state.x, state.y, (tick + 1) * physics.dt, physics.dt)
            event = "respawn" if tick == 315 else "restart" if tick == 480 else None
            if event == "respawn":
                session.respawn()
            if event == "restart":
                session.restart()
            session.tick(action)

            steps.append(
                {
                    "action": [action.steer, action.throttle],
                    "event": event,
                    "state": asdict(state),
                    "observation": observe(state, tracker, builder, action),
                    "progress": progress_values(tracker),
                    "session": {
                        "state": asdict(session.state),
                        "progress": progress_values(session.progress),
                        "observation": observe(session.state, session.progress, builder, action),
                        "ticks": session.ticks,
                        "respawn_ticks": session.respawn_ticks,
                        "crashed": session.crashed,
                        "crashes": session.crashes,
                        "terminated": session.terminated,
                        "time": session.time,
                    },
                }
            )
        # Separate positions exercise ordered checkpoints, lap wrapping, reverse, respawn.
        tracker = ProgressTracker(track, tracks)
        tracker.respawn()
        probes = []
        for tick in range(600):
            arc = track.length * (tick + 1) / 240
            if 260 <= tick < 275:
                arc = track.length * (520 - tick) / 240
            position, _ = track.sample(arc)
            tracker.update(*position, (tick + 1) * physics.dt, physics.dt)
            probes.append({"position": position.tolist(), "progress": progress_values(tracker)})
        # DDA edge cases: collinear/on-edge, near-axis, outside-grid origins.
        ray_probes = []
        for boundary in (track.left, track.right):
            for index in (0, len(boundary) // 2):
                a, b = boundary[index], boundary[(index + 1) % len(boundary)]
                direction = (b - a) / np.linalg.norm(b - a)
                for origin, vector in (
                    (a, direction),
                    (a, -direction),
                    (a, np.array([1.0, 1e-12])),
                    (a + 80 * direction, -direction),
                    (a, np.array([0.0, 1.0])),
                ):
                    distance = pool.grid.cast(
                        np.array([0]), origin[None, :], vector[None, None, :], 100.0
                    )[0, 0]
                    ray_probes.append(
                        {
                            "origin": origin.tolist(),
                            "direction": vector.tolist(),
                            "distance": float(distance),
                        }
                    )
        result[f"{track.id}.json"] = {
            "seed": track.seed,
            "versions": VERSIONS,
            "config": simulation_config(),
            "track": track.to_dict(),
            "initial_state": asdict(CarState(x, y, psi, 27.0 if number else -3.0, 5.0, 0.3, -0.2)),
            "steps": steps,
            "progress_probes": probes,
            "ray_probes": ray_probes,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/parity")
    args = parser.parse_args()
    for name, fixture in fixtures().items():
        write_json(args.output / name, fixture)
        print(args.output / name)


if __name__ == "__main__":
    main()

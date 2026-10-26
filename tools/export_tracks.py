"""Export M1 tracks only; model/replay/manifest export belongs to M4."""

import argparse
from pathlib import Path

from sim.config import load_physics, load_tracks
from sim.track import generate_track, showcase_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument(
        "--seed", type=int, help="Export one generated seed instead of the showcases"
    )
    args = parser.parse_args()
    physics = load_physics()
    config, showcases = load_tracks()
    tracks = (
        [generate_track(args.seed, config, physics)]
        if args.seed is not None
        else [showcase_track(item, config, physics) for item in showcases]
    )
    for track in tracks:
        path = args.output / f"{track.id}.track.v1.json"
        track.save(path)
        print(f"{path}: {track.length:.1f} m, {track.width:.1f} m wide")


if __name__ == "__main__":
    main()

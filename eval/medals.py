"""Select a deterministic medal ladder from retained training checkpoints."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from sim.config import ROOT


@dataclass(frozen=True)
class MedalChoice:
    medal: str
    step: int
    lap_times: dict[str, float | None]


def select_medals(
    candidates: dict[int, dict[str, float | None]],
    author: dict[str, float],
    gap: float = 0.03,
) -> list[MedalChoice]:
    previous: dict[str, float | None] = dict(author)
    remaining = dict(candidates)
    choices = []
    for medal in ("gold", "silver", "bronze"):
        eligible = {
            step: laps
            for step, laps in remaining.items()
            if any(laps.get(t) is not None for t in author)
            and all(
                laps.get(t) is None or previous.get(t) is None or laps[t] >= previous[t] * (1 + gap)
                for t in author
            )
        }
        if not eligible:
            break
        full = {
            s: laps for s, laps in eligible.items() if all(laps.get(t) is not None for t in author)
        }
        step = min(
            full or eligible,
            key=lambda s: (
                -sum(eligible[s].get(t) is not None for t in author),
                sum(v for v in eligible[s].values() if v is not None),
                s,
            ),
        )
        laps = {t: eligible[step].get(t) for t in author}
        # A missing lap must not forget the closest faster medal on that track.
        previous.update({t: lap for t, lap in laps.items() if lap is not None})
        choices.append(MedalChoice(medal, step, laps))
        del remaining[step]
    return list(reversed(choices))


def load_medals(path: Path = ROOT / "configs/medals.yaml") -> list[MedalChoice]:
    config = yaml.safe_load(path.read_text())
    evidence = json.loads((ROOT / "experiments/medal-selection.json").read_text())
    return [
        MedalChoice(m["medal"], m["step"], evidence["candidates"][str(m["step"])]["lap_times"])
        for m in config["medals"]
    ]

"""Select visible, monotonically improving demonstrations of learning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from sim.config import ROOT


@dataclass(frozen=True)
class Run:
    step: int
    source: str
    outcome: Literal["finished", "crashed", "stalled"]
    progress: float
    lap_time: float | None


def score(run: Run) -> tuple[int, float]:
    if run.outcome == "finished":
        if run.lap_time is None:
            raise ValueError("A finished run requires a lap time")
        return (1, -run.lap_time)
    return (0, run.progress)


def select_attempts(runs: list[Run], count: int = 10) -> list[Run]:
    if not runs:
        return []
    if count < 3:
        raise ValueError("At least three slots are required for milestones")
    final = runs[-1]
    pool = [run for run in runs[:-1] if score(run) < score(final)]
    if not pool:
        return [final]
    chain = [pool[0]]
    for run in pool[1:]:
        last = chain[-1]
        visible = (
            run.outcome == "finished"
            and (last.outcome != "finished" or run.lap_time <= last.lap_time * 0.99)
        ) or (
            run.outcome != "finished"
            and last.outcome != "finished"
            and run.progress >= max(last.progress * 1.10, last.progress + 20.0)
        )
        if visible and score(run) > score(last):
            chain.append(run)
    chain.append(final)
    if len(chain) > count:
        first_finish = next((i for i, r in enumerate(chain) if r.outcome == "finished"), None)
        keep = {0, len(chain) - 1}
        if first_finish is not None:
            keep.add(first_finish)
        targets = [round(float(t)) for t in np.linspace(0, len(chain) - 1, count)]
        for target in targets:
            if target in keep:
                continue
            if len(keep) == count:
                break
            index = min(
                (i for i in range(len(chain)) if i not in keep), key=lambda i: (abs(i - target), i)
            )
            keep.add(index)
        chain = [chain[i] for i in sorted(keep)]
    return chain


@dataclass(frozen=True)
class AttemptChoice:
    step: int
    source: str
    headline: str | None


def load_attempts(path: Path = ROOT / "configs/attempts.yaml") -> dict[str, list[AttemptChoice]]:
    document = yaml.safe_load(path.read_text())
    return {
        track: [AttemptChoice(**choice) for choice in choices]
        for track, choices in document["tracks"].items()
    }

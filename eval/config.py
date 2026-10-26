"""Typed evaluation/export settings; simulation values stay in sim.config."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

import yaml

from sim.config import ROOT


@dataclass(frozen=True)
class ExportConfig:
    run_dir: str
    evaluation_dir: str
    assets_dir: str
    seed: int
    stochastic_runs: int
    held_out_runs: int
    heatmap_cell_size: float
    heatmap_margin: float
    position_scale: int
    heading_scale: int
    speed_scale: int
    action_scale: int
    onnx_opset: int
    onnx_validation_samples: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and (
                type(value) is not int or value < (0 if field.name == "seed" else 1)
            ):
                raise ValueError(f"Invalid {field.name}")
            if field.type is float and (not isfinite(value) or value <= 0):
                raise ValueError(f"Invalid {field.name}")
            if field.type is str and not value:
                raise ValueError(f"Empty {field.name}")
        if self.onnx_validation_samples < 3:
            raise ValueError("Use at least three ONNX validation observations")
        if self.onnx_opset < 17:
            raise ValueError("Use ONNX opset >=17")


def load_export(path: Path = ROOT / "configs/export.yaml") -> ExportConfig:
    return ExportConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))

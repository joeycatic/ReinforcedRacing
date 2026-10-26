"""Policy graph consumes raw float64 observations and includes normalization."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from eval.config import ExportConfig
from sim.observation import OBS_SIZE
from train.config import PPOConfig
from train.ppo import ActorCritic, RunningMeanStd

ONNX_TOLERANCE = 1e-5


class PolicyGraph(nn.Module):
    def __init__(self, checkpoint: dict) -> None:
        super().__init__()
        config = PPOConfig(**checkpoint["config"])
        self.model = ActorCritic(config)
        self.model.load_state_dict(checkpoint["model"])
        self.register_buffer("mean", checkpoint["obs_stats"]["mean"].double())
        self.register_buffer(
            "std", (checkpoint["obs_stats"]["var"].double() + config.normalization_epsilon).sqrt()
        )
        self.clip = config.observation_clip

    def forward(self, raw_observations: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (
            ((raw_observations - self.mean) / self.std).clamp(-self.clip, self.clip).float()
        )
        return self.model.actor(normalized), self.model.value(normalized)


def export_policy(
    checkpoint: dict, path: Path, config: ExportConfig, observations: np.ndarray
) -> dict:
    ppo = PPOConfig(**checkpoint["config"])
    torch.set_num_threads(ppo.cpu_threads)
    graph = PolicyGraph(checkpoint).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph,
        (torch.zeros((1, OBS_SIZE), dtype=torch.float64),),
        str(path),
        input_names=["observations"],
        output_names=["action", "value"],
        dynamic_axes={"observations": {0: "batch"}, "action": {0: "batch"}, "value": {0: "batch"}},
        opset_version=config.onnx_opset,
        dynamo=False,
        external_data=False,
    )
    model = onnx.load(path)
    for key, value in {
        "versions": checkpoint["versions"],
        "snapshot_step": checkpoint["global_step"],
        "checkpoint_sha256": checkpoint["source_sha256"],
        "input": "raw obs.v1 float64; normalization embedded",
    }.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, json.dumps(value)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    values = np.asarray(observations, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != OBS_SIZE
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("ONNX validation requires finite nonempty (N,22) observations")
    stats = RunningMeanStd((OBS_SIZE,), ppo.initial_stats_count)
    stats.load_state_dict(checkpoint["obs_stats"])
    action_error = value_error = 0.0
    batch_sizes = sorted({1, min(17, len(values)), len(values)})
    for size in batch_sizes:
        batch = values[:size]
        with torch.no_grad():
            normalized = stats.normalize(batch, ppo.observation_clip, ppo.normalization_epsilon)
            tensor = torch.from_numpy(normalized).float()
            expected_action = graph.model.actor(tensor).numpy()
            expected_value = graph.model.value(tensor).numpy()
        actual_action, actual_value = session.run(None, {"observations": batch})
        if not all(
            np.isfinite(output).all()
            for output in (actual_action, actual_value, expected_action, expected_value)
        ):
            raise ValueError("Non-finite ONNX or PyTorch output")
        action_error = max(action_error, float(np.max(np.abs(actual_action - expected_action))))
        value_error = max(value_error, float(np.max(np.abs(actual_value - expected_value))))
    if (
        not np.isfinite([action_error, value_error]).all()
        or max(action_error, value_error) > ONNX_TOLERANCE
    ):
        raise ValueError(
            f"ONNX mismatch: action={action_error}, value={value_error}; limit={ONNX_TOLERANCE}"
        )
    return {
        "samples": len(values),
        "batch_sizes": batch_sizes,
        "max_action_error": action_error,
        "max_value_error": value_error,
        "tolerance": ONNX_TOLERANCE,
        "input_dtype": "float64",
        "outputs": ["action", "value"],
    }

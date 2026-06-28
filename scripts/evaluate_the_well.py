#!/usr/bin/env python3
r"""Evaluate PDEformer on a split from The Well.

The Well stores trajectories, not PDEformer DAGs. This script adapts a Well
sample to PDEformer by building a PDE graph whose initial conditions come from
the first input frame. The default ``well_equation`` preset uses the local Well
equation registry; ``generic_unknown`` remains available as an equation-agnostic
adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cell import get_model
from src.data.pde_dag import PDEAsDAG, PDENodesCollector
from src.data.well_equations import build_well_pde_dag
from src.torch_compat import Tensor, context
from src.torch_compat import dtype as mstype
from src.utils import load_config

try:
    from the_well.benchmark.metrics import validation_metric_suite
    from the_well.data import WellDataset
    from the_well.data.normalization import RMSNormalization, ZScoreNormalization
except ImportError as exc:  # pragma: no cover - exercised by users without extra.
    WellDataset = None
    validation_metric_suite = None
    ZScoreNormalization = None
    RMSNormalization = None
    THE_WELL_IMPORT_ERROR = exc
else:
    THE_WELL_IMPORT_ERROR = None


Array = np.ndarray


@dataclass
class MetricAccumulator:
    r"""Running metric accumulator."""

    rel_l2: List[float]
    rel_l2_by_channel: Dict[int, List[float]]
    well_metrics: Dict[str, List[float]]
    well_metrics_by_channel: Dict[str, Dict[int, List[float]]]

    def append(self, channel: int, pred: Array, target: Array) -> None:
        err = np.linalg.norm((pred - target).reshape(-1))
        denom = np.linalg.norm(target.reshape(-1)) + 1.0e-6
        rel_l2 = float(np.clip(err / denom, 0.0, 5.0))
        self.rel_l2.append(rel_l2)
        self.rel_l2_by_channel.setdefault(channel, []).append(rel_l2)

    def append_well_metric(self,
                           metric_name: str,
                           values: torch.Tensor,
                           channels: Sequence[int]) -> None:
        r"""Append Well metric values shaped like [B, T, ..., C]."""
        values = values.detach().cpu().float()
        self.well_metrics.setdefault(metric_name, []).extend(
            values.reshape(-1).tolist())

        channel_axis = values.shape[-1]
        if channel_axis != len(channels):
            return
        for local_idx, channel in enumerate(channels):
            channel_values = values[..., local_idx].reshape(-1).tolist()
            self.well_metrics_by_channel.setdefault(
                metric_name, {}).setdefault(channel, []).extend(channel_values)

    def summary(self) -> Dict[str, object]:
        def summarize(values: Sequence[float]) -> Dict[str, float]:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {"mean": math.nan, "median": math.nan, "max": math.nan}
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "max": float(arr.max()),
            }

        return {
            "relative_l2": summarize(self.rel_l2),
            "relative_l2_by_channel": {
                str(channel): summarize(values)
                for channel, values in self.rel_l2_by_channel.items()
            },
            "well_metrics": {
                name: summarize(values)
                for name, values in self.well_metrics.items()
            },
            "well_metrics_by_channel": {
                name: {
                    str(channel): summarize(values)
                    for channel, values in channel_dict.items()
                }
                for name, channel_dict in self.well_metrics_by_channel.items()
            },
            "num_channel_evaluations": len(self.rel_l2),
        }


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    r"""Parse comma-separated integers; ``None`` means auto-select."""
    if value is None or value.strip() == "":
        return None
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PDEformer on The Well via the WellDataset API.")
    parser.add_argument("--config", default="configs/inference/model-L.yaml",
                        help="PDEformer YAML config.")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint path overriding model.load_ckpt.")
    parser.add_argument("--well-base-path", default=None,
                        help=("Base directory containing Well datasets, or "
                              "hf://datasets/polymathic-ai/ for streaming."))
    parser.add_argument("--well-dataset", default=None,
                        help="The Well dataset name, e.g. active_matter.")
    parser.add_argument("--well-path", default=None,
                        help=("Optional direct path to one Well dataset root. "
                              "Use this for private or dummy Well-format data."))
    parser.add_argument("--split", default="test",
                        choices=["train", "valid", "test"],
                        help="Well split to evaluate.")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of Well samples to evaluate.")
    parser.add_argument("--n-steps-input", type=int, default=1,
                        help="Number of input steps requested from WellDataset.")
    parser.add_argument("--n-steps-output", type=int, default=1,
                        help="Number of output steps requested from WellDataset.")
    parser.add_argument("--dt-stride", type=int, default=1,
                        help="Temporal stride used by WellDataset.")
    parser.add_argument("--field-indices", default=None,
                        help=("Comma-separated output channels to evaluate. "
                              "Defaults to as many leading channels as fit."))
    parser.add_argument("--max-fields", type=int, default=4,
                        help="Maximum leading channels to evaluate when auto-selecting.")
    parser.add_argument("--pde-preset", default="well_equation",
                        choices=["well_equation", "generic_unknown", "constant"],
                        help="Per-channel PDE graph preset.")
    parser.add_argument("--well-normalization", default="zscore",
                        choices=["none", "zscore", "rms"],
                        help=("Normalize PDEformer inputs with The Well stats and "
                              "denormalize predictions before metrics. The Well "
                              "benchmark commonly uses normalized model inputs."))
    parser.add_argument("--normalization-path", default=None,
                        help=("Optional stats.yaml path. Defaults to the WellDataset "
                              "normalization path."))
    parser.add_argument("--well-metrics", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Compute The Well validation metric suite.")
    parser.add_argument("--points-per-batch", type=int, default=65536,
                        help="Number of query points per model call.")
    parser.add_argument("--normalize-coordinates", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Map Well spatial coordinates to [0, 1] per axis.")
    parser.add_argument("--normalize-time", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Map requested output times to [0, 1].")
    parser.add_argument("--device-target", default="CPU", choices=["CPU", "GPU"],
                        help="Device target for the compatibility context.")
    parser.add_argument("--device", default=None,
                        help="Optional torch device override, e.g. cuda:0.")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path.")
    args = parser.parse_args()
    if args.well_path is None and (args.well_base_path is None or args.well_dataset is None):
        parser.error("Provide either --well-path, or both --well-base-path and --well-dataset.")
    return args


def require_the_well() -> None:
    if WellDataset is None or validation_metric_suite is None:
        raise RuntimeError(
            "Could not import the_well. Install project dependencies with "
            "`uv sync` or install `the-well>=1.2.0`.") from THE_WELL_IMPORT_ERROR


def normalization_cls(name: str):
    if name == "zscore":
        return ZScoreNormalization
    if name == "rms":
        return RMSNormalization
    if name == "none":
        return None
    raise ValueError(f"Unknown normalization: {name}")


def load_pdeformer(config_path: str, checkpoint: Optional[str], device: torch.device):
    r"""Load PDEformer from a YAML config and optional checkpoint override."""
    config = load_config(config_path)
    if checkpoint is not None:
        config.model.load_ckpt = checkpoint
    model = get_model(config, record=None, compute_dtype=mstype.float32)
    model.to(device)
    model.eval()
    return model, config


def as_numpy(sample_value: torch.Tensor) -> Array:
    return sample_value.detach().cpu().numpy().astype(np.float32)


def denormalize_selected_channels(array: Array,
                                  norm,
                                  channels: Sequence[int]) -> Array:
    r"""Denormalize selected flattened variable channels from a Well normalizer."""
    if norm is None:
        return array.astype(np.float32)

    tensor = torch.as_tensor(array, dtype=torch.float32)
    idx = torch.as_tensor(channels, dtype=torch.long)
    if hasattr(norm, "flattened_means") and hasattr(norm, "flattened_stds"):
        means = norm.flattened_means["variable"].to(tensor.device)[idx]
        stds = norm.flattened_stds["variable"].to(tensor.device)[idx]
        return (tensor * stds + means).detach().cpu().numpy().astype(np.float32)
    if hasattr(norm, "flattened_rmss"):
        rmss = norm.flattened_rmss["variable"].to(tensor.device)[idx]
        return (tensor * rmss).detach().cpu().numpy().astype(np.float32)
    raise TypeError(f"Unsupported Well normalizer type: {type(norm)!r}")


def normalize_axis(values: Array) -> Array:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def normalize_space_grid(space_grid: Array, enabled: bool) -> Array:
    if not enabled:
        return space_grid.astype(np.float32)
    out = np.empty_like(space_grid, dtype=np.float32)
    for axis in range(space_grid.shape[-1]):
        out[..., axis] = normalize_axis(space_grid[..., axis])
    return out


def normalize_time_grid(time_grid: Array, enabled: bool) -> Array:
    time_grid = time_grid.astype(np.float32)
    if not enabled:
        return time_grid
    max_abs = float(np.max(np.abs(time_grid)))
    if max_abs < 1.0e-12:
        return np.zeros_like(time_grid, dtype=np.float32)
    return time_grid / max_abs


def coordinate_array(space_grid: Array, output_time_grid: Array) -> Array:
    r"""Build PDEformer txyz query coordinates from Well grids."""
    if space_grid.ndim < 2:
        raise ValueError("space_grid must include spatial axes and a coordinate axis.")
    n_spatial = space_grid.shape[-1]
    if n_spatial != 2:
        raise ValueError(
            "This PDEformer adapter currently supports 2D Cartesian Well grids; "
            f"got {n_spatial} spatial dimensions.")

    spatial_shape = space_grid.shape[:-1]
    n_t = output_time_grid.shape[0]
    coord = np.zeros((n_t, *spatial_shape, 4), dtype=np.float32)
    coord[..., 0] = output_time_grid.reshape((n_t,) + (1,) * len(spatial_shape))
    coord[..., 1:3] = space_grid[np.newaxis, ...]
    return coord


def interpolate_to_resolution(field: Array, space_grid: Array, resolution: int) -> Tuple[Array, Array, Array]:
    r"""Interpolate a 2D Well field onto PDEformer's function-encoder grid."""
    if field.ndim != 2:
        raise ValueError(f"Expected a scalar 2D field, got shape {field.shape}.")
    x_old = np.asarray(space_grid[:, 0, 0], dtype=np.float32)
    y_old = np.asarray(space_grid[0, :, 1], dtype=np.float32)
    x_new = np.linspace(float(x_old.min()), float(x_old.max()), resolution, dtype=np.float32)
    y_new = np.linspace(float(y_old.min()), float(y_old.max()), resolution, dtype=np.float32)
    interp = RegularGridInterpolator(
        (x_old, y_old),
        field.astype(np.float32),
        bounds_error=False,
        fill_value=None,
    )
    x_ext, y_ext = np.meshgrid(x_new, y_new, indexing="ij")
    points = np.stack([x_ext, y_ext], axis=-1)
    return interp(points).astype(np.float32), x_ext, y_ext


def build_channel_dag(
        config,
        ic_field: Array,
        space_grid: Array,
        channel: int,
        preset: str) -> PDEAsDAG:
    r"""Build a per-channel PDEformer DAG using the initial condition."""
    resolution = int(config.model.function_encoder.get("resolution", 128))
    ic_on_fenc, x_ext, y_ext = interpolate_to_resolution(
        ic_field, space_grid, resolution)

    pde = PDENodesCollector(dim=2)
    u_node = pde.new_uf()
    pde.set_ic(u_node, ic_on_fenc, x=x_ext, y=y_ext)

    if preset == "constant":
        pde.sum_eq0(u_node.dt)
    elif preset == "generic_unknown":
        channel_tag = pde.new_coef(float(channel))
        unknown_term = pde.unknown_func(u_node, channel_tag)[0]
        pde.sum_eq0(u_node.dt, unknown_term)
    else:  # pragma: no cover - argparse keeps this closed.
        raise ValueError(f"Unknown PDE preset: {preset}")

    return pde.gen_dag(config)


def tensor_batch(array: Array, device: torch.device) -> torch.Tensor:
    return Tensor(array).unsqueeze(0).to(device)


def predict_channel(
        model,
        pde_dag: PDEAsDAG,
        coordinate: Array,
        points_per_batch: int,
        device: torch.device,
        idx_var: int = 0) -> Array:
    r"""Run PDEformer for one DAG/channel over all query coordinates."""
    flat_coord = coordinate.reshape(-1, 4).astype(np.float32)
    spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(idx_var)
    graph_inputs = (
        tensor_batch(pde_dag.node_type, device),
        tensor_batch(pde_dag.node_scalar, device),
        tensor_batch(pde_dag.node_function, device),
        tensor_batch(pde_dag.in_degree, device),
        tensor_batch(pde_dag.out_degree, device),
        tensor_batch(attn_bias, device),
        tensor_batch(spatial_pos, device),
    )

    pred_chunks = []
    with torch.no_grad():
        for start in range(0, flat_coord.shape[0], points_per_batch):
            stop = min(start + points_per_batch, flat_coord.shape[0])
            coord_chunk = tensor_batch(flat_coord[start:stop], device)
            pred = model(*graph_inputs, coord_chunk)
            pred_chunks.append(pred.squeeze(0).detach().cpu().numpy())
    pred_flat = np.concatenate(pred_chunks, axis=0).astype(np.float32)
    return pred_flat.reshape(coordinate.shape[:-1])


def choose_channels(sample: Dict[str, torch.Tensor],
                    field_indices: Optional[List[int]],
                    max_fields: int,
                    max_function_nodes: int) -> List[int]:
    n_channels = int(sample["output_fields"].shape[-1])
    if field_indices is None:
        n_auto = min(n_channels, max_fields, max_function_nodes)
        return list(range(n_auto))
    invalid = [idx for idx in field_indices if idx < 0 or idx >= n_channels]
    if invalid:
        raise ValueError(
            f"Field index out of range for {n_channels} channels: {invalid}")
    return field_indices


def iter_sample_indices(num_available: int, num_requested: int) -> Iterable[int]:
    return range(min(num_available, num_requested))


def append_well_metrics(accumulator: MetricAccumulator,
                        pred: Array,
                        target: Array,
                        metadata,
                        channels: Sequence[int]) -> None:
    r"""Compute The Well validation metric suite on channel-last trajectories."""
    pred_tensor = torch.as_tensor(pred, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)
    for metric in validation_metric_suite:
        values = metric(pred_tensor, target_tensor, metadata)
        if not isinstance(values, dict):
            values = {metric.__class__.__name__: values}
        for metric_name, metric_values in values.items():
            accumulator.append_well_metric(metric_name, metric_values, channels)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    require_the_well()
    context.set_context(device_target=args.device_target)
    default_device = "cpu"
    if args.device_target == "GPU" and torch.cuda.is_available():
        default_device = "cuda:0"
    device = torch.device(args.device or default_device)

    model, config = load_pdeformer(args.config, args.checkpoint, device)
    norm_cls = normalization_cls(args.well_normalization)
    dataset_kwargs = {
        "well_split_name": args.split,
        "normalization_path": args.normalization_path,
        "use_normalization": norm_cls is not None,
        "normalization_type": norm_cls,
        "n_steps_input": args.n_steps_input,
        "n_steps_output": args.n_steps_output,
        "min_dt_stride": args.dt_stride,
        "max_dt_stride": args.dt_stride,
        "return_grid": True,
        "boundary_return_type": "padding",
    }
    if args.well_path is not None:
        dataset = WellDataset(path=args.well_path, **dataset_kwargs)
    else:
        dataset = WellDataset(
            well_base_path=args.well_base_path,
            well_dataset_name=args.well_dataset,
            **dataset_kwargs,
        )

    accumulator = MetricAccumulator(
        rel_l2=[],
        rel_l2_by_channel={},
        well_metrics={},
        well_metrics_by_channel={},
    )
    channel_names = dataset.metadata.field_names
    norm = getattr(dataset, "norm", None)
    selected_channels = None
    examples = []

    for sample_idx in iter_sample_indices(len(dataset), args.num_samples):
        sample = dataset[sample_idx]
        input_fields = as_numpy(sample["input_fields"])
        output_fields = as_numpy(sample["output_fields"])
        space_grid = normalize_space_grid(as_numpy(sample["space_grid"]),
                                          args.normalize_coordinates)
        output_time_grid = normalize_time_grid(as_numpy(sample["output_time_grid"]),
                                               args.normalize_time)
        coordinate = coordinate_array(space_grid, output_time_grid)

        if selected_channels is None:
            selected_channels = choose_channels(
                sample,
                parse_int_list(args.field_indices),
                args.max_fields,
                int(config.data.pde_dag.max_n_function_nodes),
            )

        ic_frame = input_fields[-1]
        pde_dag = None
        selected_names = [channel_names[channel] for channel in selected_channels]
        if args.pde_preset == "well_equation":
            pde_dag = build_well_pde_dag(
                config,
                dataset.metadata.dataset_name,
                ic_frame[..., selected_channels],
                space_grid,
                selected_names,
            )
        pred_channels = []
        target_channels = []
        for local_idx, channel in enumerate(selected_channels):
            if args.pde_preset != "well_equation":
                pde_dag = build_channel_dag(
                    config,
                    ic_frame[..., channel],
                    space_grid,
                    channel,
                    args.pde_preset,
                )
            pred = predict_channel(
                model,
                pde_dag,
                coordinate,
                args.points_per_batch,
                device,
                idx_var=local_idx if args.pde_preset == "well_equation" else 0,
            )
            target = output_fields[..., channel]
            pred_raw = denormalize_selected_channels(
                pred[..., np.newaxis], norm, [channel])[..., 0]
            target_raw = denormalize_selected_channels(
                target[..., np.newaxis], norm, [channel])[..., 0]
            pred_channels.append(pred_raw)
            target_channels.append(target_raw)
            accumulator.append(channel, pred_raw, target_raw)

        if pred_channels and args.well_metrics:
            pred_stack = np.stack(pred_channels, axis=-1)
            target_stack = np.stack(target_channels, axis=-1)
            append_well_metrics(
                accumulator,
                pred_stack,
                target_stack,
                dataset.metadata,
                selected_channels,
            )

        if len(examples) < 3:
            examples.append({
                "sample_index": sample_idx,
                "input_fields_shape": list(input_fields.shape),
                "output_fields_shape": list(output_fields.shape),
                "space_grid_shape": list(space_grid.shape),
            })

    if selected_channels is None:
        selected_channels = []

    result = {
        "well_dataset": dataset.metadata.dataset_name,
        "split": args.split,
        "num_samples_requested": args.num_samples,
        "num_samples_evaluated": min(len(dataset), args.num_samples),
        "pde_preset": args.pde_preset,
        "selected_channel_indices": selected_channels,
        "well_channel_names_by_tensor_order": channel_names,
        "normalization": {
            "coordinates_to_unit_box": args.normalize_coordinates,
            "time_to_unit_interval": args.normalize_time,
            "well_input_normalization": args.well_normalization,
            "normalization_path": str(getattr(dataset, "normalization_path", "")),
            "predictions_denormalized_for_metrics": norm is not None,
        },
        "metrics": accumulator.summary(),
        "examples": examples,
        "notes": [
            "The Well provides trajectories, not PDEformer symbolic DAGs.",
            "With --pde-preset well_equation, this evaluator uses the dataset-specific equations in src.data.well_equations.",
            "When --well-normalization is not 'none', PDEformer receives normalized initial conditions and metrics are computed after denormalization.",
            "The Well validation metric suite is computed on channel-last [B,T,...,C] tensors for the selected channels.",
            "Some Well descriptions reference paper equations or unresolved source terms; those DAGs include learned closure nodes for the unspecified terms.",
        ],
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

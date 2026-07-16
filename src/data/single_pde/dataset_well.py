r"""Single-PDE fine-tuning adapter for Polymathic AI's The Well datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..env import float_dtype
from ..well_equations import (
    build_well_pde_template,
    coordinate_array,
    forecast_time_grid,
    get_well_equation_spec,
    interpolate_fields_to_resolution,
    normalize_space_grid,
    normalize_time_grid,
)
from .basics import CartesianGridInputFileDataset, register_pde_type

try:
    from the_well.data import WellDataset
    from the_well.data.normalization import RMSNormalization, ZScoreNormalization
except ImportError as exc:  # pragma: no cover - user environment dependent.
    WellDataset = None
    RMSNormalization = None
    ZScoreNormalization = None
    THE_WELL_IMPORT_ERROR = exc
else:
    THE_WELL_IMPORT_ERROR = None


def _as_numpy(value: Any) -> NDArray[np.float32]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _normalization_cls(name: str):
    if name == "zscore":
        return ZScoreNormalization
    if name == "rms":
        return RMSNormalization
    if name in ("none", "", None):
        return None
    raise ValueError(f"Unknown Well normalization: {name}")


def _parse_param(pde_param: Any, default_split: str) -> Tuple[str, str]:
    text = str(pde_param)
    for sep in ("@", ":", "|"):
        if sep in text:
            dataset_name, split = text.rsplit(sep, 1)
            if split in {"train", "valid", "test"}:
                return dataset_name, split
    return text, default_split


def _field_names_from_metadata(metadata) -> Tuple[str, ...]:
    names = getattr(metadata, "field_names", None)
    if isinstance(names, (list, tuple)):
        return tuple(str(name) for name in names)
    if isinstance(names, dict):
        out = []
        for key in sorted(names):
            out.extend(str(name) for name in names[key])
        return tuple(out)
    return ()


def _infer_channel_count(fields: NDArray[np.float32],
                         spatial_shape: Tuple[int, ...],
                         array_name: str) -> int:
    if fields.ndim == len(spatial_shape) + 1:
        if tuple(fields.shape[:len(spatial_shape)]) == spatial_shape:
            return int(fields.shape[-1])
        if tuple(fields.shape[-len(spatial_shape):]) == spatial_shape:
            return int(fields.shape[0])
    if fields.ndim == len(spatial_shape) + 2:
        if tuple(fields.shape[1:1 + len(spatial_shape)]) == spatial_shape:
            return int(fields.shape[-1])
        if tuple(fields.shape[-len(spatial_shape):]) == spatial_shape:
            return int(fields.shape[1])
    raise ValueError(
        f"Cannot infer channel axis for {array_name} with shape {fields.shape} "
        f"and spatial shape {spatial_shape}.")


def _select_channels_channel_last(fields: NDArray[np.float32],
                                  selected: Tuple[int, ...],
                                  spatial_shape: Tuple[int, ...],
                                  array_name: str) -> NDArray[np.float32]:
    n_spatial = len(spatial_shape)
    if fields.ndim == n_spatial + 1:
        if tuple(fields.shape[:n_spatial]) == spatial_shape:
            channel_last = fields
        elif tuple(fields.shape[-n_spatial:]) == spatial_shape:
            channel_last = np.moveaxis(fields, 0, -1)
        else:
            raise ValueError(
                f"Cannot align {array_name} shape {fields.shape} with "
                f"spatial shape {spatial_shape}.")
    elif fields.ndim == n_spatial + 2:
        if tuple(fields.shape[1:1 + n_spatial]) == spatial_shape:
            channel_last = fields
        elif tuple(fields.shape[-n_spatial:]) == spatial_shape:
            channel_last = np.moveaxis(fields, 1, -1)
        else:
            raise ValueError(
                f"Cannot align {array_name} shape {fields.shape} with "
                f"spatial shape {spatial_shape}.")
    else:
        raise ValueError(
            f"Expected {array_name} to have spatial or time-spatial axes, "
            f"got shape {fields.shape}.")
    return np.take(channel_last, selected, axis=-1)


def _last_input_frame_channel_last(input_fields: NDArray[np.float32],
                                   selected: Tuple[int, ...],
                                   spatial_shape: Tuple[int, ...]) -> NDArray[np.float32]:
    if input_fields.ndim == len(spatial_shape) + 2:
        input_fields = input_fields[-1]
    return _select_channels_channel_last(
        input_fields, selected, spatial_shape, "input_fields")


def _coordinate_with_z_axis(space_grid: NDArray[np.float32],
                            output_time_grid: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.expand_dims(coordinate_array(space_grid, output_time_grid), axis=-2)


def _coordinate_scales(raw_space_grid: NDArray[np.float32],
                       raw_output_time_grid: NDArray[np.float32],
                       normalize_coordinates: bool,
                       normalize_time: bool) -> Dict[str, float]:
    scales: Dict[str, float] = {"x": 1.0, "y": 1.0, "time": 1.0}
    if normalize_coordinates:
        for axis, name in enumerate(("x", "y")):
            axis_values = raw_space_grid[..., axis]
            span = float(np.max(axis_values) - np.min(axis_values))
            scales[name] = span if span > 1.0e-12 else 1.0
    if normalize_time:
        time_scale = float(np.max(np.abs(raw_output_time_grid)))
        scales["time"] = time_scale if time_scale > 1.0e-12 else 1.0
    return scales


def _constant_scalar_dict(metadata, values: Any) -> Dict[str, float]:
    names = tuple(str(name) for name in getattr(metadata, "constant_scalar_names", ()) or ())
    if not names:
        return {}
    scalar_values = _as_numpy(values).reshape(-1)
    return {
        name: float(scalar_values[idx])
        for idx, name in enumerate(names)
        if idx < scalar_values.size
    }


def _field_affine_from_normalizer(norm, field_names: Sequence[str],
                                  channel_indices: Sequence[int]
                                  ) -> Dict[str, Tuple[float, float]]:
    r"""Return physical = mean + scale * normalized for selected fields."""
    if norm is None:
        return {name: (0.0, 1.0) for name in field_names}
    if hasattr(norm, "flattened_means") and hasattr(norm, "flattened_stds"):
        means = _as_numpy(norm.flattened_means["variable"]).reshape(-1)
        scales = _as_numpy(norm.flattened_stds["variable"]).reshape(-1)
    elif hasattr(norm, "flattened_rmss"):
        means = np.zeros(len(norm.flattened_rmss["variable"]), dtype=np.float32)
        scales = _as_numpy(norm.flattened_rmss["variable"]).reshape(-1)
    else:
        raise TypeError(f"Unsupported Well normalizer type: {type(norm)!r}")
    return {
        name: (float(means[idx]), float(scales[idx]))
        for name, idx in zip(field_names, channel_indices)
    }


def _delta_affine_from_normalizer(norm, field_names: Sequence[str],
                                  channel_indices: Sequence[int]
                                  ) -> Dict[str, Tuple[float, float]]:
    r"""Return physical delta = mean + scale * normalized delta."""
    if norm is None:
        return {name: (0.0, 1.0) for name in field_names}
    if (hasattr(norm, "flattened_means_delta")
            and hasattr(norm, "flattened_stds_delta")):
        means = _as_numpy(norm.flattened_means_delta["variable"]).reshape(-1)
        scales = _as_numpy(norm.flattened_stds_delta["variable"]).reshape(-1)
    elif hasattr(norm, "flattened_rmss_delta"):
        scales = _as_numpy(norm.flattened_rmss_delta["variable"]).reshape(-1)
        means = np.zeros_like(scales)
    else:
        raise TypeError(
            "The selected Well normalizer does not provide delta statistics: "
            f"{type(norm)!r}")
    return {
        name: (float(means[idx]), float(scales[idx]))
        for name, idx in zip(field_names, channel_indices)
    }


def _affine_vectors(affine: Dict[str, Tuple[float, float]],
                    field_names: Sequence[str]
                    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    means = np.asarray([affine[name][0] for name in field_names], dtype=np.float32)
    scales = np.asarray([affine[name][1] for name in field_names], dtype=np.float32)
    if np.any(~np.isfinite(scales)) or np.any(np.abs(scales) < 1.0e-12):
        raise ValueError(f"Invalid normalization scales for fields {tuple(field_names)}")
    return means, scales


def _normalized_residual_target(
        input_frame: NDArray[np.float32],
        output_fields: NDArray[np.float32],
        field_affine: Dict[str, Tuple[float, float]],
        delta_affine: Dict[str, Tuple[float, float]],
        field_names: Sequence[str]) -> NDArray[np.float32]:
    r"""Convert normalized absolute states to normalized physical deltas."""
    _, field_scales = _affine_vectors(field_affine, field_names)
    delta_means, delta_scales = _affine_vectors(delta_affine, field_names)
    while input_frame.ndim < output_fields.ndim:
        input_frame = np.expand_dims(input_frame, axis=0)
    physical_delta = (output_fields - input_frame) * field_scales
    return ((physical_delta - delta_means) / delta_scales).astype(np.float32)


def _normalized_absolute_from_residual(
        input_frame: NDArray[np.float32],
        residual: NDArray[np.float32],
        field_affine: Dict[str, Tuple[float, float]],
        delta_affine: Dict[str, Tuple[float, float]],
        field_names: Sequence[str]) -> NDArray[np.float32]:
    r"""Reconstruct normalized absolute states from normalized deltas."""
    _, field_scales = _affine_vectors(field_affine, field_names)
    delta_means, delta_scales = _affine_vectors(delta_affine, field_names)
    while input_frame.ndim < residual.ndim:
        input_frame = np.expand_dims(input_frame, axis=0)
    physical_delta = residual * delta_scales + delta_means
    return (input_frame + physical_delta / field_scales).astype(np.float32)


def _selected_equation_scalar_values(
        dataset_name: str,
        scalar_dict: Dict[str, float],
        names: Sequence[str],
        scale: float = 1.0) -> NDArray[np.float32]:
    if dataset_name == "gray_scott_reaction_diffusion":
        return np.asarray(
            [scalar_dict[name] * scale for name in names],
            dtype=float_dtype,
        )
    return np.empty(0, dtype=float_dtype)


def _coprime_stride(size: int) -> int:
    r"""Choose a deterministic stride that spreads short prefixes uniformly."""
    if size <= 1:
        return 1
    stride = max(1, int(size * 0.6180339887498949))
    while np.gcd(stride, size) != 1:
        stride -= 1
    return stride


@register_pde_type("the_well", "well", "the-well")
class TheWellInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D Well trajectories and attach dataset-specific PDE DAGs."""

    n_vars: int = 1
    var_latex = ("u",)
    pde_latex = "The Well dataset equation"
    coef_dict: Dict[str, float] = {}

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)
        if WellDataset is None:
            raise RuntimeError(
                "Could not import the_well. Install dependencies with `uv sync` "
                "or install `the-well>=1.2.0`.") from THE_WELL_IMPORT_ERROR

        well_cfg = config.data.single_pde.get("well", {})
        default_split = str(well_cfg.get("split", "train"))
        dataset_name, split = _parse_param(pde_param, default_split)
        self.dataset_name = dataset_name
        self.split = split
        self.spec = get_well_equation_spec(dataset_name)
        self.pde_latex = self.spec.pde_latex
        self.coef_dict = {"well_equation": self.spec.dataset}

        norm_name = str(well_cfg.get("normalization", "zscore"))
        norm_cls = _normalization_cls(norm_name)
        dataset_kwargs = {
            "well_split_name": split,
            "normalization_path": well_cfg.get("normalization_path", None),
            "use_normalization": norm_cls is not None,
            "normalization_type": norm_cls,
            "n_steps_input": int(well_cfg.get("n_steps_input", 1)),
            "n_steps_output": int(well_cfg.get("n_steps_output", 1)),
            "min_dt_stride": int(well_cfg.get("dt_stride", 1)),
            "max_dt_stride": int(well_cfg.get("dt_stride", 1)),
            "return_grid": True,
            "boundary_return_type": str(well_cfg.get("boundary_return_type", "padding")),
        }

        direct_path = str(well_cfg.get("path", "") or "")
        if direct_path:
            self.dataset = WellDataset(path=direct_path, **dataset_kwargs)
        else:
            base_path = str(well_cfg.get("base_path", "") or config.data.path)
            candidate_path = Path(base_path)
            if candidate_path.name == dataset_name and candidate_path.exists():
                self.dataset = WellDataset(path=str(candidate_path), **dataset_kwargs)
            else:
                self.dataset = WellDataset(
                    well_base_path=base_path,
                    well_dataset_name=dataset_name,
                    **dataset_kwargs,
                )

        first_sample = self.dataset[0]
        self.dataset_size = len(self.dataset)
        self.sample_indexing = str(well_cfg.get("sample_indexing", "uniform")).lower()
        if self.sample_indexing not in {"uniform", "sequential"}:
            raise ValueError(
                "data.single_pde.well.sample_indexing must be 'uniform' or "
                f"'sequential', got {self.sample_indexing!r}.")
        self._sample_stride = _coprime_stride(self.dataset_size)
        self._sample_offset = int(config.get("seed", 0)) % self.dataset_size
        self.target_mode = str(well_cfg.get("target_mode", "absolute")).lower()
        if self.target_mode not in {"absolute", "residual"}:
            raise ValueError(
                "data.single_pde.well.target_mode must be 'absolute' or "
                f"'residual', got {self.target_mode!r}.")
        self.normalize_coordinates = bool(well_cfg.get("normalize_coordinates", True))
        self.normalize_time = bool(well_cfg.get("normalize_time", True))
        raw_space_grid = _as_numpy(first_sample["space_grid"])
        raw_output_time_grid = forecast_time_grid(
            _as_numpy(first_sample["input_time_grid"]),
            _as_numpy(first_sample["output_time_grid"]),
        )
        self.coordinate_scales = _coordinate_scales(
            raw_space_grid,
            raw_output_time_grid,
            self.normalize_coordinates,
            self.normalize_time,
        )
        space_grid = normalize_space_grid(raw_space_grid, self.normalize_coordinates)
        if space_grid.shape[-1] != 2:
            raise ValueError(
                f"{dataset_name} has {space_grid.shape[-1]} spatial dimensions. "
                "PDEformer-2 fine-tuning currently supports 2D Well datasets.")
        spatial_shape = tuple(space_grid.shape[:-1])
        output_fields = _as_numpy(first_sample["output_fields"])
        n_channels = _infer_channel_count(output_fields, spatial_shape, "output_fields")
        field_names = _field_names_from_metadata(self.dataset.metadata)
        if not field_names:
            field_names = tuple(f"field_{idx}" for idx in range(n_channels))

        explicit_fields = well_cfg.get("field_indices", None)
        max_fields = int(well_cfg.get(
            "max_fields", config.data.pde_dag.max_n_function_nodes))
        if explicit_fields is None:
            selected = list(range(min(n_channels, max_fields)))
        else:
            selected = [int(idx) for idx in explicit_fields]
        invalid = [idx for idx in selected if idx < 0 or idx >= n_channels]
        if invalid:
            raise ValueError(
                f"Well field index out of range for {n_channels} channels: {invalid}")
        if len(selected) > int(config.data.pde_dag.max_n_function_nodes):
            raise ValueError(
                "Selected Well field count exceeds data.pde_dag.max_n_function_nodes.")

        self.selected_channels = tuple(selected)
        self.field_names = tuple(field_names[idx] for idx in selected)
        if self.dataset_name == "gray_scott_reaction_diffusion":
            required_fields = {"A", "B"}
            selected_fields = set(self.field_names)
            if not required_fields.issubset(selected_fields):
                raise ValueError(
                    "Gray-Scott Reaction-Diffusion requires both A and B fields "
                    "for the Well PDE DAG. Set FIELD_INDICES='0,1' or omit "
                    "FIELD_INDICES when MAX_FIELDS>=2.")
        self.n_vars = len(self.selected_channels)
        self.var_latex = self.field_names
        self.field_affine = _field_affine_from_normalizer(
            getattr(self.dataset, "norm", None),
            self.field_names,
            self.selected_channels,
        )
        self.delta_affine = {}
        if self.target_mode == "residual":
            self.delta_affine = _delta_affine_from_normalizer(
                getattr(self.dataset, "norm", None),
                self.field_names,
                self.selected_channels,
            )
        output_time_grid = normalize_time_grid(raw_output_time_grid, self.normalize_time)
        self.txyz_coord = _coordinate_with_z_axis(space_grid, output_time_grid)

        self.function_resolution = int(config.model.function_encoder.get("resolution", 128))
        dummy_fields = np.zeros((*space_grid.shape[:-1], self.n_vars), dtype=np.float32)
        _, x_ext, y_ext = interpolate_fields_to_resolution(
            dummy_fields, space_grid, self.function_resolution)
        pde = build_well_pde_template(
            dataset_name,
            self.field_names,
            x_ext,
            y_ext,
            coordinate_scales=self.coordinate_scales,
            field_affine=self.field_affine,
        )
        self.pde_dag = pde.gen_dag(config)
        self._space_grid = space_grid

    def _sample_index(self, idx_pde: int) -> int:
        idx_pde = int(idx_pde)
        if self.sample_indexing == "uniform":
            return (self._sample_offset + idx_pde * self._sample_stride) \
                % self.dataset_size
        return idx_pde

    def _sample_and_input_frame(
            self, idx_pde: int
            ) -> Tuple[Dict[str, Any], NDArray[np.float32]]:
        sample = self.dataset[self._sample_index(idx_pde)]
        space_grid = normalize_space_grid(
            _as_numpy(sample["space_grid"]), self.normalize_coordinates)
        input_frame = _last_input_frame_channel_last(
            _as_numpy(sample["input_fields"]),
            self.selected_channels,
            tuple(space_grid.shape[:-1]),
        )
        return sample, input_frame

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        sample, ic_frame = self._sample_and_input_frame(idx_pde)
        output_fields = _as_numpy(sample["output_fields"])
        space_grid = normalize_space_grid(
            _as_numpy(sample["space_grid"]),
            self.normalize_coordinates,
        )
        raw_output_time_grid = forecast_time_grid(
            _as_numpy(sample["input_time_grid"]),
            _as_numpy(sample["output_time_grid"]),
        )
        output_time_grid = normalize_time_grid(
            raw_output_time_grid,
            self.normalize_time,
        )

        spatial_shape = tuple(space_grid.shape[:-1])
        input_field, _, _ = interpolate_fields_to_resolution(
            ic_frame, space_grid, self.function_resolution)
        u_label = _select_channels_channel_last(
            output_fields, self.selected_channels, spatial_shape, "output_fields")
        if self.target_mode == "residual":
            u_label = _normalized_residual_target(
                ic_frame,
                u_label,
                self.field_affine,
                self.delta_affine,
                self.field_names,
            )
        u_label = np.expand_dims(u_label, axis=-2)
        txyz_coord = _coordinate_with_z_axis(space_grid, output_time_grid)
        scalar_dict = _constant_scalar_dict(
            self.dataset.metadata,
            sample.get("constant_scalars", ()),
        )
        input_scalar = _selected_equation_scalar_values(
            self.dataset_name,
            scalar_dict,
            ("F", "k"),
            self.coordinate_scales["time"],
        )
        return input_field, input_scalar, txyz_coord, u_label

    def reconstruct_prediction(self,
                               idx_pde: int,
                               idx_var: int,
                               pred: torch.Tensor,
                               label: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Reconstruct normalized absolute states for residual-mode metrics."""
        if self.target_mode != "residual":
            return pred, label
        _, input_frame = self._sample_and_input_frame(idx_pde)
        input_var = input_frame[..., idx_var].reshape(-1)
        if pred.ndim != 2 or pred.shape[-1] != 1:
            raise ValueError(
                "Well residual reconstruction expects [n_points, 1], got "
                f"{tuple(pred.shape)}.")
        n_points = int(pred.shape[0])
        if n_points % input_var.size != 0:
            raise ValueError(
                f"Prediction has {n_points} points, which is not divisible by "
                f"the input grid size {input_var.size}.")
        n_times = n_points // input_var.size
        initial = np.tile(input_var, n_times).reshape(n_points, 1)
        initial = torch.as_tensor(initial, dtype=pred.dtype, device=pred.device)

        field_name = self.field_names[idx_var]
        field_scale = self.field_affine[field_name][1]
        delta_mean, delta_scale = self.delta_affine[field_name]

        def reconstruct(value: torch.Tensor) -> torch.Tensor:
            physical_delta = value * delta_scale + delta_mean
            return initial + physical_delta / field_scale

        return reconstruct(pred), reconstruct(label)

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info.update({
            "well_dataset": self.dataset_name,
            "well_split": self.split,
            "well_selected_channels": self.selected_channels,
            "well_field_names": self.field_names,
            "well_target_mode": self.target_mode,
            "well_equation_notes": self.spec.notes,
        })
        return data_info

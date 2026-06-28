r"""Single-PDE fine-tuning adapter for Polymathic AI's The Well datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..env import float_dtype
from ..well_equations import (
    build_well_pde_template,
    coordinate_array,
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
        field_names = _field_names_from_metadata(self.dataset.metadata)
        n_channels = int(first_sample["output_fields"].shape[-1])
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
        self.n_vars = len(self.selected_channels)
        self.var_latex = self.field_names
        self.dataset_size = len(self.dataset)
        self.normalize_coordinates = bool(well_cfg.get("normalize_coordinates", True))
        self.normalize_time = bool(well_cfg.get("normalize_time", True))

        space_grid = normalize_space_grid(
            _as_numpy(first_sample["space_grid"]),
            self.normalize_coordinates,
        )
        if space_grid.shape[-1] != 2:
            raise ValueError(
                f"{dataset_name} has {space_grid.shape[-1]} spatial dimensions. "
                "PDEformer-2 fine-tuning currently supports 2D Well datasets.")
        output_time_grid = normalize_time_grid(
            _as_numpy(first_sample["output_time_grid"]),
            self.normalize_time,
        )
        self.txyz_coord = coordinate_array(space_grid, output_time_grid)

        self.function_resolution = int(config.model.function_encoder.get("resolution", 128))
        dummy_fields = np.zeros((*space_grid.shape[:-1], self.n_vars), dtype=np.float32)
        _, x_ext, y_ext = interpolate_fields_to_resolution(
            dummy_fields, space_grid, self.function_resolution)
        pde = build_well_pde_template(dataset_name, self.field_names, x_ext, y_ext)
        self.pde_dag = pde.gen_dag(config)
        self._space_grid = space_grid

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        sample = self.dataset[int(idx_pde)]
        input_fields = _as_numpy(sample["input_fields"])
        output_fields = _as_numpy(sample["output_fields"])
        space_grid = normalize_space_grid(
            _as_numpy(sample["space_grid"]),
            self.normalize_coordinates,
        )
        output_time_grid = normalize_time_grid(
            _as_numpy(sample["output_time_grid"]),
            self.normalize_time,
        )

        ic_frame = input_fields[-1, ..., self.selected_channels]
        input_field, _, _ = interpolate_fields_to_resolution(
            ic_frame, space_grid, self.function_resolution)
        u_label = output_fields[..., self.selected_channels]
        u_label = np.expand_dims(u_label, axis=-2)
        txyz_coord = coordinate_array(space_grid, output_time_grid)
        input_scalar = np.empty(0, dtype=float_dtype)
        return input_field, input_scalar, txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info.update({
            "well_dataset": self.dataset_name,
            "well_split": self.split,
            "well_selected_channels": self.selected_channels,
            "well_field_names": self.field_names,
            "well_equation_notes": self.spec.notes,
        })
        return data_info

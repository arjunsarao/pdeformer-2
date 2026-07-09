"""Helpers for resolving dataset storage paths."""
import os

from omegaconf import DictConfig, OmegaConf


def local_dataset_path(config_data: DictConfig) -> str:
    """Return the local path used for dataset files."""
    return str(config_data.get("local_scratch", "") or config_data.path)


def use_local_dataset_path(config: DictConfig) -> None:
    """Make config.data.path point to the local dataset storage path."""
    data_path = local_dataset_path(config.data)
    if data_path == config.data.path:
        return

    os.makedirs(data_path, exist_ok=True)
    OmegaConf.set_struct(config.data, False)
    config.data.path = data_path

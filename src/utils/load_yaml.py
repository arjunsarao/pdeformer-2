r"""This module provides a function to load a configuration file."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(file_path: str) -> DictConfig:
    r"""
    Load a configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Tuple[dict, str]: The configuration dictionary and its string
            representation.
    """
    if not file_path.endswith(".yaml"):
        raise ValueError("The configuration file must be a yaml file")

    def _load_with_base(path: Path, seen: set[Path]) -> DictConfig:
        path = path.expanduser().resolve()
        if path in seen:
            raise ValueError(f"Circular base_config reference detected: {path}")
        seen.add(path)

        config = OmegaConf.load(path)
        base_config_path = str(config.get("base_config", "none"))
        if base_config_path.lower() != "none":
            base_path = Path(base_config_path).expanduser()
            if not base_path.is_absolute():
                file_relative_path = path.parent / base_path
                base_path = file_relative_path if file_relative_path.exists() else base_path
            base_config = _load_with_base(base_path, seen)
            base_config.merge_with(config)
            config = base_config

        seen.remove(path)
        return config

    config = _load_with_base(Path(file_path), set())

    return config

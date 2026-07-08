#!/usr/bin/env python
r"""Smoke-test The Well dataloader without launching training."""
from __future__ import annotations

import argparse
from typing import Any

import sys
from pathlib import Path

workspace_root = Path.cwd().resolve()
if not (workspace_root / "src").exists():workspace_root = workspace_root.parent
sys.path.append(str(workspace_root))

from src.data import load_dataset
from src.utils import load_config


def shape_of(value: Any) -> Any:
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file_path", "-c", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    config = load_config(args.config_file_path)
    _, _, train_iter_dict, test_iter_dict = load_dataset(config)
    iter_dict = train_iter_dict if args.split == "train" else test_iter_dict

    for pde_type, param_dict in iter_dict.items():
        for pde_param, (dataset_iter, _) in param_dict.items():
            print(f"{args.split} {pde_type} {pde_param}")
            batch = next(iter(dataset_iter))
            for idx, item in enumerate(batch):
                print(f"  [{idx}] {shape_of(item)}")
            return

    raise RuntimeError(f"No {args.split} dataloader found.")


if __name__ == "__main__":
    main()

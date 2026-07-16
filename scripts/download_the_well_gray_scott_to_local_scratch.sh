#!/bin/bash

set -euo pipefail

DATASET="${WELL_DATASET:-gray_scott_reaction_diffusion}"
DEST_ROOT="${DEST_ROOT:-/local_scratch/${USER:-$(id -un)}/the_well}"
SPLITS="${SPLITS:-train,valid,test}"
FIRST_ONLY="${FIRST_ONLY:-false}"
PARALLEL="${PARALLEL:-true}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_BIN="uv run python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No Python interpreter found. Run \`uv sync\` first or install python3." >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

export DATASET DEST_ROOT SPLITS FIRST_ONLY PARALLEL

${PYTHON_BIN} - <<'PY'
import os
from pathlib import Path

from the_well.utils.download import well_download


def as_bool(name: str) -> bool:
    return os.environ[name].strip().lower() in {"1", "true", "yes", "on"}


dataset = os.environ["DATASET"].strip()
dest_root = Path(os.environ["DEST_ROOT"]).expanduser().resolve()
splits = list(dict.fromkeys(
    split.strip() for split in os.environ["SPLITS"].split(",") if split.strip()
))
first_only = as_bool("FIRST_ONLY")
parallel = as_bool("PARALLEL")

print(f"Downloading {dataset} into {dest_root}")
print(f"Splits: {', '.join(splits)}")
print(f"First only: {first_only}")
print(f"Parallel: {parallel}")

for split in splits:
    well_download(
        base_path=str(dest_root),
        dataset=dataset,
        split=split,
        first_only=first_only,
        parallel=parallel,
    )

datasets_root = dest_root / "datasets"
dataset_src = datasets_root / dataset
dataset_link = dest_root / dataset

if not dataset_src.exists():
    raise FileNotFoundError(
        f"Expected downloaded dataset at {dataset_src}, but it does not exist."
    )

if dataset_link.exists() or dataset_link.is_symlink():
    try:
        same_target = dataset_link.resolve() == dataset_src.resolve()
    except FileNotFoundError:
        same_target = False
    if not same_target:
        raise FileExistsError(
            f"{dataset_link} already exists and does not point to {dataset_src}. "
            "Move it aside or set DEST_ROOT to a different scratch directory."
        )
else:
    dataset_link.symlink_to(Path("datasets") / dataset, target_is_directory=True)

print()
print(f"Finished downloading The Well dataset: {dataset}.")
print(f"Use WELL_BASE_PATH={dest_root} for training/eval.")
print(f"Dataset directory: {dataset_src}")
print(f"Convenience link: {dataset_link}")
PY

#!/bin/bash
#SBATCH --job-name=pdeformer-well
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Example, local downloaded Well data:
#   sbatch --export=ALL,WELL_BASE_PATH=/scratch/the_well,WELL_DATASET=active_matter scripts/submit_the_well_eval_slurm.sh
#
# Example, direct private/dummy Well-format path:
#   sbatch --export=ALL,WELL_PATH=/scratch/my_well_dataset,WELL_DATASET=my_dataset scripts/submit_the_well_eval_slurm.sh
#
# Example, Hugging Face streaming:
#   sbatch --export=ALL,WELL_BASE_PATH=hf://datasets/polymathic-ai/,WELL_DATASET=active_matter scripts/submit_the_well_eval_slurm.sh

CONFIG_PATH="${CONFIG_PATH:-configs/inference/model-L.yaml}"
CHECKPOINT="${CHECKPOINT:-model-L.pt}"
WELL_BASE_PATH="${WELL_BASE_PATH:-}"
WELL_DATASET="${WELL_DATASET:-active_matter}"
WELL_PATH="${WELL_PATH:-}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
N_STEPS_INPUT="${N_STEPS_INPUT:-1}"
N_STEPS_OUTPUT="${N_STEPS_OUTPUT:-1}"
DT_STRIDE="${DT_STRIDE:-1}"
FIELD_INDICES="${FIELD_INDICES:-}"
MAX_FIELDS="${MAX_FIELDS:-4}"
PDE_PRESET="${PDE_PRESET:-well_equation}"
WELL_NORMALIZATION="${WELL_NORMALIZATION:-zscore}"
TARGET_MODE="${TARGET_MODE:-auto}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-65536}"
SAMPLE_SEED="${SAMPLE_SEED:-123456}"
SAMPLE_INDEXING="${SAMPLE_INDEXING:-uniform}"
AUTO_PREPARE_WELL_DATASET="${AUTO_PREPARE_WELL_DATASET:-auto}"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
# Empty means auto-select CUDA when it can actually be initialized, else CPU.
DEVICE="${DEVICE:-}"
OUTPUT_PATH="${OUTPUT_PATH:-exp/the_well/${WELL_DATASET}_${SPLIT}_pdeformer_eval_${SLURM_JOB_ID:-local}.json}"

# Use the repo-local virtualenv when available; otherwise fall back to uv/python.
if [ -x ".venv/bin/python" ]; then
  PYTHON_CMD="${PYTHON_CMD:-.venv/bin/python}"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="${PYTHON_CMD:-uv run python}"
else
  PYTHON_CMD="${PYTHON_CMD:-python}"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs "$(dirname "${OUTPUT_PATH}")"

split_has_hdf5() {
  local split_dir="$1"
  compgen -G "${split_dir}/*.hdf5" >/dev/null \
    || compgen -G "${split_dir}/*.h5" >/dev/null
}

prepare_well_dataset_if_needed() {
  # Direct paths and remote backends manage their own storage.
  if [ -n "${WELL_PATH}" ] || [[ "${WELL_BASE_PATH}" == *"://"* ]]; then
    return 0
  fi

  local root_dir="${WELL_BASE_PATH}"
  local dataset_dir alt_dataset_dir
  if [ -z "${root_dir}" ]; then
    return 0
  fi
  if [ "$(basename "${root_dir}")" = "${WELL_DATASET}" ]; then
    dataset_dir="${root_dir}"
    root_dir="$(dirname "${root_dir}")"
  else
    dataset_dir="${root_dir}/${WELL_DATASET}"
  fi
  alt_dataset_dir="${root_dir}/datasets/${WELL_DATASET}"

  if split_has_hdf5 "${dataset_dir}/data/${SPLIT}"; then
    return 0
  fi
  if split_has_hdf5 "${alt_dataset_dir}/data/${SPLIT}"; then
    mkdir -p "${root_dir}"
    if [ -L "${dataset_dir}" ] || [ ! -e "${dataset_dir}" ]; then
      ln -sfn "datasets/${WELL_DATASET}" "${dataset_dir}"
      echo "Created ${dataset_dir} -> datasets/${WELL_DATASET}"
    fi
    return 0
  fi

  case "${AUTO_PREPARE_WELL_DATASET}" in
    true|TRUE|1|yes|YES|on|ON)
      ;;
    auto|AUTO)
      if [[ "${root_dir}" != /local_scratch/* ]]; then
        return 0
      fi
      ;;
    *)
      return 0
      ;;
  esac

  echo "Preparing ${WELL_DATASET}/${SPLIT} on $(hostname) at ${root_dir}"
  DEST_ROOT="${root_dir}" \
  WELL_DATASET="${WELL_DATASET}" \
  SPLITS="${SPLIT}" \
  bash scripts/download_the_well_gray_scott_to_local_scratch.sh

  if ! split_has_hdf5 "${dataset_dir}/data/${SPLIT}"; then
    echo "Dataset staging completed but ${dataset_dir}/data/${SPLIT} has no HDF5 files." >&2
    exit 1
  fi
}

prepare_well_dataset_if_needed

echo "job_id: ${SLURM_JOB_ID:-local}"
echo "host: $(hostname)"
echo "config_path: ${CONFIG_PATH}"
echo "checkpoint: ${CHECKPOINT}"
echo "well_base_path: ${WELL_BASE_PATH}"
echo "well_dataset: ${WELL_DATASET}"
echo "well_path: ${WELL_PATH}"
echo "split: ${SPLIT}"
echo "num_samples: ${NUM_SAMPLES}"
echo "field_indices: ${FIELD_INDICES}"
echo "well_normalization: ${WELL_NORMALIZATION}"
echo "target_mode: ${TARGET_MODE}"
echo "sample_indexing: ${SAMPLE_INDEXING}"
echo "sample_seed: ${SAMPLE_SEED}"
echo "output_path: ${OUTPUT_PATH}"
echo "device_target: ${DEVICE_TARGET}"
echo "device: ${DEVICE}"
echo "python_cmd: ${PYTHON_CMD}"

ARGS=(
  scripts/evaluate_the_well.py
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --num-samples "${NUM_SAMPLES}"
  --n-steps-input "${N_STEPS_INPUT}"
  --n-steps-output "${N_STEPS_OUTPUT}"
  --dt-stride "${DT_STRIDE}"
  --max-fields "${MAX_FIELDS}"
  --pde-preset "${PDE_PRESET}"
  --well-normalization "${WELL_NORMALIZATION}"
  --target-mode "${TARGET_MODE}"
  --points-per-batch "${POINTS_PER_BATCH}"
  --sample-seed "${SAMPLE_SEED}"
  --sample-indexing "${SAMPLE_INDEXING}"
  --device-target "${DEVICE_TARGET}"
  --output "${OUTPUT_PATH}"
)

if [ -n "${DEVICE}" ]; then
  ARGS+=(--device "${DEVICE}")
fi

if [ -n "${WELL_PATH}" ]; then
  ARGS+=(--well-path "${WELL_PATH}")
else
  ARGS+=(--well-base-path "${WELL_BASE_PATH}" --well-dataset "${WELL_DATASET}")
fi

if [ -n "${FIELD_INDICES}" ]; then
  ARGS+=(--field-indices "${FIELD_INDICES}")
fi

if [ -n "${NORMALIZATION_PATH}" ]; then
  ARGS+=(--normalization-path "${NORMALIZATION_PATH}")
fi

${PYTHON_CMD} "${ARGS[@]}"

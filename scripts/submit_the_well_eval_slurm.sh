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
NORMALIZATION_PATH="${NORMALIZATION_PATH:-}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-65536}"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_PATH="${OUTPUT_PATH:-exp/the_well/${WELL_DATASET}_${SPLIT}_pdeformer_eval.json}"

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
  --points-per-batch "${POINTS_PER_BATCH}"
  --device-target "${DEVICE_TARGET}"
  --device "${DEVICE}"
  --output "${OUTPUT_PATH}"
)

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

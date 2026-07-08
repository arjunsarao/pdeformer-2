#!/bin/bash
#SBATCH --job-name=pdeformer-well-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Example:
#   sbatch --export=ALL,WELL_BASE_PATH=/scratch/the_well,WELL_DATASET=gray_scott_reaction_diffusion scripts/submit_the_well_finetune_slurm.sh

BASE_CONFIG="${BASE_CONFIG:-configs/finetune/the_well_model-L.yaml}"
WELL_BASE_PATH="${WELL_BASE_PATH:-/path/to/the_well}"
WELL_DATASET="${WELL_DATASET:-gray_scott_reaction_diffusion}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-valid}"
CHECKPOINT="${CHECKPOINT:-model-L.pt}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-8}"
TEST_SAMPLES="${TEST_SAMPLES:-8}"
MAX_FIELDS="${MAX_FIELDS:-8}"
FIELD_INDICES="${FIELD_INDICES:-}"
N_STEPS_INPUT="${N_STEPS_INPUT:-1}"
N_STEPS_OUTPUT="${N_STEPS_OUTPUT:-1}"
DT_STRIDE="${DT_STRIDE:-1}"
WELL_NORMALIZATION="${WELL_NORMALIZATION:-zscore}"
NORMALIZE_COORDINATES="${NORMALIZE_COORDINATES:-true}"
NORMALIZE_TIME="${NORMALIZE_TIME:-true}"
EPOCHS="${EPOCHS:-20}"
NUM_TXYZ_SAMP_PTS="${NUM_TXYZ_SAMP_PTS:-4096}"
LR_INIT="${LR_INIT:-5.e-6}"
SEED="${SEED:-123456}"
BEST_METRIC="${BEST_METRIC:-eval_error_mean}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
PRESERVE_EVAL_RNG_STATE="${PRESERVE_EVAL_RNG_STATE:-true}"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
DEVICE_ID="${DEVICE_ID:-0}"
MODE="${MODE:-PYNATIVE}"
RECORD_DIR="${RECORD_DIR:-exp/finetune/the_well/${WELL_DATASET}/model-L}"
WANDB_MODE="${WANDB_MODE:-online}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_CMD="${PYTHON_CMD:-.venv/bin/python}"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="${PYTHON_CMD:-uv run python}"
else
  PYTHON_CMD="${PYTHON_CMD:-python}"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs "${RECORD_DIR}"

CONFIG_PATH="${CONFIG_PATH:-slurm_logs/the_well_finetune_${WELL_DATASET}_${SLURM_JOB_ID:-local}.yaml}"

export BASE_CONFIG WELL_BASE_PATH WELL_DATASET TRAIN_SPLIT TEST_SPLIT CHECKPOINT
export TRAIN_SAMPLES TEST_SAMPLES MAX_FIELDS FIELD_INDICES N_STEPS_INPUT N_STEPS_OUTPUT
export DT_STRIDE WELL_NORMALIZATION NORMALIZE_COORDINATES NORMALIZE_TIME
export EPOCHS NUM_TXYZ_SAMP_PTS LR_INIT SEED BEST_METRIC EVAL_INTERVAL PRESERVE_EVAL_RNG_STATE
export RECORD_DIR WANDB_MODE CONFIG_PATH

${PYTHON_CMD} - <<'PY'
import os
from pathlib import Path

field_indices = os.environ.get("FIELD_INDICES", "").strip()
if os.environ["WELL_DATASET"] == "gray_scott_reaction_diffusion":
    if not field_indices:
        field_indices = "0,1"
    elif field_indices == "0" and int(os.environ["MAX_FIELDS"]) >= 2:
        print(
            "Warning: Gray-Scott received FIELD_INDICES=0; using 0,1. "
            "Quote Slurm exports like FIELD_INDICES='0,1'.")
        field_indices = "0,1"
field_yaml = "null"
if field_indices:
    field_yaml = "[" + ", ".join(part.strip() for part in field_indices.split(",")) + "]"

text = f"""base_config: {os.environ["BASE_CONFIG"]}
seed: {os.environ["SEED"]}

model:
  load_ckpt: {os.environ["CHECKPOINT"]}

data:
  path: {os.environ["WELL_BASE_PATH"]}
  type: single_pde
  num_workers: 0
  num_samples_per_file:
    train: {os.environ["TRAIN_SAMPLES"]}
    test: {os.environ["TEST_SAMPLES"]}
  pde_dag:
    max_n_scalar_nodes: 256
    max_n_function_nodes: {os.environ["MAX_FIELDS"]}
    disconn_attn_bias: -inf
  single_pde:
    param_name: the_well
    regularize_ratio: 0.
    train: ["{os.environ["WELL_DATASET"]}:{os.environ["TRAIN_SPLIT"]}"]
    test: ["{os.environ["WELL_DATASET"]}:{os.environ["TEST_SPLIT"]}"]
    well:
      normalization: {os.environ["WELL_NORMALIZATION"]}
      n_steps_input: {os.environ["N_STEPS_INPUT"]}
      n_steps_output: {os.environ["N_STEPS_OUTPUT"]}
      dt_stride: {os.environ["DT_STRIDE"]}
      max_fields: {os.environ["MAX_FIELDS"]}
      field_indices: {field_yaml}
      normalize_coordinates: {os.environ["NORMALIZE_COORDINATES"]}
      normalize_time: {os.environ["NORMALIZE_TIME"]}

train:
  total_batch_size: 1
  num_txyz_samp_pts: {os.environ["NUM_TXYZ_SAMP_PTS"]}
  lr_init: {os.environ["LR_INIT"]}
  epochs: {os.environ["EPOCHS"]}
  best_metric: {os.environ["BEST_METRIC"]}

eval:
  total_batch_size: 1
  interval: {os.environ["EVAL_INTERVAL"]}
  preserve_rng_state: {os.environ["PRESERVE_EVAL_RNG_STATE"]}
  plot_num_per_type: 0
  dataset_per_type: 1

record_dir: {os.environ["RECORD_DIR"]}
wandb:
  enabled: true
  project: pdeformer-2-the-well-finetuning
  name: the-well-{os.environ["WELL_DATASET"]}-model-L
  tags: [finetune, the_well, {os.environ["WELL_DATASET"]}, model-L, pdeformer-2]
  mode: {os.environ["WANDB_MODE"]}
"""
path = Path(os.environ["CONFIG_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(text, encoding="utf-8")
print(path)
PY

echo "job_id: ${SLURM_JOB_ID:-local}"
echo "host: $(hostname)"
echo "well_base_path: ${WELL_BASE_PATH}"
echo "well_dataset: ${WELL_DATASET}"
echo "train_split: ${TRAIN_SPLIT}"
echo "test_split: ${TEST_SPLIT}"
echo "field_indices: ${FIELD_INDICES}"
echo "max_fields: ${MAX_FIELDS}"
echo "normalize_coordinates: ${NORMALIZE_COORDINATES}"
echo "normalize_time: ${NORMALIZE_TIME}"
echo "seed: ${SEED}"
echo "best_metric: ${BEST_METRIC}"
echo "preserve_eval_rng_state: ${PRESERVE_EVAL_RNG_STATE}"
echo "config_path: ${CONFIG_PATH}"
echo "record_dir: ${RECORD_DIR}"
echo "device_target: ${DEVICE_TARGET}"
echo "device_id: ${DEVICE_ID}"
echo "python_cmd: ${PYTHON_CMD}"

${PYTHON_CMD} train.py \
  --config_file_path "${CONFIG_PATH}" \
  --no_distributed \
  --device_target "${DEVICE_TARGET}" \
  --device_id "${DEVICE_ID}" \
  --mode "${MODE}"

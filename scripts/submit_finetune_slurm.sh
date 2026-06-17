#!/bin/bash
#SBATCH --job-name=pdeformer-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

# Override these at submission time, for example:
#   sbatch --export=ALL,CONFIG_PATH=configs/finetune/ins-tracer_model-M.yaml scripts/submit_finetune_slurm.sh
CONFIG_PATH="${CONFIG_PATH:-configs/finetune/pdebench-swe-rdb_model-M.yaml}"
DEVICE_TARGET="${DEVICE_TARGET:-GPU}"
DEVICE_ID="${DEVICE_ID:-0}"
MODE="${MODE:-PYNATIVE}"

# Use the repo-local uv environment when available; otherwise fall back to python.
if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="${PYTHON_CMD:-uv run python}"
else
  PYTHON_CMD="${PYTHON_CMD:-python}"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

echo "job_id: ${SLURM_JOB_ID:-local}"
echo "host: $(hostname)"
echo "config_path: ${CONFIG_PATH}"
echo "device_target: ${DEVICE_TARGET}"
echo "device_id: ${DEVICE_ID}"
echo "python_cmd: ${PYTHON_CMD}"

${PYTHON_CMD} train.py \
  --config_file_path "${CONFIG_PATH}" \
  --no_distributed \
  --device_target "${DEVICE_TARGET}" \
  --device_id "${DEVICE_ID}" \
  --mode "${MODE}"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_tokenizer_ddp.sh --gpus 4 --config pose_tokenizer/configs/default.yaml

GPUS=2
CONFIG="pose_tokenizer/configs/default.yaml"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME=""
RUN_TIMESTAMP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"; shift 2;;
    --config)
      CONFIG="$2"; shift 2;;
    --master-port)
      MASTER_PORT="$2"; shift 2;;
    --resume)
      RESUME="$2"; shift 2;;
    --run-timestamp)
      RUN_TIMESTAMP="$2"; shift 2;;
    *)
      break;;
  esac
done

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NORM_STATS_PATH="${NORM_STATS_PATH:-runs/norm_stat.json}"
export NORM_STATS_SAMPLE_LIMIT="${NORM_STATS_SAMPLE_LIMIT:-20}"
export POSE_FILE_LIST_CACHE_DIR="${POSE_FILE_LIST_CACHE_DIR:-/tmp/pose_file_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Running DDP with ${GPUS} GPU(s)"
echo "Config: ${CONFIG}"
echo "Master port: ${MASTER_PORT}"
if [[ -n "${RUN_TIMESTAMP}" ]]; then
  echo "Run timestamp: ${RUN_TIMESTAMP}"
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resume: ${RESUME}"
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "Error: torchrun not found in PATH."
  exit 1
fi

# Use torchrun for single-node multi-GPU training
TORCHRUN_CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${GPUS}"
  --master_port="${MASTER_PORT}"
  scripts/train_tokenizer.py
  --config "${CONFIG}"
)

if [[ -n "${RUN_TIMESTAMP}" ]]; then
  TORCHRUN_CMD+=(--run-timestamp "${RUN_TIMESTAMP}")
fi
if [[ -n "${RESUME}" ]]; then
  TORCHRUN_CMD+=(--resume "${RESUME}")
fi

# Pass through any additional args
if [[ $# -gt 0 ]]; then
  TORCHRUN_CMD+=("$@")
fi

"${TORCHRUN_CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

export WANDB_DISABLED=true 
export WANDB_MODE=disabled
export NORM_STATS_PATH=runs/norm_stat.json
export NORM_STATS_SAMPLE_LIMIT="${NORM_STATS_SAMPLE_LIMIT:-20}"
export POSE_FILE_LIST_CACHE_DIR="${POSE_FILE_LIST_CACHE_DIR:-/tmp/pose_file_cache}"

# Default config (can override by passing --config /path/to/yaml)
CONFIG="pose_tokenizer/configs/default.yaml"

# Optional: override config via args
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
  CONFIG="$2"
  shift 2
fi

# Output directories
EVAL_OUT="eval_out"

echo "Using config: $CONFIG"

# Ensure project root is on PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Train
python scripts/train_tokenizer.py --config "$CONFIG" "$@"

# Evaluate with best checkpoint
CKPT_PATH="checkpoints/best.pth"
if [[ ! -f "$CKPT_PATH" ]]; then
  if [[ -f "checkpoints/final.pth" ]]; then
    CKPT_PATH="checkpoints/final.pth"
    echo "best.pth not found, fallback to $CKPT_PATH"
  else
    echo "Error: neither checkpoints/best.pth nor checkpoints/final.pth exists."
    exit 1
  fi
fi

python scripts/eval_tokenizer.py --config "$CONFIG" \
  --checkpoint "$CKPT_PATH" \
  --split val \
  --output-dir "$EVAL_OUT" \
  --visualize

echo "Done. Eval outputs in $EVAL_OUT"

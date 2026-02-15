#!/usr/bin/env bash
set -euo pipefail

# Top-level convenience wrapper.
# Example:
#   bash run_tokenizer_ddp.sh --gpus 4 --config pose_tokenizer/configs/default.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

bash scripts/run_tokenizer_ddp.sh "$@"

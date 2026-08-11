#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/default.yaml}"
TASK="${TASK:-intrusion}"
SOURCE="${SOURCE:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$PROJECT_DIR"
exec "$PYTHON" -m edge_inspection run \
  --task "$TASK" \
  --source "$SOURCE" \
  --config "$CONFIG" \
  "$@"

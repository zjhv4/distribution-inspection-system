#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/default.yaml}"
TASK="${TASK:-intrusion}"
SOURCE="${SOURCE:-0}"

exec "$PYTHON" -m edge_inspection run \
  --task "$TASK" \
  --source "$SOURCE" \
  --config "$CONFIG" \
  "$@"

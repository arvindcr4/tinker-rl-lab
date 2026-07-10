#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONHASHSEED=0
exec "$PYTHON_BIN" "$DEMO_DIR/run_demo.py" "$@"


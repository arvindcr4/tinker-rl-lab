#!/bin/bash
# Run a single Tinker-Atropos experiment end-to-end with an explicit environment module.
#
# Usage:
#   ./run_experiment_generic.sh gsm8k_tinker configs/gsm8k_qwen_8b.yaml
#   ./run_experiment_generic.sh math_tinker configs/math_qwen_8b.yaml
#   ./run_experiment_generic.sh bootstrap_threshold_tinker configs/bootstrap_threshold_easy.yaml

set -e

ENV_MODULE="${1:?Usage: ./run_experiment_generic.sh <env_module> <config.yaml>}"
CONFIG="${2:?Usage: ./run_experiment_generic.sh <env_module> <config.yaml>}"
EXPERIMENT_NAME=$(basename "$CONFIG" .yaml)
LOG_DIR="./logs/${EXPERIMENT_NAME}"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Tinker RL Experiment: $EXPERIMENT_NAME"
echo "  Environment Module: $ENV_MODULE"
echo "  Config: $CONFIG"
echo "  Logs: $LOG_DIR/"
echo "============================================"

if [ -z "$TINKER_API_KEY" ]; then
    echo "ERROR: TINKER_API_KEY not set"
    echo "  export TINKER_API_KEY='your-key-here'"
    exit 1
fi

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv-at" ]; then
    source ../.venv-at/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
else
    echo "ERROR: no virtualenv found at .venv, ../.venv-at, or ../.venv"
    exit 1
fi

API_PORT=$(python - "$CONFIG" <<'PY'
import sys
from urllib.parse import urlparse

import yaml

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f) or {}

url = cfg.get("env", {}).get("rollout_server_url", "http://localhost:8000")
print(urlparse(url).port or 8000)
PY
)

cleanup() {
    echo ""
    echo "Cleaning up processes..."
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null && echo "  Stopped Atropos API (PID $API_PID)"
    [ -n "$ENV_PID" ] && kill $ENV_PID 2>/dev/null && echo "  Stopped Environment (PID $ENV_PID)"
    [ -n "$TRAINER_PID" ] && kill $TRAINER_PID 2>/dev/null && echo "  Stopped Trainer (PID $TRAINER_PID)"
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT

echo ""
echo "[1/3] Starting Atropos API server..."
run-api --port "$API_PORT" > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
sleep 3
echo "  Atropos API running on port $API_PORT (PID $API_PID)"

echo "[2/3] Starting Tinker trainer..."
python launch_training.py --config "$CONFIG" > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
sleep 5
echo "  Trainer running (PID $TRAINER_PID)"

echo "[3/3] Starting environment..."
python -m "tinker_atropos.environments.${ENV_MODULE}" serve --config "$CONFIG" \
    > "$LOG_DIR/env.log" 2>&1 &
ENV_PID=$!
echo "  Environment running (PID $ENV_PID)"

echo ""
echo "============================================"
echo "  All processes started!"
echo "  Trainer PID: $TRAINER_PID"
echo "============================================"

wait $TRAINER_PID
TRAINER_EXIT=$?

echo ""
echo "============================================"
echo "  Trainer finished (exit code: $TRAINER_EXIT)"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logs saved to: $LOG_DIR/"
echo "============================================"

exit $TRAINER_EXIT

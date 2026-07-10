#!/bin/bash
# Start a local inference server for a base model or saved checkpoint, then run
# the ToxicChat evaluation harness against it to track toxicity.
#
# Examples:
#   ./run_toxicchat_eval.sh --model Qwen/Qwen3-8B
#   ./run_toxicchat_eval.sh --model Qwen/Qwen3-8B --weights tinker://.../step_50
#   ./run_toxicchat_eval.sh --config configs/gsm8k_qwen_8b.yaml --weights tinker://...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG=""
MODEL=""
WEIGHTS=""
PORT="8002"
OUTPUT_DIR="logs/toxicchat_eval"
MAX_EXAMPLES=""
CLASSIFIER="unitary/toxic-bert"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --classifier)
            CLASSIFIER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

source .venv/bin/activate

if [[ -n "$CONFIG" && -z "$MODEL" ]]; then
    MODEL="$(python - <<PY
from tinker_atropos.config import TinkerAtroposConfig
cfg = TinkerAtroposConfig.from_yaml("$CONFIG")
print(cfg.base_model)
PY
)"
fi

if [[ -z "$MODEL" ]]; then
    echo "Provide --model, or provide --config so the model can be inferred."
    exit 1
fi

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  ToxicChat Evaluation Harness"
echo "  Model: $MODEL"
echo "  Weights: ${WEIGHTS:-<base model>}"
echo "  Port: $PORT"
echo "  Classifier: $CLASSIFIER"
echo "============================================"

SERVER_CMD=(python serve.py --model "$MODEL" --port "$PORT")
if [[ -n "$WEIGHTS" ]]; then
    SERVER_CMD+=(--weights "$WEIGHTS")
fi

"${SERVER_CMD[@]}" > "$OUTPUT_DIR/serve.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for serve.py to become healthy..."
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "serve.py did not become healthy. Check $OUTPUT_DIR/serve.log"
    exit 1
fi

EVAL_CMD=(
    python eval_toxicchat.py
    --model "$MODEL"
    --base-url "http://127.0.0.1:${PORT}/v1"
    --output-dir "$OUTPUT_DIR"
    --classifier "$CLASSIFIER"
)

if [[ -n "$CONFIG" ]]; then
    EVAL_CMD+=(--config "$CONFIG")
fi
if [[ -n "$MAX_EXAMPLES" ]]; then
    EVAL_CMD+=(--max-examples "$MAX_EXAMPLES")
fi

"${EVAL_CMD[@]}"

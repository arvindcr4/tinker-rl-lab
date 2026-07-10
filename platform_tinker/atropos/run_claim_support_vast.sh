#!/bin/bash

set -euo pipefail

SUITE="${1:-core}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-/workspace/venvs/atropos}"
PYTHON_BIN="${VENV_PATH}/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Missing Python interpreter at ${PYTHON_BIN}"
    echo "Create the venv first, then rerun this script."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN must be set to download/push models."
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
else
    unset WANDB_MODE
fi

export ATROPOS_USE_UNSLOTH=0
export HF_PUSH="${HF_PUSH:-1}"
export HF_PUSH_PRIVATE="${HF_PUSH_PRIVATE:-1}"

if [ -z "${HF_REPO_OWNER:-}" ]; then
    HF_REPO_OWNER="$("$PYTHON_BIN" - <<'PY'
from huggingface_hub import HfApi
import os
print(HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"])
PY
)"
    export HF_REPO_OWNER
fi

mkdir -p logs

run_one() {
    local config="$1"
    local seed_arg=()
    local base_name
    base_name="$(basename "$config" .yaml)"

    case "$base_name" in
        gsm8k_qwen_8b_seed1)
            seed_arg=(--seed 137)
            ;;
        gsm8k_qwen_8b_seed2)
            seed_arg=(--seed 271)
            ;;
        gsm8k_qwen_8b_seed3)
            seed_arg=(--seed 314)
            ;;
    esac

    echo ""
    echo "============================================================"
    echo "  START ${base_name}"
    echo "============================================================"

    "$PYTHON_BIN" train_grpo_unsloth.py --config "$config" "${seed_arg[@]}" \
        2>&1 | tee "logs/${base_name}.log"
}

case "$SUITE" in
    core)
        run_one "configs/gsm8k_qwen_4b.yaml"
        run_one "configs/gsm8k_qwen_8b_no_prefix.yaml"
        run_one "configs/gsm8k_qwen_8b_seed1.yaml"
        ;;
    remaining_core)
        run_one "configs/gsm8k_qwen_8b_no_prefix.yaml"
        run_one "configs/gsm8k_qwen_8b_seed1.yaml"
        ;;
    scaling)
        run_one "configs/gsm8k_qwen_4b.yaml"
        run_one "configs/gsm8k_qwen_8b.yaml"
        run_one "configs/gsm8k_qwen_14b.yaml"
        ;;
    controls)
        run_one "configs/gsm8k_qwen_8b_seed1.yaml"
        run_one "configs/gsm8k_qwen_8b_seed2.yaml"
        run_one "configs/gsm8k_qwen_8b_seed3.yaml"
        run_one "configs/gsm8k_qwen_8b_no_prefix.yaml"
        ;;
    all)
        "$0" scaling
        "$0" controls
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Expected one of: core, remaining_core, scaling, controls, all"
        exit 1
        ;;
esac

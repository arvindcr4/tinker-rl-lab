#!/bin/bash
# Run the experiment blocks needed to defend stronger reasoning claims.
#
# Usage:
#   ./run_claim_support_train.sh scaling
#   ./run_claim_support_train.sh controls
#   ./run_claim_support_train.sh hard_math
#   ./run_claim_support_train.sh all

set -e

SUITE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_block() {
    local label="$1"
    shift
    echo ""
    echo "============================================================"
    echo "  Block: $label"
    echo "============================================================"
    while [ "$#" -gt 0 ]; do
        local env_module="$1"
        local config="$2"
        shift 2
        ./run_experiment_generic.sh "$env_module" "$config"
        echo ""
        echo "--- Waiting 10s before next experiment ---"
        sleep 10
    done
}

case "$SUITE" in
    scaling)
        run_block "Matched Qwen GSM8K Scaling" \
            "gsm8k_tinker" "configs/gsm8k_qwen_4b.yaml" \
            "gsm8k_tinker" "configs/gsm8k_qwen_8b.yaml" \
            "gsm8k_tinker" "configs/gsm8k_qwen_14b.yaml"
        ;;
    controls)
        run_block "Replication / Prompt / Bootstrap Controls" \
            "gsm8k_tinker" "configs/gsm8k_qwen_8b_seed1.yaml" \
            "gsm8k_tinker" "configs/gsm8k_qwen_8b_seed2.yaml" \
            "gsm8k_tinker" "configs/gsm8k_qwen_8b_seed3.yaml" \
            "gsm8k_tinker" "configs/gsm8k_qwen_8b_no_prefix.yaml" \
            "bootstrap_threshold_tinker" "configs/bootstrap_threshold_easy.yaml" \
            "bootstrap_threshold_tinker" "configs/bootstrap_threshold_hardest.yaml"
        ;;
    hard_math)
        run_block "Harder Math Benchmarks" \
            "math_tinker" "configs/math_qwen_4b.yaml" \
            "math_tinker" "configs/math_qwen_8b.yaml" \
            "math_tinker" "configs/math_qwen_14b.yaml"
        ;;
    all)
        "$0" scaling
        "$0" controls
        "$0" hard_math
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Expected one of: scaling, controls, hard_math, all"
        exit 1
        ;;
esac

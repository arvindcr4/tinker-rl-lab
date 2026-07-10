#!/bin/bash
# 10x Structural Ceiling — Master Launch Script
# Runs all experiment blocks sequentially (Tinker single-GPU constraint)
# Each run: ~$1.90 on Tinker, total: ~$95-120 for 50 runs
#
# Usage:
#   ./run_all.sh           # Run all blocks
#   ./run_all.sh block_b   # Run only block B configs
#   ./run_all.sh --dry-run # Dry run all

set -euo pipefail
cd "$(dirname "$0")"

FILTER="${1:-}"
DRY_RUN=""
if [[ "$FILTER" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    FILTER=""
fi

CONFIGS_DIR="./configs"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Priority order: blocks that produce the most impactful results first
BLOCK_ORDER=(
    "block_b"  # Model family isolation (breaks Qwen confound)
    "block_g"  # Group saturation ablation (novel diagnostic)
    "block_c"  # Size ladder (scaling curve)
    "block_f"  # Constrained decoding (decoder confound)
    "block_h"  # Benchmark transfer (MATH, HumanEval)
    "block_d"  # PPO baseline
    "block_e"  # DPO baseline
    "block_i"  # LR sweep
    "block_j"  # Tool-use cross-family
    "block_a"  # Multi-seed replication (statistical power)
)

echo "============================================================"
echo "  10x Structural Ceiling — Experiment Matrix"
echo "  Total configs: $(ls $CONFIGS_DIR/*.yaml 2>/dev/null | wc -l)"
echo "  Estimated cost: ~\$1.90/run on Tinker"
echo "  Filter: ${FILTER:-all}"
echo "============================================================"
echo ""

RUN_COUNT=0
FAIL_COUNT=0

for block in "${BLOCK_ORDER[@]}"; do
    if [[ -n "$FILTER" && "$block" != *"$FILTER"* ]]; then
        continue
    fi

    configs=($CONFIGS_DIR/${block}_*.yaml)
    if [[ ${#configs[@]} -eq 0 ]]; then
        continue
    fi

    echo "--- Block: $block (${#configs[@]} configs) ---"

    for cfg in "${configs[@]}"; do
        name=$(basename "$cfg" .yaml)
        logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

        echo -n "  Running: $name ... "
        RUN_COUNT=$((RUN_COUNT + 1))

        if python grpo_10x_runner.py --config "$cfg" $DRY_RUN 2>&1 | tee "$logfile"; then
            echo "DONE"
        else
            echo "FAILED (see $logfile)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        echo ""
    done
done

echo "============================================================"
echo "  Completed: $RUN_COUNT runs, $FAIL_COUNT failures"
echo "  Logs: $LOG_DIR/"
echo "============================================================"

#!/bin/bash
# Rerun all experiments that were incomplete (< 50 steps) when credits ran out.
# Uses upgraded runner with checkpoint_utils for proper state saving.
# If interrupted again, rerun this script — it auto-resumes from last checkpoint.
#
# Estimated cost: ~$2/run × 12 runs = ~$24
# Priority: highest-impact experiments first

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# All experiments that ran partial (< 50 steps) in the first round.
# Ordered by paper impact.
CONFIGS=(
    # Block G — Group saturation ablation (novel diagnostic)
    "configs/block_g_gsm8k_group64.yaml"          # G=64: only 29/50 steps
    "configs/block_g_gsm8k_group32.yaml"          # G=32: only 35/50 steps
    "configs/block_g_gsm8k_group4.yaml"           # G=4: 48/50 steps

    # Block C — Size ladder
    "configs/block_c_gsm8k_qwen_0_6b.yaml"       # 0.6B: partial
    "configs/block_c_gsm8k_qwen_1_7b.yaml"       # 1.7B: partial

    # Block H — Benchmark transfer
    "configs/block_h_math_qwen8b.yaml"            # MATH-500: 37/50 steps
    "configs/block_h_humaneval_qwen8b.yaml"       # HumanEval: 37/50 steps

    # Block I — LR sweep
    "configs/block_i_gsm8k_lr_1e5.yaml"           # LR=1e-5: 36/50 steps
    "configs/block_i_gsm8k_lr_3e4.yaml"           # LR=3e-4: 37/50 steps

    # Block A — Multi-seed statistical power
    "configs/block_a_gsm8k_qwen8b_seed4.yaml"    # seed4: 37/50 steps
    "configs/block_a_gsm8k_qwen8b_seed5.yaml"    # seed5: partial
)

echo "============================================================"
echo "  10x Structural Ceiling — Incomplete Experiment Rerun"
echo "  Total configs: ${#CONFIGS[@]}"
echo "  Estimated cost: ~\$2/run on Tinker"
echo "  All runs start fresh (no prior state checkpoints)"
echo "  State checkpoints saved every 10 steps for resumption"
echo "============================================================"
echo ""

RUN_COUNT=0
FAIL_COUNT=0

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/v2_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "$(date '+%H:%M:%S') Running: $name"
    RUN_COUNT=$((RUN_COUNT + 1))

    if python -u grpo_10x_runner.py --config "$cfg" 2>&1 | tee "$logfile"; then
        echo "$(date '+%H:%M:%S') DONE: $name ✓"
    else
        echo "$(date '+%H:%M:%S') FAILED: $name ✗ (see $logfile)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "============================================================"
echo "  Completed: $RUN_COUNT runs, $FAIL_COUNT failures"
echo "  Logs: $LOG_DIR/"
echo "============================================================"

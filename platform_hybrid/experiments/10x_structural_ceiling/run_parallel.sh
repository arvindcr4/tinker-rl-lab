#!/bin/bash
# Run remaining experiments IN PARALLEL (each gets its own Tinker GPU session).
# Skips block_g_gsm8k_group64 which is already running in the sequential batch.
#
# Estimated cost: ~$2/run × 10 runs = ~$20
# Estimated time: ~40 min (longest single run, all parallel)

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "configs/block_g_gsm8k_group32.yaml"
    "configs/block_g_gsm8k_group4.yaml"
    "configs/block_c_gsm8k_qwen_0_6b.yaml"
    "configs/block_c_gsm8k_qwen_1_7b.yaml"
    "configs/block_h_math_qwen8b.yaml"
    "configs/block_h_humaneval_qwen8b.yaml"
    "configs/block_i_gsm8k_lr_1e5.yaml"
    "configs/block_i_gsm8k_lr_3e4.yaml"
    "configs/block_a_gsm8k_qwen8b_seed4.yaml"
    "configs/block_a_gsm8k_qwen8b_seed5.yaml"
)

echo "============================================================"
echo "  10x Structural Ceiling — PARALLEL Experiment Launch"
echo "  Total configs: ${#CONFIGS[@]}"
echo "  All launched simultaneously"
echo "============================================================"
echo ""

PIDS=()

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/par_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "  Launching: $name -> $logfile"
    python -u grpo_10x_runner.py --config "$cfg" > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "  All ${#CONFIGS[@]} experiments launched. PIDs: ${PIDS[*]}"
echo "  Waiting for all to complete..."
echo ""

FAIL_COUNT=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    cfg=${CONFIGS[$i]}
    name=$(basename "$cfg" .yaml)
    if wait "$pid"; then
        echo "  ✓ $name (PID $pid) completed"
    else
        echo "  ✗ $name (PID $pid) FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "============================================================"
echo "  All done. Failures: $FAIL_COUNT / ${#CONFIGS[@]}"
echo "============================================================"

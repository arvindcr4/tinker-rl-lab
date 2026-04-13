#!/bin/bash
# =============================================================================
# Multi-Seed Experiment Runner
# =============================================================================
# Runs an experiment command with multiple seeds for statistical validity.
#
# Usage:
#   ./scripts/run_seeds.sh "python experiment.py" [seed1 seed2 ...]
#
# Default seeds: 42 123 456 789 1024
# =============================================================================

set -euo pipefail

COMMAND="${1:?Usage: ./run_seeds.sh \"command\" [seed1 seed2 ...]}"
shift

# Default seeds if none provided
if [ $# -eq 0 ]; then
    SEEDS=(42 123 456 789 1024)
else
    SEEDS=("$@")
fi

echo "============================================"
echo "Multi-Seed Runner"
echo "Command: ${COMMAND}"
echo "Seeds: ${SEEDS[*]}"
echo "============================================"

TOTAL=${#SEEDS[@]}
COMPLETED=0
FAILED=0

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running seed ${SEED} ($(( COMPLETED + 1 ))/${TOTAL}) ---"
    
    START_TIME=$(date +%s)
    
    if eval "${COMMAND} --seed ${SEED}"; then
        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo "Seed ${SEED} completed in ${ELAPSED}s"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "ERROR: Seed ${SEED} failed!"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "Results: ${COMPLETED} completed, ${FAILED} failed out of ${TOTAL}"
echo "============================================"

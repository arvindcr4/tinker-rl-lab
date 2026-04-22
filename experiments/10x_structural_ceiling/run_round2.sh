#!/bin/bash
# Round 2: Five experiments from ChatGPT Pro NeurIPS analysis
# Budget: ~$200 total across all experiments
#
# Exp 1: 300-step continuation (3 runs) — ~$40-60
# Exp 2: REINFORCE baseline (1 run)     — ~$20-30
# Exp 3: Causal GU phase diagram (3 runs) — ~$60-80
# Exp 4: Code dense reward (1 run)      — ~$20-40
#
# Total: 8 runs, estimated ~$150-200

set -euo pipefail
cd "$(dirname "$0")"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running this script.}"

LOG_DIR="./logs/round2"
mkdir -p "$LOG_DIR"

RUNNER="python -u round2_runner.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  Round 2: NeurIPS Extension Experiments"
echo "  Timestamp: $TIMESTAMP"
echo "  W&B Project: tinker-structural-ceiling"
echo "============================================================"

# ── Experiment 1: 300-step continuations (resume from round 1 checkpoints) ──
echo ""
echo "[Exp 1a] LR=1e-5 continuation to 300 steps..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_continuation_lr1e5_300.yaml \
    --resume \
    2>&1 | tee "$LOG_DIR/exp1a_lr1e5_300_${TIMESTAMP}.log" &
PID_1A=$!

echo "[Exp 1b] MATH-500 continuation to 300 steps..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_continuation_math500_300.yaml \
    --resume \
    2>&1 | tee "$LOG_DIR/exp1b_math500_300_${TIMESTAMP}.log" &
PID_1B=$!

echo "[Exp 1c] HumanEval continuation to 200 steps..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_continuation_humaneval_200.yaml \
    --resume \
    2>&1 | tee "$LOG_DIR/exp1c_humaneval_200_${TIMESTAMP}.log" &
PID_1C=$!

# ── Experiment 2: REINFORCE baseline ──
echo "[Exp 2] REINFORCE baseline on GSM8K..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_reinforce_gsm8k.yaml \
    2>&1 | tee "$LOG_DIR/exp2_reinforce_${TIMESTAMP}.log" &
PID_2=$!

# Wait for initial experiments before starting phase diagram
# (phase diagram needs pre-sampling which is expensive)
echo ""
echo "Waiting for Exp 1-2 to complete before starting phase diagram..."
echo "  PIDs: $PID_1A $PID_1B $PID_1C $PID_2"

# ── Experiment 3: Phase diagram (run sequentially — shares model instance) ──
# Start phase diagram experiments in parallel after a brief delay
sleep 5

echo ""
echo "[Exp 3a] Phase diagram — EASY bin..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_phase_easy.yaml \
    2>&1 | tee "$LOG_DIR/exp3a_phase_easy_${TIMESTAMP}.log" &
PID_3A=$!

echo "[Exp 3b] Phase diagram — MID bin (Goldilocks zone)..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_phase_mid.yaml \
    2>&1 | tee "$LOG_DIR/exp3b_phase_mid_${TIMESTAMP}.log" &
PID_3B=$!

echo "[Exp 3c] Phase diagram — HARD bin..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_phase_hard.yaml \
    2>&1 | tee "$LOG_DIR/exp3c_phase_hard_${TIMESTAMP}.log" &
PID_3C=$!

# ── Experiment 4: Dense code reward ──
echo "[Exp 4] HumanEval dense reward..."
PYTHONUNBUFFERED=1 $RUNNER \
    --config configs/round2_dense_code.yaml \
    2>&1 | tee "$LOG_DIR/exp4_dense_code_${TIMESTAMP}.log" &
PID_4=$!

echo ""
echo "============================================================"
echo "  All 8 experiments launched!"
echo "  PIDs: $PID_1A $PID_1B $PID_1C $PID_2 $PID_3A $PID_3B $PID_3C $PID_4"
echo "  Logs: $LOG_DIR/"
echo "  W&B: https://wandb.ai/arvindcr4-pes-university/tinker-structural-ceiling"
echo "============================================================"

# Wait for all
wait $PID_1A $PID_1B $PID_1C $PID_2 $PID_3A $PID_3B $PID_3C $PID_4

echo ""
echo "============================================================"
echo "  Round 2 COMPLETE — all 8 experiments finished"
echo "  Check W&B: https://wandb.ai/arvindcr4-pes-university/tinker-structural-ceiling"
echo "  Logs: $LOG_DIR/"
echo "============================================================"

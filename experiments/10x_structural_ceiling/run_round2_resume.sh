#!/bin/bash
# Round 2 RESUME: Continue experiments after credit recharge
# Phase diagram: pre-bin once, then train on each bin (saves ~$40 in redundant sampling)
set -euo pipefail
cd "$(dirname "$0")"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running this script.}"

LOG_DIR="./logs/round2"
mkdir -p "$LOG_DIR"
RUNNER="python -u round2_runner.py"
TS=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  Round 2 RESUME — $TS"
echo "============================================================"

# ── Step 1: Pre-bin GSM8K prompts (run once, ~$8-12) ──
if [ ! -f "./phase_bins.json" ]; then
    echo "[Pre-bin] Classifying 500 GSM8K prompts by difficulty..."
    PYTHONUNBUFFERED=1 python -u prebin_gsm8k.py 2>&1 | tee "$LOG_DIR/prebin_${TS}.log"
    echo "[Pre-bin] Done. Bins saved to phase_bins.json"
else
    echo "[Pre-bin] phase_bins.json already exists — skipping"
fi

echo ""

# ── Step 2: Resume all experiments in parallel ──

# Exp 1a: LR=1e-5 continuation (resume from step 85)
echo "[Exp 1a] Resuming LR=1e-5 continuation (step 85→300)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_continuation_lr1e5_300.yaml --resume \
    2>&1 | tee "$LOG_DIR/exp1a_resume_${TS}.log" &
PID_1A=$!

# Exp 1b: MATH-500 continuation (resume from step 85)
echo "[Exp 1b] Resuming MATH-500 continuation (step 85→300)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_continuation_math500_300.yaml --resume \
    2>&1 | tee "$LOG_DIR/exp1b_resume_${TS}.log" &
PID_1B=$!

# Exp 1c: HumanEval continuation (resume from step 84)
echo "[Exp 1c] Resuming HumanEval continuation (step 84→200)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_continuation_humaneval_200.yaml --resume \
    2>&1 | tee "$LOG_DIR/exp1c_resume_${TS}.log" &
PID_1C=$!

# Exp 2: REINFORCE baseline (resume from step 35)
echo "[Exp 2] Resuming REINFORCE baseline (step 35→50)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_reinforce_gsm8k.yaml --resume \
    2>&1 | tee "$LOG_DIR/exp2_resume_${TS}.log" &
PID_2=$!

# Exp 3a-c: Phase diagram (fresh start with pre-computed bins)
echo "[Exp 3a] Phase diagram — EASY bin (100 steps)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_phase_easy.yaml \
    2>&1 | tee "$LOG_DIR/exp3a_resume_${TS}.log" &
PID_3A=$!

echo "[Exp 3b] Phase diagram — MID bin (100 steps)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_phase_mid.yaml \
    2>&1 | tee "$LOG_DIR/exp3b_resume_${TS}.log" &
PID_3B=$!

echo "[Exp 3c] Phase diagram — HARD bin (100 steps)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_phase_hard.yaml \
    2>&1 | tee "$LOG_DIR/exp3c_resume_${TS}.log" &
PID_3C=$!

# Exp 4: Dense code reward (resume from step 35)
echo "[Exp 4] Resuming dense code reward (step 35→100)..."
PYTHONUNBUFFERED=1 $RUNNER --config configs/round2_dense_code.yaml --resume \
    2>&1 | tee "$LOG_DIR/exp4_resume_${TS}.log" &
PID_4=$!

echo ""
echo "============================================================"
echo "  8 experiments running — PIDs: $PID_1A $PID_1B $PID_1C $PID_2 $PID_3A $PID_3B $PID_3C $PID_4"
echo "  Logs: $LOG_DIR/"
echo "  W&B: https://wandb.ai/arvindcr4-pes-university/tinker-structural-ceiling"
echo "============================================================"

wait $PID_1A $PID_1B $PID_1C $PID_2 $PID_3A $PID_3B $PID_3C $PID_4

echo ""
echo "============================================================"
echo "  Round 2 COMPLETE — all experiments finished"
echo "============================================================"

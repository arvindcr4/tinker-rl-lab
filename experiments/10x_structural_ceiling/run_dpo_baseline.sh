#!/bin/bash
# Block E: DPO baseline using tinker_cookbook recipe
# Purpose: Offline RL alternative — does DPO succeed where GRPO doesn't on GSM8K?
#
# Uses tinker_cookbook.recipes.preference.dpo.train directly
# Estimated cost: ~$2-4 on Tinker

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/block_e_dpo_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "  Block E: DPO Baseline — Qwen3-8B on GSM8K"
echo "  Using tinker_cookbook.recipes.preference.dpo.train"
echo "  Log: $LOGFILE"
echo "============================================================"

PYTHONUNBUFFERED=1 python -u -m tinker_cookbook.recipes.preference.dpo.train \
    log_path="./checkpoints/10x/gsm8k_qwen8b_dpo/" \
    model_name="Qwen/Qwen3-8B" \
    dataset=gsm8k \
    renderer_name=qwen \
    learning_rate=2e-5 \
    dpo_beta=0.1 \
    max_steps=50 \
    wandb_project=tinker-structural-ceiling \
    wandb_name="block_e_dpo_qwen8b_gsm8k" \
    behavior_if_log_dir_exists=delete \
    2>&1 | tee "$LOGFILE"

echo ""
echo "DPO baseline complete. Log: $LOGFILE"

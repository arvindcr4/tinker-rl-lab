#!/bin/bash
# run_one.sh — runs a single experiment defined by MODEL_CONFIG env var.
# All credentials via env: WANDB_API_KEY, HF_TOKEN, MODEL_CONFIG
# Usage: MODEL_CONFIG=configs/gsm8k_qwen_0_6b.yaml bash run_one.sh
set -euo pipefail
exec >> /root/experiment.log 2>&1

echo "=== $(date) run_one.sh START | MODEL_CONFIG=${MODEL_CONFIG:-NOT SET} ==="
nvidia-smi | head -3

# ── install deps (idempotent) ─────────────────────────────────────────────────
python3 -c "import unsloth" 2>/dev/null || {
    echo "=== $(date) installing unsloth ===";
    pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -3;
}
python3 -c "import trl" 2>/dev/null || pip install -q "trl>=0.9.0" peft>=0.12.0 accelerate>=0.33.0 2>&1 | tail -2
python3 -c "import math_verify" 2>/dev/null || pip install -q math-verify latex2sympy2-extended datasets wandb pyyaml huggingface_hub 2>&1 | tail -2
echo "=== $(date) deps ready ==="

# ── clone / update repo ───────────────────────────────────────────────────────
if [ ! -d /root/trl-lab ]; then
    git clone --depth 1 https://github.com/pes-llm-research/tinker-rl-lab.git /root/trl-lab
else
    git -C /root/trl-lab pull --rebase 2>&1 | tail -2
fi
cd /root/trl-lab/atropos
echo "=== $(date) repo ready ==="

# ── auth ──────────────────────────────────────────────────────────────────────
python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

export ATROPOS_USE_UNSLOTH=1
export HF_PUSH=1
export HF_PUSH_PRIVATE=1
export WANDB_SILENT=false

echo "=== $(date) STARTING: ${MODEL_CONFIG} ==="
python3 train_grpo_unsloth.py \
    --config "${MODEL_CONFIG}" \
    --seed 42 \
    && echo "=== $(date) DONE: ${MODEL_CONFIG} ===" \
    || echo "=== $(date) FAILED: ${MODEL_CONFIG} ==="

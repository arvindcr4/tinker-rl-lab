#!/bin/bash
# run_coding.sh — runs humaneval + tool_use experiments on one instance.
# Env: WANDB_API_KEY, HF_TOKEN, ATROPOS_USE_UNSLOTH=1, HF_PUSH=1
set -euo pipefail
exec >> /root/coding_experiments.log 2>&1

echo "=== $(date) run_coding.sh START ==="
nvidia-smi | head -3

pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -2
pip install -q "trl>=0.9.0" peft>=0.12.0 accelerate>=0.33.0 2>&1 | tail -1
pip install -q math-verify latex2sympy2-extended datasets wandb pyyaml huggingface_hub 2>&1 | tail -1

[ -d /root/trl-lab ] && git -C /root/trl-lab pull --rebase 2>&1 | tail -1 \
  || git clone --depth 1 https://github.com/pes-llm-research/tinker-rl-lab.git /root/trl-lab
cd /root/trl-lab/atropos

python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"

export ATROPOS_USE_UNSLOTH=1 HF_PUSH=1 HF_PUSH_PRIVATE=1 WANDB_MODE=online

echo "=== $(date) HumanEval Qwen3-8B ==="
python3 train_grpo_humaneval.py \
  --config configs/humaneval_qwen_8b.yaml --task humaneval --seed 42 \
  && echo "=== $(date) DONE humaneval ===" || echo "=== $(date) FAILED humaneval ==="

echo "=== $(date) Tool-use Qwen3-8B ==="
python3 train_grpo_humaneval.py \
  --config configs/tool_use_qwen_8b.yaml --task tool_use --seed 42 \
  && echo "=== $(date) DONE tool_use_8b ===" || echo "=== $(date) FAILED tool_use_8b ==="

echo "=== $(date) Tool-use Qwen2.5-0.5B ==="
python3 train_grpo_humaneval.py \
  --config configs/tool_use_qwen_0_5b.yaml --task tool_use --seed 42 \
  && echo "=== $(date) DONE tool_use_0.5b ===" || echo "=== $(date) FAILED tool_use_0.5b ==="

echo "=== $(date) ALL CODING EXPERIMENTS DONE ==="

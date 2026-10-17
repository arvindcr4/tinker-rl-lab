#!/bin/bash
# run_coding.sh — runs humaneval + tool_use experiments on one instance.
# Env: WANDB_API_KEY, HF_TOKEN, ATROPOS_USE_UNSLOTH=1, HF_PUSH=1
set -euo pipefail
overall_status=0
exec >> /root/coding_experiments.log 2>&1

echo "=== $(date) run_coding.sh START ==="
nvidia-smi | head -3

pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -2
pip install -q "trl>=0.9.0" 'peft>=0.12.0' 'accelerate>=0.33.0' 2>&1 | tail -1
pip install -q math-verify latex2sympy2-extended datasets wandb pyyaml huggingface_hub 2>&1 | tail -1

[ -d /root/trl-lab ] && git -C /root/trl-lab pull --rebase 2>&1 | tail -1 \
  || git clone --depth 1 https://github.com/pes-llm-research/tinker-rl-lab.git /root/trl-lab
cd /root/trl-lab/platform_tinker/atropos

python3 -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"])'
python3 -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])'

export ATROPOS_USE_UNSLOTH=1 HF_PUSH=1 HF_PUSH_PRIVATE=1 WANDB_MODE=online

echo "=== $(date) HumanEval Qwen3-8B ==="
python3 train_grpo_humaneval.py \
  --config configs/humaneval_qwen_8b.yaml --task humaneval --seed 42 \
  && echo "=== $(date) DONE humaneval ===" || { job_status=$?; echo "=== $(date) FAILED humaneval ==="; overall_status=$job_status; }

echo "=== $(date) Tool-use Qwen3-8B ==="
python3 train_grpo_humaneval.py \
  --config configs/tool_use_qwen_8b.yaml --task tool_use --seed 42 \
  && echo "=== $(date) DONE tool_use_8b ===" || { job_status=$?; echo "=== $(date) FAILED tool_use_8b ==="; overall_status=$job_status; }

echo "=== $(date) Tool-use Qwen2.5-0.5B ==="
python3 train_grpo_humaneval.py \
  --config configs/tool_use_qwen_0_5b.yaml --task tool_use --seed 42 \
  && echo "=== $(date) DONE tool_use_0.5b ===" || { job_status=$?; echo "=== $(date) FAILED tool_use_0.5b ==="; overall_status=$job_status; }

echo "=== $(date) ALL CODING EXPERIMENTS DONE ==="

exit "$overall_status"

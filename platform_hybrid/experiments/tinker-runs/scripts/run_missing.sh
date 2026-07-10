#!/bin/bash
# Run missing experiments sequentially (one at a time) to avoid API hangs
# Credentials are loaded from environment variables or .env file
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_MODE=online
export TINKER_API_KEY="${TINKER_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"

cd /home/user/workspace/tinker-rl-lab
python3 experiments/tinker-runs/scripts/tinker_parallel_runner.py --max-parallel 2 2>&1
echo "RUNNER_EXIT_CODE=$?"

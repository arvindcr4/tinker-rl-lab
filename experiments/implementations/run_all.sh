#!/bin/bash
# Tinker RL Implementations - Run All Experiments
# ================================================

set -e

echo "=========================================="
echo "Tinker RL Implementations"
echo "=========================================="

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "Consider running: python -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo ""
echo "--- Installing Dependencies ---"
pip install -r requirements.txt

# Function to run experiment with error handling
run_experiment() {
    local name=$1
    local script=$2

    echo ""
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="

    if python "$script"; then
        echo "✅ $name completed successfully"
    else
        echo "❌ $name failed (continuing...)"
    fi
}

# Run experiments in order of complexity

echo ""
echo "=========================================="
echo "Part 1: TRL Experiments (LLM Training)"
echo "=========================================="

# TRL GRPO Math (simplest, fastest)
run_experiment "TRL GRPO Math (Arithmetic)" "trl_grpo_math.py"

# TRL Chat SFT
run_experiment "TRL Chat SFT (NoRobots)" "trl_chat_sft.py"

# TRL DPO Shorter
run_experiment "TRL DPO Shorter Responses" "trl_dpo_shorter.py"

# TRL GSM8K (harder, slower)
run_experiment "TRL GRPO GSM8K (Word Problems)" "trl_gsm8k_math.py"

# TRL Distillation
run_experiment "TRL Distillation (On/Off-Policy)" "trl_distillation.py"

echo ""
echo "=========================================="
echo "Part 2: Classic RL Experiments"
echo "=========================================="

# Stable Baselines3
run_experiment "Stable Baselines3 PPO" "sb3_ppo_math.py"

# CleanRL
run_experiment "CleanRL PPO" "cleanrl_ppo_math.py"

# Tianshou
run_experiment "Tianshou PPO" "tianshou_ppo_math.py"

echo ""
echo "=========================================="
echo "Part 3: Advanced/High-Performance RL"
echo "=========================================="

# PufferLib
run_experiment "PufferLib (Config Only)" "pufferlib_math.py"

# rl_games
run_experiment "rl_games NVIDIA (Config Only)" "rl_games_math.py"

# d3rlpy Offline RL
run_experiment "d3rlpy Offline RL" "d3rlpy_offline.py"

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - ./grpo_math_final/"
echo "  - ./chat_sft_final/"
echo "  - ./dpo_shorter_final/"
echo "  - ./grpo_gsm8k_final/"
echo "  - ./distillation_*_final/"
echo "  - ./sb3_math_ppo_final.zip"
echo "  - ./cleanrl_math_agent.pt"
echo "  - ./tianshou_ppo_math.pt"
echo "  - ./*.d3 (d3rlpy models)"

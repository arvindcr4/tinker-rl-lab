#!/bin/bash
# Tinker RL Training Script

# Activate virtual environment
source ~/tinker-env/bin/activate

# Set API key
export TINKER_API_KEY="${TINKER_API_KEY:?Set TINKER_API_KEY environment variable}"

# Run arithmetic RL training
python -m tinker_cookbook.recipes.math_rl.train \
  model_name="meta-llama/Llama-3.2-1B" \
  group_size=4 \
  groups_per_batch=100 \
  learning_rate=1e-4

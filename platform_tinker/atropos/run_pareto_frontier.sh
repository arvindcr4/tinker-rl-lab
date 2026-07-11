#!/bin/bash
# Pre-registered Group-size Pareto frontier ablation (P1)
# Evaluates G ∈ {4, 8, 16, 32} under a strict total token budget.

set -e

CONFIG="${1:?Usage: ./run_pareto_frontier.sh <config.yaml>}"
TOKEN_BUDGET=${2:-200000000}
BASE_BATCH_SIZE=${3:-128}

echo "============================================"
echo "  P1: Group-size Pareto Frontier"
echo "  Config: $CONFIG"
echo "  Total Token Budget: $TOKEN_BUDGET"
echo "  Total Batch Size: $BASE_BATCH_SIZE"
echo "============================================"

# Ensure wandb key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set"
    exit 1
fi

source .venv/bin/activate 2>/dev/null || true

for G in 4 8 16 32; do
    echo "--------------------------------------------"
    echo "  Running Group Size = $G"
    echo "--------------------------------------------"
    
    # We maintain a constant total batch size (sequences per step)
    # This means the number of unique prompts per step = BASE_BATCH_SIZE / G
    # Ensure it's perfectly divisible, or TRL might complain.
    # Note: 128 is divisible by 4, 8, 16, 32.

    python3 train_grpo_unsloth.py \
        --config "$CONFIG" \
        --total_token_budget "$TOKEN_BUDGET" \
        --group_size "$G" \
        --batch_size "$BASE_BATCH_SIZE" \
        --seed 42
    
    echo "Finished G = $G"
done

echo "Pareto frontier ablation complete."

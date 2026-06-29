#!/usr/bin/env bash
set -euo pipefail

# AI Scientist launcher for GRPO GSM8K research
# Usage: ./run_ai_scientist.sh [num_ideas] [model]
#
# Prerequisites:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."   # Required for review stage

NUM_IDEAS="${1:-2}"
MODEL="${2:-claude-sonnet-4-6}"

# Validate API keys
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set — review stage will fail"
    echo "Run: export OPENAI_API_KEY='sk-...'"
fi

AI_SCIENTIST_DIR="$HOME/AI-Scientist"

if [ ! -d "$AI_SCIENTIST_DIR" ]; then
    echo "ERROR: AI Scientist not found at $AI_SCIENTIST_DIR"
    echo "Run: cd ~ && git clone https://github.com/SakanaAI/AI-Scientist.git"
    exit 1
fi

# Sync template
echo "Syncing template..."
cp -r "$(dirname "$0")/ai-scientist-template/"* "$AI_SCIENTIST_DIR/templates/grpo_gsm8k/"

cd "$AI_SCIENTIST_DIR"
source .venv/bin/activate

echo "=== AI Scientist: GRPO GSM8K ==="
echo "Model:     $MODEL"
echo "Ideas:     $NUM_IDEAS"
echo "Template:  templates/grpo_gsm8k"
echo ""

python launch_scientist.py \
    --model "$MODEL" \
    --experiment grpo_gsm8k \
    --num-ideas "$NUM_IDEAS" \
    --improvement

echo ""
echo "=== Results ==="
ls -la results/grpo_gsm8k/ 2>/dev/null || echo "No results yet"

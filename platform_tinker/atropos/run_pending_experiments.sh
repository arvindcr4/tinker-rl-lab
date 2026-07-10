#!/bin/bash
# Run all pending experiments for final report (March 28 deadline)
# Each experiment launches env + trainer in parallel, waits for completion.

set -e
export TINKER_API_KEY="${TINKER_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

run_experiment() {
    local name="$1"
    local env_module="$2"
    local config="$3"

    echo "========================================"
    echo "Starting: $name"
    echo "Config: $config"
    echo "========================================"

    export TINKER_CONFIG_PATH="$config"

    # Launch trainer (background)
    python launch_training.py --config "$config" \
        > "$LOG_DIR/${name}_trainer.log" 2>&1 &
    local trainer_pid=$!

    # Launch env (background)
    python -m "tinker_atropos.environments.${env_module}" \
        --config "$config" \
        > "$LOG_DIR/${name}_env.log" 2>&1 &
    local env_pid=$!

    echo "$name: trainer PID=$trainer_pid, env PID=$env_pid"
    echo "$trainer_pid $env_pid" > "$LOG_DIR/${name}.pids"
}

# Kill all on Ctrl+C
cleanup() {
    echo "Stopping all experiments..."
    for pidfile in "$LOG_DIR"/*.pids; do
        [ -f "$pidfile" ] && kill $(cat "$pidfile") 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- Experiment 1: Tool Use Qwen3-8B ---
run_experiment "tool_use_qwen8b" "tool_use_tinker" "configs/tool_use_qwen_8b.yaml"
sleep 30  # stagger startup to avoid port conflicts

# --- Experiment 2: HumanEval Qwen3-8B ---
run_experiment "humaneval_qwen8b" "humaneval_tinker" "configs/humaneval_qwen_8b.yaml"
sleep 30

# --- Experiment 3: Tool Use Qwen0.5B (small model capacity test) ---
run_experiment "tool_use_qwen0_5b" "tool_use_tinker" "configs/tool_use_qwen_0_5b.yaml"

echo ""
echo "All 3 experiments launched. Logs at: $LOG_DIR/"
echo "Monitor with: tail -f $LOG_DIR/tool_use_qwen8b_trainer.log"
echo ""
echo "To check progress:"
echo "  tail -f $LOG_DIR/tool_use_qwen8b_env.log"
echo "  tail -f $LOG_DIR/humaneval_qwen8b_env.log"
echo "  tail -f $LOG_DIR/tool_use_qwen0_5b_env.log"

# Wait for all background processes
wait
echo "All experiments complete."

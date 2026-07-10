#!/bin/bash
# vast_run.sh — startup script executed on the Vast.ai instance
# Runs all systematic scaling study experiments sequentially.
# Environment variables WANDB_API_KEY and HF_TOKEN are injected by vastai create.
set -euo pipefail
exec > /root/vast_run.log 2>&1   # redirect all output to log file

echo "=== $(date) | vast_run.sh starting ==="
nvidia-smi | head -8

# ── 1. Install dependencies ──────────────────────────────────────────────────
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q 'trl>=0.9.0' 'peft>=0.12.0' 'accelerate>=0.33.0'
pip install -q math-verify latex2sympy2-extended datasets wandb pyyaml
echo "=== $(date) | deps installed ==="

# ── 2. Clone repo ────────────────────────────────────────────────────────────
git clone --depth 1 https://github.com/pes-llm-research/tinker-rl-lab.git /root/tinker-rl-lab
cd /root/tinker-rl-lab/atropos
echo "=== $(date) | repo cloned ==="

# ── 3. WandB login ───────────────────────────────────────────────────────────
python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
export ATROPOS_USE_UNSLOTH=1   # enable Unsloth memory-efficient kernels
export WANDB_SILENT=false
echo "=== $(date) | wandb login OK ==="

# ── 4. Run experiments sequentially ─────────────────────────────────────────
run_exp() {
    local config="$1"
    echo ""
    echo "========================================================"
    echo "=== $(date) | STARTING: $config"
    echo "========================================================"
    python3 train_grpo_unsloth.py --config "$config" --seed 42 --wandb_key "${WANDB_API_KEY}" \
        && echo "=== $(date) | DONE: $config" \
        || echo "=== $(date) | FAILED: $config — continuing ==="
}

# Block C — smallest first (fail-fast detection)
run_exp configs/gsm8k_qwen_0_6b.yaml
run_exp configs/gsm8k_qwen_1_7b.yaml
run_exp configs/gsm8k_qwen_4b.yaml

# Block A + B anchor
run_exp configs/gsm8k_llama_8b_base.yaml

# Block C — large
run_exp configs/gsm8k_qwen_14b.yaml
run_exp configs/gsm8k_qwen_30b_moe.yaml

echo ""
echo "=== $(date) | ALL EXPERIMENTS COMPLETE ==="

# Collect all reward CSVs into one summary
python3 - <<'EOF'
import os, csv, json

results = {}
base = '/root/tinker-rl-lab/atropos/checkpoints'
for run_dir in os.listdir(base):
    csv_path = os.path.join(base, run_dir, 'reward_log.csv')
    if not os.path.exists(csv_path):
        continue
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        rewards = [float(r['mean_reward']) for r in rows]
        results[run_dir] = {
            'step0': rewards[0],
            'final': rewards[-1],
            'peak':  max(rewards),
            'n_steps': len(rewards),
            'rewards': rewards,
        }

with open('/root/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results summary saved to /root/results_summary.json')
print(json.dumps({k: {'step0': v['step0'], 'final': v['final']} for k,v in results.items()}, indent=2))
EOF

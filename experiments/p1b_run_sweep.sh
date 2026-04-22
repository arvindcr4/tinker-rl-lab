#!/usr/bin/env bash
# P1-B audit-table probe sweep launcher.
#
# Validates base-policy hit rate predictions for the rows in Table 5.1 of the
# capstone report that are reachable via Tinker's model zoo and the reward
# parsers we have in p1b_audit_probe.py.
#
# Rows covered in this sweep (6 of 10):
#   1. GSM8K Llama-8B Base             Qwen G=32
#   2. GSM8K Llama-8B Instruct         Qwen G=32
#   3. GSM8K Qwen3-8B G=16             Qwen G=16
#   4. MATH-500 Qwen3-8B               Qwen G=32
#   5. Arithmetic Llama-1B             Qwen G=16
#   6. GSM8K Qwen3.5-4B (sanity ref)   Qwen G=16
#
# Rows NOT covered here:
#   - Tool-Use: needs the project-specific JSON schema reward, not in this driver.
#   - HumanEval: needs sandbox. Blocked on P0-B.
#   - DeepSeek-V3.1: not a Tinker base model.
#
# Cost estimate: ~$10-15 for all six runs combined, ~30-45 min wall clock.

set -euo pipefail

if [[ -z "${TINKER_API_KEY:-}" ]]; then
    echo "ERROR: TINKER_API_KEY is not set." >&2
    echo "Export the ROTATED key in your shell before running this script." >&2
    echo "  export TINKER_API_KEY='<new-key-from-tinker-console>'" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

run() {
    local tag="$1"; shift
    echo "=== launching $tag ==="
    python3 experiments/p1b_audit_probe.py --tag "$tag" "$@" \
        2>&1 | tee "experiments/results/${tag}.log"
}

# 1. GSM8K Llama-8B Base (audit predicts p_x ~0.05, ZVF 0.95 -> dead)
run "p1b_gsm8k_llama8b_base_G32" \
    --model "meta-llama/Llama-3.1-8B" \
    --task gsm8k --group 32 --prompts 30

# 2. GSM8K Llama-8B Instruct (audit predicts p_x ~0.45, ZVF 0.15 -> alive)
run "p1b_gsm8k_llama8b_inst_G32" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --task gsm8k --group 32 --prompts 30

# 3. GSM8K Qwen3-8B G=16 (audit predicts p_x ~0.35, ZVF 0.10 -> alive)
run "p1b_gsm8k_qwen3_8b_G16" \
    --model "Qwen/Qwen3-8B" \
    --task gsm8k --group 16 --prompts 30

# 4. MATH-500 Qwen3-8B G=32 (audit predicts p_x ~0.15, ZVF 0.20 -> alive but hard)
run "p1b_math500_qwen3_8b_G32" \
    --model "Qwen/Qwen3-8B" \
    --task math500 --group 32 --prompts 25

# 5. Arithmetic Llama-1B G=16 (audit predicts p_x ~0.95, ZVF 0.60 -> saturating)
run "p1b_arith_llama1b_G16" \
    --model "meta-llama/Llama-3.2-1B" \
    --task arith --group 16 --prompts 30

# 6. Cross-ref: GSM8K Qwen3.5-4B G=16 (matches live_zvf_probe baseline)
run "p1b_gsm8k_qwen35_4b_G16" \
    --model "Qwen/Qwen3.5-4B" \
    --task gsm8k --group 16 --prompts 30

echo ""
echo "=== sweep complete. Summary: ==="
python3 - <<'PY'
import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob("experiments/results/p1b_*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") != "completed":
            continue
        rows.append({
            "tag": d.get("tag", ""),
            "model": d.get("model", ""),
            "task": d.get("task", ""),
            "G": d.get("group_size", ""),
            "n": d.get("n_prompts", 0),
            "mean_p_x": d.get("mean_p_x", 0.0),
            "predicted_usable": d.get("predicted_usable_group_rate", 0.0),
            "empirical_zvf": d.get("empirical_initial_zvf", 0.0),
        })
    except Exception:
        continue
print(f"{'tag':<45} {'G':>3} {'n':>3} {'mean_p_x':>9} {'pred_usable':>12} {'emp_zvf':>8}")
for r in rows:
    print(f"{r['tag']:<45} {r['G']:>3} {r['n']:>3} "
          f"{r['mean_p_x']:>9.3f} {r['predicted_usable']:>12.3f} "
          f"{r['empirical_zvf']:>8.3f}")
PY

#!/usr/bin/env bash
set -euo pipefail

: "${TINKER_API_KEY:?Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)}"

cd "$(dirname "$0")/reports/final"

# P0 Checklist: Independent Evaluation Seeds
# We test a single frozen checkpoint (e.g., the primary Qwen3-8B run)
# across multiple independent evaluation seeds (42, 43, 44).
PRIMARY_RUN_ID="899d909e-7821-5b2b-a8d9-d8b3113ebd64"

SEEDS=(
  "042:${PRIMARY_RUN_ID}"
  "043:${PRIMARY_RUN_ID}"
  "044:${PRIMARY_RUN_ID}"
)

echo "=========================================="
echo "GSM8K HELD-OUT EVALUATION — ALL 5 SEEDS"
echo "Full test set: 1,319 examples per seed"
echo "Started: $(date)"
echo "=========================================="

for entry in "${SEEDS[@]}"; do
  SEED="${entry%%:*}"
  RUN_ID="${entry#*:}"
  OUT="gsm8k_heldout_seed${SEED}.json"

  echo ""
  echo "--- Seed ${SEED} | Run ${RUN_ID} ---"
  echo "Output: ${OUT}"
  echo "Start: $(date)"

  python3 evaluate_gsm8k_test.py \
    --use_tinker \
    --run_id "${RUN_ID}" \
    --seed "${SEED}" \
    --output "${OUT}" \
    --max_tokens 2048 \
    2>&1 | tee "eval_seed${SEED}.log"

  echo "Finished seed ${SEED}: $(date)"
done

echo ""
echo "=========================================="
echo "ALL SEEDS COMPLETE: $(date)"
echo "=========================================="

# Aggregate summary
python3 -c "
import json, glob, statistics

files = sorted(glob.glob('gsm8k_heldout_seed*.json'))
accs = []
print('\n=== AGGREGATE RESULTS ===')
for f in files:
    d = json.load(open(f))
    acc = d['summary']['accuracy']
    ci = d['summary'].get('accuracy_ci_95_percent', ['?','?'])
    accs.append(acc)
    print(f'{f}: {acc:.1%}  CI: [{ci[0]}, {ci[1]}]')

if accs:
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0
    print(f'\nMean accuracy: {mean:.1%} ± {std:.1%} ({len(accs)} seeds)')
    print('Ready for paper: YES' if len(accs) == 5 else f'Only {len(accs)}/5 seeds done')
"

#!/usr/bin/env bash
set -euo pipefail

: "${TINKER_API_KEY:?Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)}"

cd /home/arvind/tinker-rl-lab/reports/final

SEEDS=(
  "042:899d909e-7821-5b2b-a8d9-d8b3113ebd64"
  "137:5db4e965-07f3-575c-a806-2578d230a30d"
  "256:aabb48cb-cb8b-5ebb-85f8-ccd6443eff00"
  "512:99971b26-f228-5726-88c7-ab92c73b72ec"
  "999:b3ba8df6-753b-545a-816d-0cbff0b059f7"
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
    print(f'\nMean accuracy: {mean:.1%} ± {std:.1%} (5 seeds)')
    print(f'Ready for paper: YES' if len(accs) == 5 else f'Only {len(accs)}/5 seeds done')
"

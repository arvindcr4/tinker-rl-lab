#!/usr/bin/env bash
set -uo pipefail

export TINKER_API_KEY="tml-lpYVuVb7Zy4op6Bc6cvivFArSb4KHff208nZIsG3QNETRETCwIxu4vDUf9YtYgHbEAAAA"
cd /home/arvind/tinker-rl-lab/reports/final

LIMIT=200

echo "=========================================="
echo "GSM8K HELD-OUT EVAL — 5 SEEDS × ${LIMIT} EXAMPLES (PARALLEL)"
echo "Started: $(date)"
echo "=========================================="

# Launch all 5 in parallel
python3 evaluate_gsm8k_test.py --use_tinker --run_id 899d909e-7821-5b2b-a8d9-d8b3113ebd64 --seed 42  --limit $LIMIT --output gsm8k_heldout_seed042.json --max_tokens 2048 2>&1 | tee eval_seed042.log &
PID1=$!

python3 evaluate_gsm8k_test.py --use_tinker --run_id 5db4e965-07f3-575c-a806-2578d230a30d --seed 137 --limit $LIMIT --output gsm8k_heldout_seed137.json --max_tokens 2048 2>&1 | tee eval_seed137.log &
PID2=$!

python3 evaluate_gsm8k_test.py --use_tinker --run_id aabb48cb-cb8b-5ebb-85f8-ccd6443eff00 --seed 256 --limit $LIMIT --output gsm8k_heldout_seed256.json --max_tokens 2048 2>&1 | tee eval_seed256.log &
PID3=$!

python3 evaluate_gsm8k_test.py --use_tinker --run_id 99971b26-f228-5726-88c7-ab92c73b72ec --seed 512 --limit $LIMIT --output gsm8k_heldout_seed512.json --max_tokens 2048 2>&1 | tee eval_seed512.log &
PID4=$!

python3 evaluate_gsm8k_test.py --use_tinker --run_id b3ba8df6-753b-545a-816d-0cbff0b059f7 --seed 999 --limit $LIMIT --output gsm8k_heldout_seed999.json --max_tokens 2048 2>&1 | tee eval_seed999.log &
PID5=$!

echo "PIDs: $PID1 $PID2 $PID3 $PID4 $PID5"
echo "Waiting for all seeds..."

wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "=========================================="
echo "ALL SEEDS COMPLETE: $(date)"
echo "=========================================="

# Aggregate
python3 -c "
import json, glob, statistics

files = sorted(glob.glob('gsm8k_heldout_seed*.json'))
accs = []
print('\n=== AGGREGATE RESULTS (200-sample held-out) ===')
for f in files:
    d = json.load(open(f))
    s = d['summary']
    acc = s['accuracy']
    ci = s.get('accuracy_ci_95_percent', ['?','?'])
    accs.append(acc)
    print(f'{f}: {acc:.1%}  ({s[\"correct\"]}/{s[\"attempted\"]})  CI: [{ci[0]}, {ci[1]}]')

if accs:
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0
    print(f'\n*** Mean accuracy: {mean:.1%} ± {std:.1%} (5 seeds, n=200 each) ***')
    print(f'Paper-ready: {\"YES\" if len(accs) == 5 else \"NO\"} ({len(accs)}/5 seeds)')
"

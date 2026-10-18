#!/bin/bash
# =============================================================================
# Causal ZVF Experiment — Run All Arms
# =============================================================================
# This script runs the causal ZVF experiment:
#   1. Bin GSM8K prompts by difficulty (once)
#   2. Run three matched GRPO arms (dead, mixed, saturated)
#   3. Evaluate on held-out GSM8K
#
# Usage:
#   bash experiments/run_causal_zvf.sh [--model Qwen/Qwen3-8B] [--seed 42]
#
# The primary success criterion is NOT held-out accuracy improvement.
# It is: mixed arm has higher first-5-step GU and positive reward slope
#        than dead and saturated arms.
# =============================================================================

set -euo pipefail

# Defaults
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SEED="${SEED:-42}"
GROUP_SIZE="${GROUP_SIZE:-8}"
STEPS="${STEPS:-30}"
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
MAX_PROMPTS="${MAX_PROMPTS:-200}"
BIN_SAMPLES="${BIN_SAMPLES:-8}"
HELDOUT_N="${HELDOUT_N:-500}"

# Parse advertised controls; never reuse an unverified legacy cache.
while (($#)); do
    case "$1" in
        --model) MODEL="${2:?missing model}"; shift 2 ;;
        --seed) SEED="${2:?missing seed}"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be nonnegative integer" >&2; exit 2; }
RUN_DIR=$(mktemp -d "${CAUSAL_OUTPUT_ROOT:-.}/causal-zvf-v3.XXXXXXXX")
BIN_DATA="$RUN_DIR/bins.json"
RESULT_DEAD="$RUN_DIR/dead.json"
RESULT_MIXED="$RUN_DIR/mixed.json"
RESULT_SATURATED="$RUN_DIR/saturated.json"
HELDOUT_OUTPUT="$RUN_DIR/heldout.json"
export RUN_DIR MODEL SEED GROUP_SIZE STEPS LR LORA_RANK MAX_PROMPTS BIN_SAMPLES HELDOUT_N
python3 - <<'RECEIPT'
import json, os, hashlib
from pathlib import Path
config = {k: os.environ[k] for k in ("MODEL", "SEED", "GROUP_SIZE", "STEPS", "LR", "LORA_RANK", "MAX_PROMPTS", "BIN_SAMPLES", "HELDOUT_N")}
config.update(protocol="prompt-pool-v3-no-cache", model_revision="unresolved", reuse_policy="never")
config["configuration_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
Path(os.environ["RUN_DIR"], "launch.json").write_text(json.dumps(config, indent=2))
RECEIPT

# W&B project (set to empty to disable)
WANDB_PROJECT="${WANDB_PROJECT:-tinker-causal-zvf}"

echo "============================================================"
echo "Causal ZVF Experiment"
echo "============================================================"
echo "Model:       $MODEL"
echo "Seed:        $SEED"
echo "Group size:  $GROUP_SIZE"
echo "Steps:       $STEPS"
echo "LR:          $LR"
echo "LoRA rank:   $LORA_RANK"
echo "Max prompts: $MAX_PROMPTS"
echo "Bin samples: $BIN_SAMPLES"
echo "============================================================"
echo ""

# Check Tinker API key
if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "ERROR: TINKER_API_KEY environment variable not set."
    echo "Run: export TINKER_API_KEY=<your-key>"
    exit 1
fi

# =============================================================================
# Phase 1: Bin prompts by difficulty
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Binning prompts by difficulty"
echo "============================================================"
echo ""

if [ -f "$BIN_DATA" ]; then
    echo "Bin data exists at $BIN_DATA — skipping binning."
    echo "To re-bin, delete $BIN_DATA first."
else
    python -m experiments.causal_zvf_experiment bin \
        --model "$MODEL" \
        --max-prompts $MAX_PROMPTS \
        --num-samples $BIN_SAMPLES \
        --output "$BIN_DATA"
fi

# Show bin summary
echo ""
echo "Bin summary:"
python3 -c "
import json
d = json.load(open('$BIN_DATA'))
for bin_name in ['dead', 'mixed', 'saturated']:
    info = d['bins'][bin_name]
    rates = info['rates']
    mean_rate = sum(rates) / max(len(rates), 1)
    print(f'  {bin_name:12s}: {info[\"count\"]:3d} prompts, mean success rate = {mean_rate:.3f}')
"

# =============================================================================
# Phase 2: Run three GRPO arms
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: Running GRPO training arms"
echo "============================================================"
echo ""

for PHASE in dead mixed saturated; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running GRPO arm: $PHASE"
    echo "------------------------------------------------------------"

    OUTPUT_VAR="RESULT_${PHASE^^}"
    OUTPUT="${!OUTPUT_VAR}"

    # Check if result already exists
    if [ -f "$OUTPUT" ]; then
        echo "Result exists at $OUTPUT — skipping (to re-run, delete the file)."
        continue
    fi

    python -m experiments.causal_zvf_experiment train \
        --phase "$PHASE" \
        --bin-data "$BIN_DATA" \
        --model "$MODEL" \
        --seed $SEED \
        --group-size $GROUP_SIZE \
        --steps $STEPS \
        --lr $LR \
        --lora-rank $LORA_RANK \
        --output "$OUTPUT"

    echo ""
done

# =============================================================================
# Phase 3: Evaluate on held-out GSM8K
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 3: Held-out GSM8K evaluation"
echo "============================================================"
echo ""

RUN_IDS=()

for PHASE in dead mixed saturated; do
    RESULT_VAR="RESULT_${PHASE^^}"
    RESULT="${!RESULT_VAR}"

    if [ ! -f "$RESULT" ]; then
        echo "ERROR: Result file not found: $RESULT"
        echo "Skipping held-out evaluation."
        exit 1
    fi

    RUN_ID=$(python3 -c "
import json
d = json.load(open('$RESULT'))
assert d.get('steps_completed') == int('$STEPS'), 'incomplete arm'
assert d['config']['model'] == '$MODEL' and d['config']['seed'] == int('$SEED'), 'arm identity mismatch'
print(d.get('sampler_checkpoint_path', ''))
")

    if [ -z "$RUN_ID" ]; then
        echo "WARNING: No run_id found in $RESULT"
        echo "Skipping this arm."
        exit 1
    fi

    RUN_IDS+=("$RUN_ID")
    echo "$PHASE arm checkpoint: $RUN_ID"
    printf '%s\t%s\n' "$PHASE" "$RUN_ID" >> "$RUN_DIR/arm-checkpoints.tsv"
done

if [ ${#RUN_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Evaluating ${#RUN_IDS[@]} checkpoints on held-out GSM8K..."
    python -m experiments.causal_zvf_experiment evaluate \
        --checkpoint-paths "${RUN_IDS[@]}" \
        --model "$MODEL" \
        --heldout-n $HELDOUT_N \
        --output "$HELDOUT_OUTPUT"
else
    echo "ERROR: No run IDs found. Cannot evaluate."
    exit 1
fi

# =============================================================================
# Phase 4: Summary
# =============================================================================
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""

python3 -c "
import json
from pathlib import Path

bin_data = json.load(open('$BIN_DATA'))

print('Bin composition:')
for bin_name in ['dead', 'mixed', 'saturated']:
    info = bin_data['bins'][bin_name]
    rates = info['rates']
    mean_rate = sum(rates) / max(len(rates), 1)
    print(f'  {bin_name:12s}: {info[\"count\"]:3d} prompts, mean base rate = {mean_rate:.3f}')

print()
print('Training results:')
for phase in ['dead', 'mixed', 'saturated']:
    result_var = f'RESULT_{phase.upper()}'
    result_path = Path('$RUN_DIR') / f'{phase}.json'
    if not result_path.exists():
        print(f'  {phase:12s}: NOT COMPLETED')
        continue
    d = json.load(open(result_path))
    slope = d.get('reward_slope', 0)
    peak = d.get('peak_reward', 0)
    last10 = d.get('last10_reward', 0)
    final_zvf = d.get('final_zvf', -1)
    final_gu = d.get('final_gu', -1)
    print(f'  {phase:12s}: slope={slope:.4f}, peak={peak:.3f}, last10={last10:.3f}, ZVF={final_zvf:.2f}, GU={final_gu:.2f}')

print()
print('Held-out evaluation:')
if Path('$HELDOUT_OUTPUT').exists():
    d = json.load(open('$HELDOUT_OUTPUT'))
    for r in d.get('results', []):
        print(f'  {r[\"run_id\"][:20]}: acc={r[\"accuracy\"]:.3f} [CI: {r[\"ci95_low\"]:.3f}, {r[\"ci95_high\"]:.3f}]')
else:
    print('  NOT COMPLETED')

print()
print('Primary success criterion:')
print('  Mixed arm should have higher first-5-step GU and positive reward slope')
print('  than dead and saturated arms.')
print()
print('  Expected pattern:')
print('    dead:      ZVF≈1.0, flat slope, low reward')
print('    mixed:     ZVF<1.0, positive slope, improving reward')
print('    saturated: ZVF≈1.0, flat slope, already-high reward')
"
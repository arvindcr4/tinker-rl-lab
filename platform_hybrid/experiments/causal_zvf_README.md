> PROTOCOL CORRECTION (review 43): This is a prompt-pool intervention. Pool assignment changes content, difficulty, headroom and length as well as ZVF; it does not isolate a causal ZVF mechanism. The v3 launcher uses fresh directories and never reuses legacy caches. Model revisions remain unresolved unless captured by the service receipt.

# Prompt-Pool Association Experiment

## Overview

This experiment compares learning signals and outcomes across prompt pools selected by estimated base success rate.

The intended training settings are held fixed, but pool assignment changes content, difficulty, headroom and length together with ZVF/GU. This is a confounded prompt-pool association, not an isolated causal test of ZVF.

## Experiment Design

### Phase 1: Bin prompts by difficulty
Sample G=8 completions from each GSM8K prompt and estimate base success rate:
- **Dead**: p̂ < 25% success (cold-start, no learning signal)
- **Mixed**: 25% ≤ p̂ ≤ 75% success (Goldilocks zone)
- **Saturated**: p̂ > 75% success (already solved, no learning signal)

### Phase 2: Run three matched GRPO arms
Train Qwen3-8B on GSM8K with identical config but different prompt pools:
```
Qwen3-8B + GSM8K + same GRPO config + dead prompts
Qwen3-8B + GSM8K + same GRPO config + mixed prompts
Qwen3-8B + GSM8K + same GRPO config + saturated prompts
```

### Phase 3: Evaluate on held-out GSM8K
Evaluate each checkpoint on a held-out GSM8K test slice.

## Descriptive Comparison

Report first-five-step GU, reward slopes and held-out accuracy by pool. Differences do not identify which pool characteristic caused the outcome.

The primary endpoint is:
```
Mixed-prompt arm has substantially higher first-5-step GU and positive reward slope
than the dead and saturated arms.
```

## Expected Results

| Arm       | First-5 ZVF | Reward slope | Last-10 reward | Held-out effect |
| --------- | ----------: | -----------: | -------------: | --------------: |
| Dead      |        ≈1.0 |         flat |            low |            none |
| Mixed     |         low |     positive |       improves |     maybe small |
| Saturated |        ≈1.0 | flat or tiny |   already high |    none / small |

That pattern would be an association consistent with multiple mechanisms; it would not establish causal ZVF effects, even with a held-out improvement.

## Usage

```bash
# Set Tinker API key
export TINKER_API_KEY=<your-key>

# Phase 1: Bin prompts
python -m experiments.causal_zvf_experiment bin \
    --model Qwen/Qwen3-8B \
    --max-prompts 200 \
    --num-samples 8 \
    --output ./causal_zvf_bins.json

# Phase 2: Run training arms
for PHASE in dead mixed saturated; do
    python -m experiments.causal_zvf_experiment train \
        --phase $PHASE \
        --bin-data ./causal_zvf_bins.json \
        --model Qwen/Qwen3-8B \
        --seed 42 \
        --output ./causal_zvf_result_${PHASE}.json
done

# Phase 3: Evaluate on held-out
python -m experiments.causal_zvf_experiment evaluate \
    --run-ids <run-id-1> <run-id-2> <run-id-3> \
    --model Qwen/Qwen3-8B \
    --heldout-n 500 \
    --output ./causal_zvf_heldout.json
```

Or use the shell script:
```bash
bash experiments/run_causal_zvf.sh
```

## Expected Output Structure

```json
{
  "phase": "mixed",
  "model": "Qwen/Qwen3-8B",
  "seed": 42,
  "run_id": "...",
  "steps_completed": 30,
  "metrics": [
    {"step": 0, "mean_reward": 0.45, "zvf": 0.35, "gu": 0.65, ...},
    ...
  ],
  "peak_reward": 0.85,
  "last10_reward": 0.72,
  "reward_slope": 0.015,
  "final_zvf": 0.25,
  "final_gu": 0.75,
  "heldout_accuracy": 0.89,
  "heldout_ci95": [0.86, 0.92]
}
```
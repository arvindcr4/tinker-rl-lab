# Causal ZVF Experiment

## Overview

This experiment directly tests the paper's central claim: **GRPO learns only when sampled groups contain reward diversity.**

By holding everything fixed (model, backend, reward parser, seed, optimizer, LoRA config, group size, evaluator) and changing only the prompt pool, we can determine whether ZVF/GU is merely observational or actually causal.

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

## Primary Success Criterion

The primary endpoint is **NOT** held-out accuracy. The paper's thesis is not "GRPO improves GSM8K"; it is "GRPO has a learning signal only under reward diversity."

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

That pattern would make the paper much more credible even if held-out GSM8K still does not improve significantly.

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
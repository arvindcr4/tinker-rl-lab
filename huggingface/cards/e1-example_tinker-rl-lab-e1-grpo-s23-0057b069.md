---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
datasets:
- openai/gsm8k
tags:
- peft
- lora
- reinforcement-learning
- grpo
- grpo
- gsm8k
- trl
---

# E1 confirmatory GRPO-family run: `grpo`, seed 23 (Qwen3-8B, GSM8K)

**Evidence class: `confirmatory`.** This is one of the 8 pre-registered seeds per arm (11, 23, 37, 53, 71, 89, 107, 131).

Arm: baseline GRPO (token-level ratio, clip 0.2/0.2, group-std scaling).

## Contents

- `final/`: LoRA adapter after the last step, plus tokenizer and chat template.
- `checkpoints/checkpoint-<N>/`: intermediate checkpoints (every 5 steps, with optimizer/RNG state for exact resume).
- `run_manifest.json`: run config, runtime versions, per-step telemetry (reward, GU, ZVF),
  per-item held-out trace (completion SHA-256 only, no text), stack fingerprint and W&B link.

## Training

- Script: [`zvf-program/colab-experiments/e1_grpo_confirmatory.py`](https://github.com/arvindcr4/tinker-rl-lab/blob/86110ffb78f9f15cbf5a9bd38eef6c7eeea5b6dc/zvf-program/colab-experiments/e1_grpo_confirmatory.py) (commit `86110ffb7`). Treatment lock: [`zvf-program/audit/preregistration_e1_treatments.json`](https://github.com/arvindcr4/tinker-rl-lab/blob/86110ffb78f9f15cbf5a9bd38eef6c7eeea5b6dc/zvf-program/audit/preregistration_e1_treatments.json).
- TRL 1.8.0 GRPOTrainer, transformers generation, 1×NVIDIA A100-SXM4-40GB (Google Colab).
- Base `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`; LoRA r=16, α=32, dropout 0, all-linear.
- 30 steps; 4 prompts × 4 completions per optimizer step; lr 1e-6; β (KL) = 0; temperature 1.0;
  max completion 1,024 tokens.
- Data: `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, train split. Reward: binary exact match of the integer after the final `####`.
- Stack fingerprint: `4737be74fdc97dd400b846e93c6b0c03f443ee65da2cc89697bf5b078d16aa0d`.

## Results

| Metric | This run |
|---|---|
| Held-out GSM8K accuracy (n=500, test split) | 0.636 |
| Mean training reward, last 10 steps | 0.7375 |
| Mean ZVF over training | 0.717 |

Across the 8 confirmatory seeds per arm (held-out n=500 each; 95% CI = percentile
bootstrap over seeds):

| Arm | Seeds | Mean held-out acc. | 95% CI |
|---|---|---|---|
| `aero` | 8 | 0.6322 | [0.6300, 0.6350] |
| `dapo` | 8 | 0.6340 | [0.6298, 0.6388] |
| `drgrpo` | 8 | 0.6310 | [0.6252, 0.6372] |
| `grpo` | 8 | 0.6330 | [0.6275, 0.6388] |
| `gspo` | 8 | 0.6380 | [0.6342, 0.6415] |

All five arms fall within about 1 accuracy point of each other, so with 30 steps at
this scale the variant label does not separate outcomes. The held-out accuracy of the
untrained base model is not stored in these manifests.

## Limitations

30 steps, one model, one task. The binary reward cannot tell formatting failures from
reasoning failures. These are research artefacts, not models for deployment.

## Licence

Apache-2.0, matching `Qwen/Qwen3-8B`. GSM8K is MIT-licensed.

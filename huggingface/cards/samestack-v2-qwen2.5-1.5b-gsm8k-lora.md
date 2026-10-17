---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
datasets:
- openai/gsm8k
tags:
- peft
- lora
- reinforcement-learning
- grpo
- ppo
- gsm8k
- math
---

# Same-stack GRPO vs PPO LoRA adapters (Qwen2.5-1.5B-Instruct, GSM8K, v2)

27 LoRA adapters from a controlled comparison in which **every arm uses the same
trainer** (model, LoRA, optimizer, clipped surrogate, generation budget) and only the
advantage estimator and the prompts×completions split differ. One subfolder per run:
`<arm>_s<seed>/` (`adapter_config.json`, `adapter_model.safetensors`, ~17 MB each).

| Arm | Advantage | Prompts × completions per step |
|---|---|---|
| `grpo_g8` | (r − mean_group) / (std_group + ε) | 8 × 8 |
| `grpo_g2` | same | 32 × 2 |
| `ppo` | r − V(prompt), batch-normalised; value head = zero-init linear probe on the detached prompt-end hidden state, 4 warm-up batches | 64 × 1 |

Seeds: `grpo_g8`, `grpo_g2`: 42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777.
`ppo`: 42, 123, 456, 789, 1024, 2048, 3141. The pre-registered `ppo_8x8` arm and `ppo`
seeds 4096/5555/7777 were not run.

## Training

- Script: [`platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py`](https://github.com/arvindcr4/tinker-rl-lab/blob/81ffbccdac8ffe3bef3f60921ec54f1fef08fe9d/platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py) (commit `81ffbccda`). A Colab port without Modal,
  `platform_colab/publication_fixes/samestack_v2_colab.py`, is being added to the repository.
- Base model `Qwen/Qwen2.5-1.5B-Instruct`; LoRA r=16, α=32, dropout 0, q/k/v/o projections.
- 30 steps, 64 generations/step, K=2 inner epochs, clip 0.2, per-response length
  normalisation, Adam lr 1e-5, grad clip 1.0, max 512 new tokens, bf16, 1×A10G (Modal),
  ~42–48 min per run.
- Data: GSM8K `main` train split (prompts), chain-of-thought with a `\boxed{}` answer.
- Reward: 1 if the last `\boxed{}` in the completion equals the gold answer after
  numeric normalisation, else 0. No fallback to the last number.

## Evaluation (greedy, full GSM8K test split, 1,319 items per run)

| Arm | Seeds | Base acc. | Post acc. (mean) | SD across seeds | Truncation (post) |
|---|---|---|---|---|---|
| `grpo_g2` | 10 | 0.6547 | 0.6908 | 0.0076 | 0.021 |
| `grpo_g8` | 10 | 0.6546 | 0.6987 | 0.0049 | 0.018 |
| `ppo` | 7 | 0.6540 | 0.6905 | 0.0075 | 0.018 |

Paired contrasts on post-training accuracy (unit = seed; paired-t 95% CI):

| Contrast | n | Mean diff | 95% CI | p (t) | p (exact sign-flip) | TOST ±0.02 p |
|---|---|---|---|---|---|---|
| grpo_g8 − ppo | 7 | +0.0076 | [−0.0034, +0.0185] | 0.14 | 0.16 | 0.016 (equivalent) |
| grpo_g8 − grpo_g2 | 10 | +0.0079 | [+0.0002, +0.0156] | 0.045 | 0.051 | 0.003 |

Holm-adjusted over the three pre-registered contrasts: 0.28 (g8−ppo), 0.14 (g8−g2).
An independent re-grade of the stored greedy completions agreed with the stored
per-item scores on 100% of items. All arms improve on the base model by about +0.036 to
+0.044; at this scale GRPO and PPO are statistically equivalent within ±2 accuracy points.
An earlier v1 of this experiment (200-token cap, cold critic, 200 test items) reported
+0.050 for GRPO over PPO. That result came from truncation and a critic that never
learned, and v2 does not reproduce it.

## Limitations

- One model (1.5B), one task, 30 steps; it says nothing about larger models or longer training.
- PPO has 7 seeds, and the exposure-matched PPO arm is missing.
- These are research artefacts for checking the comparison, not models for deployment.

## Licence

Apache-2.0, matching the base model `Qwen/Qwen2.5-1.5B-Instruct`. GSM8K is MIT-licensed.

## Reproduce

Analysis-only replay and full-rerun commands: `REPRODUCE.md`, Part A, in https://github.com/arvindcr4/tinker-rl-lab.

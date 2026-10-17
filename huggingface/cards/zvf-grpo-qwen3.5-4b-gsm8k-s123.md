---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
datasets:
- openai/gsm8k
tags:
- peft
- lora
- reinforcement-learning
- grpo
- tinker
- gsm8k
- zero-variance-fraction
---

# GRPO LoRA adapter, Qwen3.5-4B on GSM8K (seed 123), from the ZVF study

A LoRA adapter exported from a 15-step GRPO run on Tinker. It is one run from the
head-to-head sweep used in the Zero-Variance-Fraction (ZVF) analysis. ZVF is the
fraction of prompt groups whose G rewards are all equal, which gives them zero
group-relative advantage.

## Training

- Run record: [`platform_hybrid/experiments/tinker-runs/results/h2h2_grpo_Qwen3.5-4B_s123_G8_Dmedium_R16_S15.json`](https://github.com/arvindcr4/tinker-rl-lab/blob/ddcbb8680e637a663577fc655471e4156e3d94ff/platform_hybrid/experiments/tinker-runs/results/h2h2_grpo_Qwen3.5-4B_s123_G8_Dmedium_R16_S15.json); Tinker run `fae28299-45a5-53d9-8d09-34d4e9edacf0:train:0`,
  checkpoint `tinker://fae28299-45a5-53d9-8d09-34d4e9edacf0:train:0/sampler_weights/final`.
- Trainer: Tinker managed GRPO via [`platform_tinker/tinkerrl/grpo_cli.py`](https://github.com/arvindcr4/tinker-rl-lab/blob/81ffbccdac8ffe3bef3f60921ec54f1fef08fe9d/platform_tinker/tinkerrl/grpo_cli.py).
- Base `Qwen/Qwen3.5-4B`; LoRA rank 16, α=32, all linear layers (attn, MLP, unembed).
- 15 steps, group size G=8, batch 2 prompts/step, lr 1e-5, temperature 0.8, seed 123.
- Data: GSM8K `main` train split ("medium" difficulty bucket). Reward: binary exact match of the final numeric answer.

## Results (training-time only)

| Metric | Value |
|---|---|
| Mean reward, first 5 steps | 0.4875 |
| Mean reward, last 10 steps | 0.6875 |
| Peak step reward | 1.00 |
| Steps with zero loss (all groups zero-variance) | 20% |

No held-out evaluation was run for this adapter. Step rewards come from 2 prompts × 8
samples (16 rollouts), so they are very noisy, and no confidence interval is
meaningful for a single 15-step run.

## How this relates to the ZVF claim

The ZVF numbers in the thesis and paper do not come from this adapter. They come from
stored per-prompt reward tensors (Qwen3-8B, G=8, 3 seeds: ZVF 0.130 / 0.190 / 0.155)
and from W&B project-level runs. To first order, ZVF is the binomial identity
mean_i[p_i^G + (1−p_i)^G] over per-prompt solve rates. See `REPRODUCE.md`, Part B.

## Limitations

A single seed with 15 steps. This is a provenance artefact, not a model for deployment.

## Licence

Apache-2.0, matching `Qwen/Qwen3.5-4B`. GSM8K is MIT-licensed.

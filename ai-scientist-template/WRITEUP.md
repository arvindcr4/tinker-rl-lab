# GRPO GSM8K Experiment — Notes for Paper Writing

## Context
This experiment investigates Group Relative Policy Optimization (GRPO) for training small language models (0.5B-3B parameters) on grade-school math reasoning (GSM8K).

## Key Prior Results (from larger-scale Tinker runs)
- **Capacity threshold**: Models at or below 3B fail to learn with GRPO (zero gradient signal), while 4B+ succeed
- **Two-phase learning**: Format compliance emerges first (steps 1-20), followed by reasoning improvement (steps 21+)
- **LoRA rank**: Affects initial learning speed but not asymptotic accuracy (tested ranks 8-64)
- **Cross-seed stability**: 30.5% +/- 3.3% last-10 accuracy with Qwen3-8B (5 seeds)
- **MoE volatility**: MoE models show 2.43x higher training reward variance than dense models

## What This Template Explores
Smaller-scale (0.5B-3B) experiments that can test ideas quickly before scaling up:
- Reward function design (binary vs partial credit vs step-level)
- Curriculum strategies (easy-first, difficulty-based sampling)
- GRPO hyperparameters (group size, learning rate, KL penalty)
- Prompt engineering (chain-of-thought formats)

## Metrics
- **last_10_accuracy**: Mean reward over last 10 training steps (primary metric)
- **peak_accuracy**: Best single-step reward during training
- **first_5_accuracy**: Mean reward over first 5 steps (measures learning speed)
- **training_loss**: GRPO policy gradient loss

## References to Include
- DeepSeekMath (Shao et al., 2024) — GRPO algorithm
- GSM8K (Cobbe et al., 2021) — benchmark
- LoRA (Hu et al., 2021) — parameter-efficient fine-tuning
- Curriculum Learning (Bengio et al., 2009) — training schedule
- Step-level verification (Lightman et al., 2023) — reward design

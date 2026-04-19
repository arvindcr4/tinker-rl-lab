# Tinker RL Lab — Master Experiment Summary

_Generated: 2026-04-19 02:40 UTC_

**Total experiments:** 44  
**W&B Project:** https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class

## TRL GRPO Baseline (Qwen2.5-0.5B, 5 seeds)
Mean accuracy: **0.734** ± 0.070 (seeds: [42, 123, 456, 789, 1024])


## Tinker GRPO

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |
|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|
| ✅ | frontier gsm8k deepseek v3.1 | deepseek-v3.1 | GRPO | gsm8k | 20 | 1.000 | 0.850 | — | [W&B](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/56b99b24-03bd-5d00-ae23-831933ef53b2) | [HF](tinker://56b99b24-03bd-5d00-ae23-831933ef53b2:train:0/sampler_weights/final) |
| ❌ | cross tool llama 8b inst | llama-8b-inst | GRPO | tool_use | 30 | 0.000 | 0.000 | Complete failure — tool_use task too hard for Llama-3.1-8B-Instruct (100% zero reward) | [W&B](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/ca2e3a24-7401-5770-af34-a0d27177aeaa) | [HF](tinker://ca2e3a24-7401-5770-af34-a0d27177aeaa:train:0/sampler_weights/final) |
| ✅ | scale gsm8k qwen3 8b | qwen3-8b | GRPO | gsm8k | 30 | 0.625 | 0.344 | — | [W&B](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/3154dee5-4fd7-59b6-ba3b-721eef675bfc) | [HF](tinker://3154dee5-4fd7-59b6-ba3b-721eef675bfc:train:0/sampler_weights/final) |
| ❌ | scale gsm8k qwen3 32b | qwen3-32b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | scale gsm8k qwen3.5 4b | qwen3.5-4b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | scale gsm8k qwen3.5 27b | qwen3.5-27b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | scale gsm8k llama 8b inst | llama-8b-inst | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | frontier gsm8k qwen3 235b | qwen3-235b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | frontier gsm8k nemotron 120b | nemotron-120b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | moe gsm8k qwen3 30b moe | qwen3-30b-moe | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | moe gsm8k qwen3 30b inst | qwen3-30b-inst | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | cross tool qwen3 32b | qwen3-32b | GRPO | tool | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | arch gsm8k gpt oss 20b | gpt-oss-20b | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |
| ❌ | arch gsm8k kimi k2 | kimi-k2 | GRPO | gsm8k | — | — | — | Blocked by Tinker JWT auth error; no training data collected | — | — |

## Modal PPO

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |
|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|
| ✅ | ppo_gsm8k_Qwen3-8B_s42 | qwen3-8b | PPO-REINFORCE | gsm8k | 30 | 1.000 | 0.350 | High variance — Qwen3-8B peaks 1.0 but last-10 avg only 0.35; unstable PPO training | [W&B](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/ppo-gsm8k-Qwen3-8B-s42) | [HF](https://huggingface.co/arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42) |
| ✅ | ppo_gsm8k_Llama-3.1-8B-Instruct_s42 | llama-8b-inst | PPO-REINFORCE | gsm8k | 30 | 1.000 | 0.950 | Strong performance — Llama-3.1-8B peaks 1.0, last-10 avg 0.95 on GSM8K | [W&B](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/ppo-gsm8k-Llama-3.1-8B-Instruct-s42) | — |

## Modal Other

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |
|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|
| ⚠️ | humaneval qwen3 8b | qwen3-8b | eval | humaneval | — | — | — | Partial result: 40/164 problems, cum_pass@1=65.0%; timed out at 3600s | — | — |
| ⚠️ | heldout qwen3.5 27b | qwen3.5-27b | eval | heldout | — | — | — | Partial result: 100/200 samples, 86/100=86.0%; timed out at 3600s | — | — |
| ⚠️ | heldout qwen3 32b | qwen3-32b | eval | heldout | — | — | — | Partial result: 100/200 samples, 33/100=33.0%; timed out at 3600s | — | — |
| ❌ | kl qwen3 8b | qwen3-8b | PPO+KL | gsm8k | — | — | — | Gradient error — KL divergence PPO variant failed immediately on Qwen3-8B | — | — |

## Team Member

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |
|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|
| ✅ | Sandhya – GRPO Tool Calling (3B) | 3b-tool | GRPO vs SFT | tool_calling | — | 0.910 | 0.910 | GRPO 0.91 vs SFT baseline 0.72 on tool-calling — +0.19 absolute improvement with RL | — | — |
| ✅ | Madhu – HumanEval (Qwen3-8B) | qwen3-8b | GRPO | code_generation | — | 0.860 | 0.860 | HumanEval 86% (141/164 problems) — best code-gen result in the project | — | [HF](https://github.com/madhukumara1993/qwen3-grpo) |
| ✅ | Mohammad Rafi – Math Reasoning (Qwen3-4B) | qwen3-4b | GRPO | math_reasoning | — | 0.678 | 0.678 | +0.6pp accuracy gain (67.2% → 67.8%) on math reasoning with Qwen3-4B GRPO | — | — |
| ✅ | Arumugam – DPO Keyword Eval (0.5B) | 0.5b-dpo | DPO | keyword_generation | — | — | — | +25% keyword metric with DPO on 0.5B model (8 training examples — very low-data regime) | — | — |

## Old TRL

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Finding | W&B | HF Checkpoint |
|--------|-----------|-------|--------|------|-------|------|-------------|---------|-----|---------------|
| ✅ | TRL GRPO Math seed=42 | qwen2.5-0.5b | GRPO | math | 125 | 0.735 | 0.735 | Seed 42: accuracy=0.735, loss=0.00896, elapsed=141s | — | — |
| ✅ | TRL GRPO Math seed=123 | qwen2.5-0.5b | GRPO | math | 125 | 0.810 | 0.810 | Seed 123: accuracy=0.810, loss=0.00699, elapsed=135s | — | — |
| ✅ | TRL GRPO Math seed=456 | qwen2.5-0.5b | GRPO | math | 125 | 0.620 | 0.620 | Seed 456: accuracy=0.620, loss=0.00792, elapsed=142s | — | — |
| ✅ | TRL GRPO Math seed=789 | qwen2.5-0.5b | GRPO | math | 125 | 0.740 | 0.740 | Seed 789: accuracy=0.740, loss=0.00322, elapsed=141s | — | — |
| ✅ | TRL GRPO Math seed=1024 | qwen2.5-0.5b | GRPO | math | 125 | 0.765 | 0.765 | Seed 1024: accuracy=0.765, loss=0.00418, elapsed=183s | — | — |
| ✅ | SB3 PPO Math seed=42 | sb3-ppo | PPO | math | 100352 | 0.035 | 0.008 | SB3/PPO: final accuracy=0.003 — policy gradient on raw math fails without LLM | — | — |
| ✅ | SB3 PPO Math seed=123 | sb3-ppo | PPO | math | 100352 | 0.045 | 0.008 | SB3/PPO: final accuracy=0.014 — policy gradient on raw math fails without LLM | — | — |
| ✅ | SB3 PPO Math seed=456 | sb3-ppo | PPO | math | 100352 | 0.020 | 0.006 | SB3/PPO: final accuracy=0.011 — policy gradient on raw math fails without LLM | — | — |
| ✅ | SB3 PPO Math seed=789 | sb3-ppo | PPO | math | 100352 | 0.020 | 0.010 | SB3/PPO: final accuracy=0.012 — policy gradient on raw math fails without LLM | — | — |
| ✅ | SB3 PPO Math seed=1024 | sb3-ppo | PPO | math | 100352 | 0.025 | 0.010 | SB3/PPO: final accuracy=0.010 — policy gradient on raw math fails without LLM | — | — |
| ✅ | CleanRL PPO Math seed=42 | cleanrl-ppo | PPO | math | 97280 | 0.020 | 0.011 | CleanRL/PPO: final accuracy=0.010 — near-zero, confirms LLM mandatory | — | — |
| ✅ | CleanRL PPO Math seed=123 | cleanrl-ppo | PPO | math | 97280 | 0.005 | 0.002 | CleanRL/PPO: final accuracy=0.008 — near-zero, confirms LLM mandatory | — | — |
| ✅ | CleanRL PPO Math seed=456 | cleanrl-ppo | PPO | math | 97280 | 0.015 | 0.008 | CleanRL/PPO: final accuracy=0.014 — near-zero, confirms LLM mandatory | — | — |
| ✅ | CleanRL PPO Math seed=789 | cleanrl-ppo | PPO | math | 97280 | 0.025 | 0.009 | CleanRL/PPO: final accuracy=0.009 — near-zero, confirms LLM mandatory | — | — |
| ✅ | CleanRL PPO Math seed=1024 | cleanrl-ppo | PPO | math | 97280 | 0.010 | 0.003 | CleanRL/PPO: final accuracy=0.004 — near-zero, confirms LLM mandatory | — | — |
| ✅ | Tianshou PPO Math seed=42 | tianshou-ppo | PPO | math | 100000 | 0.025 | 0.013 | Tianshou/PPO: final accuracy=0.011 — near-zero, <3s runtime indicates trivial baseline | — | — |
| ✅ | Tianshou PPO Math seed=123 | tianshou-ppo | PPO | math | 100000 | 0.020 | 0.005 | Tianshou/PPO: final accuracy=0.002 — near-zero, <3s runtime indicates trivial baseline | — | — |
| ✅ | Tianshou PPO Math seed=456 | tianshou-ppo | PPO | math | 100000 | 0.020 | 0.010 | Tianshou/PPO: final accuracy=0.005 — near-zero, <3s runtime indicates trivial baseline | — | — |
| ✅ | Tianshou PPO Math seed=789 | tianshou-ppo | PPO | math | 100000 | 0.020 | 0.006 | Tianshou/PPO: final accuracy=0.002 — near-zero, <3s runtime indicates trivial baseline | — | — |
| ✅ | Tianshou PPO Math seed=1024 | tianshou-ppo | PPO | math | 100000 | 0.020 | 0.007 | Tianshou/PPO: final accuracy=0.009 — near-zero, <3s runtime indicates trivial baseline | — | — |

## Summary Statistics

| Status | Count |
|--------|-------|
| ✅ completed | 28 |
| ❌ failed | 13 |
| ⚠️ partial | 3 |

### Key Findings

- **Best overall performer:** Llama-3.1-8B-Instruct (Modal PPO) — last-10 avg **0.95** on GSM8K
- **Best frontier model:** DeepSeek-V3.1 (Tinker GRPO) — peak **1.0**, last-10 avg **0.85**
- **Best team member result:** Madhu (Qwen3-8B GRPO) — HumanEval **86%** (141/164)
- **Most improvement vs baseline:** Sandhya (3B GRPO) — **+0.19** absolute over SFT (0.72→0.91)
- **TRL GRPO baseline** (Qwen2.5-0.5B, 5 seeds): mean **0.734** ± 0.065
- **Classical PPO (SB3/CleanRL/Tianshou)** on raw math: all < 0.02 — LLM backbone essential
- **JWT failures:** 11 Tinker runs blocked by auth errors, 0 training data collected
- **Tinker tool_use failure:** Llama-3.1-8B-Instruct scored 0.0 on all 30 steps — task too hard

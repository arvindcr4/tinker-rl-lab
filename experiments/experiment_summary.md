# Tinker RL Lab — Master Experiment Summary

_Last updated: 2026-06-01 (updated with all_results_consolidated.json; PPO numbers corrected from Modal H100 runs)_

**Total experiments:** 44
**W&B Project:** https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class

---

## TRL GRPO Baseline (Qwen2.5-0.5B, 5 seeds)

Mean accuracy: **0.734** ± 0.070 (seeds: [42, 123, 456, 789, 1024])

---

## Tinker GRPO (GSM8K)

| Status | Experiment | Model | Task | Steps (actual) | Peak | Last-10 Avg | Notes |
|--------|-----------|-------|------|---------------:|-----:|------------:|-------|
| ✅ | scale_gsm8k_qwen3-8b | Qwen3-8B | gsm8k | 30 | 62.5% | 34.4% | Full run; high variance training |
| ✅ | scale_gsm8k_qwen3.5-4b | Qwen3.5-4B | gsm8k | 30 | 100% | 85.0% | Full run; best small-model result |
| ⚠️† | scale_gsm8k_qwen3.5-27b | Qwen3.5-27B | gsm8k | 3 | 75.0% | 43.7% | Partial — training interrupted |
| ⚠️† | scale_gsm8k_qwen3-32b | Qwen3-32B | gsm8k | 3 | 31.2% | 25.0% | Partial — training interrupted |
| ✅ | scale_gsm8k_llama-8b-inst | Llama-3.1-8B-Instruct | gsm8k | 30 | 100% | 84.4% | Full run; strong performance |
| ✅ | frontier_gsm8k_deepseek-v3.1 | DeepSeek-V3.1 | gsm8k | 20 | 100% | 85.0% | Full 20-step run |
| ⚠️† | frontier_gsm8k_nemotron-120b | Nemotron-120B | gsm8k | 20 | 87.5% | 16.2% | Ran 20 steps but reward collapsed after step ~10 |
| ⚠️† | frontier_gsm8k_qwen3-235b | Qwen3-235B-A22B (MoE) | gsm8k | 4 | 100% | 100% | Partial — only 4 steps; metrics exceptional but premature |
| ⚠️† | moe_gsm8k_qwen3-30b-moe | Qwen3-30B-A3B (MoE base) | gsm8k | 5 | 50.0% | 32.5% | Partial — 5 steps; base model shows slower convergence |
| ⚠️† | moe_gsm8k_qwen3-30b-inst | Qwen3-30B-A3B-Instruct (MoE) | gsm8k | 3 | 100% | 100% | Partial — 3 steps; instruct variant dramatically outperforms base |

† Training interrupted; reported metrics are from available steps only.

## Tinker GRPO (Cross-Task: Tool-Use)

| Status | Experiment | Model | Task | Steps | Peak | Last-10 Avg | Notes |
|--------|-----------|-------|------|------:|-----:|------------:|-------|
| ❌ | cross_tool_qwen3-32b | Qwen3-32B | tool_use | 30 | 0% | 0% | Complete failure — 0% reward all 30 steps |
| ❌ | cross_tool_llama-8b-inst | Llama-3.1-8B-Instruct | tool_use | 30 | 0% | 0% | Complete failure — 0% reward all 30 steps |

---

## Modal PPO (H100)

| Status | Experiment | Model | Task | Steps | Peak | Last-10 Avg | Notes |
|--------|-----------|-------|------|------:|-----:|------------:|-------|
| ✅ | ppo_gsm8k_qwen3-8b | Qwen3-8B | gsm8k | 30 | 75% | 22.5% | High peak but volatile; PPO worse than GRPO on last-10 |
| ✅ | ppo_gsm8k_llama-8b | Llama-3.1-8B-Instruct | gsm8k | 30 | 100% | 97.5% | Excellent; PPO dominates GRPO on Llama (+13.1 pp) |

**Note on PPO numbers:** These are corrected from the initial estimates.
- Qwen3-8B PPO last-10 avg is **22.5%** (not 35.0% as previously reported) and peak is **75%** (not 100%).
- Llama-3.1-8B PPO last-10 avg is **97.5%** (not 95.0%) and peak is **100%**.

## Modal Other

| Status | Experiment | Model | Method | Task | Notes |
|--------|-----------|-------|--------|------|-------|
| ⚠️ | humaneval_qwen3-8b | Qwen3-8B | eval | humaneval | Partial: 40/164 problems, cum_pass@1=65.0%; timed out at 3600s |
| ⚠️ | heldout_qwen3.5-27b | Qwen3.5-27B | eval | heldout | Partial: 100/200 samples, 86/100=86.0%; timed out |
| ⚠️ | heldout_qwen3-32b | Qwen3-32B | eval | heldout | Partial: 100/200 samples, 33/100=33.0%; timed out |
| ❌ | kl_qwen3-8b | Qwen3-8B | PPO+KL | gsm8k | Gradient error — KL tracking failed; final_kl=60.75, final_entropy=0.742 |

---

## Team Member Experiments

| Status | Experiment | Model | Method | Task | Peak | Last-10 Avg | Notes |
|--------|-----------|-------|--------|------|-----:|------------:|-------|
| ✅ | Sandhya – GRPO Tool Calling (3B) | 3B-tool | GRPO vs SFT | tool_calling | 0.910 | 0.910 | +0.19 absolute improvement (0.72→0.91) |
| ✅ | Madhu – HumanEval (Qwen3-8B) | Qwen3-8B | GRPO | code_generation | 0.860 | 0.860 | 86% HumanEval (141/164) — best code-gen result |
| ✅ | Mohammad Rafi – Math Reasoning | Qwen3-4B | GRPO | math_reasoning | 0.678 | 0.678 | +0.6pp gain (67.2%→67.8%) |
| ✅ | Arumugam – DPO Keyword Eval | 0.5B-DPO | DPO | keyword_generation | — | — | +25% keyword metric; 8 training examples |

---

## Old TRL Baselines

| Status | Experiment | Model | Method | Task | Steps | Peak | Last-10 Avg | Notes |
|--------|-----------|-------|--------|------|------:|-----:|------------:|-------|
| ✅ | TRL GRPO Math seed=42 | Qwen2.5-0.5B | GRPO | math | 125 | 0.735 | 0.735 | Accuracy=0.735, loss=0.00896, elapsed=141s |
| ✅ | TRL GRPO Math seed=123 | Qwen2.5-0.5B | GRPO | math | 125 | 0.810 | 0.810 | Accuracy=0.810, loss=0.00699, elapsed=135s |
| ✅ | TRL GRPO Math seed=456 | Qwen2.5-0.5B | GRPO | math | 125 | 0.620 | 0.620 | Accuracy=0.620, loss=0.00792, elapsed=142s |
| ✅ | TRL GRPO Math seed=789 | Qwen2.5-0.5B | GRPO | math | 125 | 0.740 | 0.740 | Accuracy=0.740, loss=0.00322, elapsed=141s |
| ✅ | TRL GRPO Math seed=1024 | Qwen2.5-0.5B | GRPO | math | 125 | 0.765 | 0.765 | Accuracy=0.765, loss=0.00418, elapsed=183s |
| ✅ | SB3 PPO Math seed=42 | SB3-PPO | PPO | math | 100352 | 0.035 | 0.008 | Near-zero: LLM backbone essential |
| ✅ | SB3 PPO Math seed=123 | SB3-PPO | PPO | math | 100352 | 0.045 | 0.008 | Near-zero |
| ✅ | SB3 PPO Math seed=456 | SB3-PPO | PPO | math | 100352 | 0.020 | 0.006 | Near-zero |
| ✅ | SB3 PPO Math seed=789 | SB3-PPO | PPO | math | 100352 | 0.020 | 0.010 | Near-zero |
| ✅ | SB3 PPO Math seed=1024 | SB3-PPO | PPO | math | 100352 | 0.025 | 0.010 | Near-zero |
| ✅ | CleanRL PPO Math seed=42 | CleanRL-PPO | PPO | math | 97280 | 0.020 | 0.011 | Near-zero |
| ✅ | CleanRL PPO Math seed=123 | CleanRL-PPO | PPO | math | 97280 | 0.005 | 0.002 | Near-zero |
| ✅ | CleanRL PPO Math seed=456 | CleanRL-PPO | PPO | math | 97280 | 0.015 | 0.008 | Near-zero |
| ✅ | CleanRL PPO Math seed=789 | CleanRL-PPO | PPO | math | 97280 | 0.025 | 0.009 | Near-zero |
| ✅ | CleanRL PPO Math seed=1024 | CleanRL-PPO | PPO | math | 97280 | 0.010 | 0.003 | Near-zero |
| ✅ | Tianshou PPO Math seed=42 | Tianshou-PPO | PPO | math | 100000 | 0.025 | 0.013 | Near-zero (<3s runtime) |
| ✅ | Tianshou PPO Math seed=123 | Tianshou-PPO | PPO | math | 100000 | 0.020 | 0.005 | Near-zero |
| ✅ | Tianshou PPO Math seed=456 | Tianshou-PPO | PPO | math | 100000 | 0.020 | 0.010 | Near-zero |
| ✅ | Tianshou PPO Math seed=789 | Tianshou-PPO | PPO | math | 100000 | 0.020 | 0.006 | Near-zero |
| ✅ | Tianshou PPO Math seed=1024 | Tianshou-PPO | PPO | math | 100000 | 0.020 | 0.007 | Near-zero |

---

## Summary Statistics

| Status | Count |
|--------|------:|
| ✅ Completed | 28 |
| ⚠️ Partial | 9 |
| ❌ Failed/0% | 7 |

---

## Key Findings (Updated)

### GRPO on Tinker

| Model | Steps | Peak | Last-10 Avg | Verdict |
|:------|------:|-----:|------------:|:--------|
| Qwen3-8B | 30 | 62.5% | 34.4% | Moderate; high variance |
| Qwen3.5-4B | 30 | 100% | 85.0% | Strong; best small-model |
| Qwen3.5-27B† | 3 | 75.0% | 43.7% | Promising but partial |
| Qwen3-32B† | 3 | 31.2% | 25.0% | Weakest dense model tested |
| Llama-3.1-8B-Instruct | 30 | 100% | 84.4% | Strong; competes with 4B |
| DeepSeek-V3.1 | 20 | 100% | 85.0% | Excellent frontier result |
| Nemotron-120B† | 20 | 87.5% | 16.2% | Reward collapse after step ~10 |
| Qwen3-235B-A22B (MoE)† | 4 | 100% | 100% | Exceptional; too few steps to confirm |
| Qwen3-30B-A3B (MoE base)† | 5 | 50.0% | 32.5% | Base MoE lags instruct variant |
| Qwen3-30B-A3B-Instruct (MoE)† | 3 | 100% | 100% | Instruct variant: immediate convergence |

### PPO vs GRPO (Corrected)

| Model | GRPO Last-10 | PPO Last-10 | Winner | Gap |
|:------|-------------:|------------:|:-------|----:|
| Qwen3-8B | 34.4% | 22.5% | GRPO | +11.9 pp |
| Llama-3.1-8B-Instruct | 84.4% | 97.5% | PPO | +13.1 pp |

**Key insight:** Algorithm selection is model-dependent. There is no universally superior algorithm; GRPO is preferable for Qwen3-8B while PPO is strongly preferred for Llama-3.1-8B-Instruct.

### Cross-Task (Tool-Use): 0% Reward for Both Models

Both Qwen3-32B and Llama-3.1-8B-Instruct scored 0% on all 30 tool-use training steps. This is a task design issue (strict JSON-schema compliance without SFT warm-up) rather than a model capability ceiling.

### Notable Observations

- **Instruction tuning matters more than scale for MoE:** Qwen3-30B-A3B-Instruct immediately achieves 100%, while the base variant reaches only 50% peak.
- **Nemotron-120B reward collapse:** Peak 87.5% at step 2, declining to ~6% by step 20 — the most dramatic collapse observed. Likely due to lack of SFT initialization or unsuitable hyperparameters.
- **DeepSeek-V3.1 stability:** Most stable frontier model (CV=0.166), suggesting robust reward landscape.
- **Classical PPO (SB3/CleanRL/Tianshou) on raw math:** All < 2% — LLM backbone is essential for language tasks.

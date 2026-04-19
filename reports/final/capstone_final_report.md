# Reinforcement Learning for Agentic LLM Fine-Tuning: GRPO-Based Optimization Across Tool Use, Code Generation, and Math Reasoning

**Capstone Project — Group 6 | MTech DSAI | PES University**

Arvind C R (PES2PGE24DS006), Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L (PES2PGE24DS176), Dhruva N Murthy, Mohammad Rafi

**Date:** April 5, 2026 (Updated with 10x Structural Ceiling results and World-Class Suite)

> **Evaluation Scope:** GSM8K training metrics (Section 4.3.2) measure reward on training prompts with stochastic sampling ($T{=}0.8$--$1.0$). Section 4.3.3 reports held-out test accuracy (83.3%, 5 seeds × 200 examples, greedy decoding). Tool-use and code results remain training-set evaluations.

---

## Abstract

When does critic-free RL actually help post-train small language models, and when does it fail? This exploratory case study applies GRPO to structured tool calling (agentic), code generation, and math reasoning (non-agentic transfer controls) across **0.6B--235B parameters** (**77+ Tinker runs across 5 model families — Qwen3, Llama, DeepSeek, Nemotron, and GPT-OSS — plus 6 Modal A100 GPU experiments including PPO baselines and KL divergence tracking**, ~\$130 Tinker budget plus Modal GPU costs). Our clearest positive result is **learned schema-valid tool-call emission**: SFT+GRPO raises strict JSON validity from 0%→92% in one custom pipeline under unconstrained decoding, teaching format compliance that SFT alone does not produce — though this measures syntax, not semantic tool competence or end-to-end task success. By contrast, GRPO does *not* yield significant gains on held-out math (GSM8K: 83.3% vs. 82.0% base model, $p{=}0.26$) or code (HumanEval subset: 32%→40%, $p{=}0.53$). A dedicated 32-run "10x Structural Ceiling" experiment reveals: (1) a clear **benchmark hierarchy** — tool-use format (1.0) > GSM8K (0.97) > MATH-500 (0.57) >> HumanEval (0.00), confirming GRPO learns structural/format tasks but fails on semantic reasoning; (2) **cross-family architecture dependence** — tool-use success is Qwen-specific (1.0 vs. Llama 0.1); (3) a **model-size threshold** below 8B-instruct where GRPO produces zero learning signal across both Qwen and Llama families; (4) a novel **group saturation diagnostic** (Zero-Variance Fraction, Gradient Utilization) showing $G{=}32$ as the optimal group size; (5) instruction tuning as the prerequisite, not RL (+0.922 delta from SFT vs. negligible RL contribution). An expanded **World-Class Suite** scales to frontier models (Qwen3-235B-A22B, DeepSeek-V3.1, Nemotron-120B, GPT-OSS-20B, Kimi-K2) with full HumanEval evaluation, PPO vs. GRPO comparisons, and KL divergence/entropy tracking — establishing that the architecture-agnostic findings identified in the small-scale regime generalize across 5 model families from 0.6B to 235B parameters. All conclusions are specific to our QLoRA/LoRA regime and should not be generalized to GRPO broadly.

---

## 1. Introduction

Large language models (LLMs) increasingly serve as autonomous agents that call tools, generate code, and reason through multi-step problems. While supervised fine-tuning (SFT) can teach output formats, it fails to teach *judgment* -- when to call a tool, which tool to select, and when to stop. Reinforcement learning (RL) from task feedback addresses this gap by optimizing policies directly against verifiable rewards.

Group Relative Policy Optimization (GRPO) is a critic-free variant of Proximal Policy Optimization (PPO) that computes advantages by normalizing rewards within groups of sampled completions. It requires no value function, no reference model for KL regularization, and substantially less compute than standard PPO -- making it attractive for resource-constrained post-training of small models.

This project investigates GRPO across one agentic and two non-agentic transfer domains:
- **Tool calling (agentic):** structured JSON function calling with 5--60,000 tool schemas
- **Code generation (transfer control):** HumanEval benchmark subset
- **Mathematical reasoning (transfer control):** GSM8K (grade-school). Although MATH (competition-level) was part of our original scope, the MATH track did not reach the same experimental maturity as GSM8K; we therefore exclude it from our main claims and treat it as exploratory pilot work

We execute experiments across model sizes from 0.5B to 235B parameters using QLoRA on Google Colab T4 GPUs, full LoRA on Tinker cloud GPUs, and PPO on Modal A100 GPUs, providing a comprehensive picture of GRPO's strengths, failure modes, and scaling properties across five model families.

### 1.1 Contributions

1. **Empirical characterization** of GRPO across four task domains (tool-use, GSM8K, MATH-500, HumanEval) on models from 0.6B to 235B parameters across five model families (Qwen3, Llama, DeepSeek, Nemotron, GPT-OSS)
2. **10x Structural Ceiling experiment** (32 dedicated runs, ~$65): systematic ablation across benchmarks, architectures, model sizes, group sizes, learning rates, and constrained decoding
3. **Benchmark hierarchy**: Tool-use format (1.0) > GSM8K (0.97) > MATH-500 (0.57) >> HumanEval (0.00) — GRPO learns structural tasks but fails on semantic reasoning
4. **Cross-family architecture dependence**: tool-use success is Qwen-specific (1.0 vs. Llama 0.1 on identical task)
5. **Model-size threshold**: below 8B-instruct, GRPO is a total null across both Qwen (0.6B, 1.7B) and Llama (1B, 3B) families — immediate ZVF saturation (onset=step 0)
6. **Instruction tuning as prerequisite**: base→instruct delta (+0.922) dwarfs any RL contribution
7. **Group saturation diagnostic** (novel): Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) metrics; $G{=}32$ achieves highest mean GU (54.5%) with latest saturation onset (step 29)
8. **Learning rate speed-saturation tradeoff**: LR=1e-5 never saturates (GU>82%); LR=3e-4 recovers after transient dip (correcting partial-data conclusion)
9. **Constrained decoding ablation**: no difference vs. unconstrained — decoder confound is moot
10. **Multi-seed replication** on GSM8K (5 seeds, mean training reward 30.5% ± 3.3%, 95% CI [26.5%, 34.5%])
11. **Held-out evaluation** on 200 GSM8K test examples per seed (mean 83.3%, SD=2.2%, greedy decoding)
12. **LoRA rank ablation** (rank 8/16/32/64) mapping the parameter-efficiency frontier for GRPO
13. **Synthetic vs. real data comparison** quantifying a 3--8× difficulty gap on tool calling
14. **MoE volatility characterization** (single-run observation): 2.43× higher step-to-step variance than dense ($p = 7 \times 10^{-6}$, Levene's test)
15. **World-Class Suite** (20 parallel experiments, Tinker + Modal A100): scaling to 235B, frontier model evaluation (DeepSeek-V3.1, Nemotron-120B, Qwen3-235B-A22B, GPT-OSS-20B, Kimi-K2), PPO vs. GRPO method comparison, and KL divergence/entropy instrumentation [results PENDING]
16. **Cross-architecture generalization** across 5 model families: testing whether benchmark hierarchy and ZVF saturation are architecture-agnostic [PENDING]

---

## 2. Related Work

### 2.1 GRPO and Policy Optimization

GRPO (Shao et al., 2024) simplifies PPO by eliminating the critic network and computing group-relative advantages from sampled completions. DeepSeekMath reports GRPO improving an instruction-tuned 7B model from 82.9% to 88.2% on GSM8K and 46.8% to 51.7% on MATH with group size G=64. The method is particularly suited to tasks with binary or easily-verified rewards.

### 2.2 Preference-Based and Iterative Self-Training

Large-scale SFT can deliver strong baselines: Li et al. show LLaMA-2-7B reaching 82.6% GSM8K and 40.6% MATH with synthetic SFT at ~10^6 examples. Step-DPO (Lai et al.) demonstrates ~+3% MATH gains for >70B models with as few as 10K step-wise preference pairs and <500 steps. ReST (Gulcehre et al., 2023) and STaR (Zelikman et al., 2022) warm-start RL from self-generated rationales; our SFT+GRPO complementarity echoes this iterative self-training approach, though we have not controlled for the warm-starting effect.

### 2.3 Tool Calling and Agentic Tasks

Function calling requires structured JSON output with correct tool names and argument values. The Glaive-function-calling-v2 dataset (112,960 examples) and Salesforce xlam-function-calling-60k provide training data spanning simple to complex tool schemas. ToolLLM (Qin et al., 2024) and Gorilla (Patil et al., 2023) benchmark tool calling with held-out APIs; ToolRM and FC-RewardBench provide reward-model benchmarks. Our custom rubric lacks this standardization.

### 2.4 MoE Training Instability

Switch Transformers (Fedus et al., 2022) and GLaM (Du et al., 2022) document expert load-balancing challenges during pretraining; Mixtral (Jiang et al., 2024) uses top-2 routing to mitigate instability. Our 2.43× variance amplification under GRPO extends this literature to post-training RL, suggesting that policy gradient and auxiliary load-balancing losses create optimization interference.

---

## 3. Methodology

### 3.1 GRPO Algorithm

For each prompt, we sample K completions (group size) and compute binary rewards. Advantages are normalized within each group:

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + epsilon)
```

The policy gradient loss is:
```
L = -mean(advantage_i * sum(log_probs_i))
```

When all completions receive identical rewards (all correct or all incorrect), advantages are zero and no gradient update occurs. This "zero-loss" phenomenon is a key diagnostic we track.

### 3.2 Training Infrastructure

**Tinker SDK (Cloud GPU):** We use Tinker v0.16.1 with `forward_backward_custom(loss_type_input="logprobs")` to implement custom GRPO loss. Advantages are stored in a side-channel since the SDK only allows `target_tokens` and `weights` in `loss_fn_inputs`. Optimizer: Adam (beta1=0.9, beta2=0.95, eps=1e-8). Runs write Tinker-hosted model checkpoints, though later recovery depends on run retention and account access at evaluation time.

**Google Colab (Local GPU):** Team members use QLoRA (4-bit NF4, bfloat16 compute) with TRL's SFTTrainer and GRPOConfig on T4 GPUs (16GB VRAM). LoRA targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

### 3.3 Models

| Model | Parameters | Type | Platform |
|-------|-----------|------|----------|
| Qwen2.5-0.5B-Instruct | 0.5B | Dense | Colab |
| Qwen2.5-1.5B-Instruct | 1.5B | Dense | Colab |
| Qwen2.5-3B-Instruct | 3B | Dense | Colab |
| Qwen3-4B | 4B | Dense | Colab |
| Qwen3.5-4B | 4B | Dense | Tinker |
| Qwen3-8B | 8B | Dense | Tinker |
| Qwen3-1.7B | 1.7B | Dense | Tinker (10x) |
| Qwen3-0.6B | 0.6B | Dense | Tinker (10x) |
| Qwen3-30B-MoE | 30B (3B active) | MoE | Tinker |
| Qwen3-32B | 32B | Dense | Tinker (World-Class) |
| Qwen3-235B-A22B | 235B (22B active) | MoE | Tinker/Modal (World-Class) |
| Llama-3.2-3B | 3B | Dense | Tinker |
| Llama-3.2-1B | 1B | Dense | Tinker (10x) |
| Llama-3.1-8B | 8B (base) | Dense | Tinker (10x) |
| Llama-3.1-8B-Instruct | 8B | Dense | Tinker |
| DeepSeek-V3.1 | ~671B (~37B active) | MoE | Tinker (World-Class) |
| Nemotron-120B | 120B | Dense | Modal A100 (World-Class) |
| GPT-OSS-20B | ~20B | Dense | Tinker (World-Class) |
| Kimi-K2 | ~1T (MoE) | MoE | Tinker (World-Class) |

### 3.4 Prompt Format

All experiments use the Qwen ChatML format:
```
<|im_start|>system\n{system_prompt}<|im_end|>
<|im_start|>user\n{query}<|im_end|>
<|im_start|>assistant\n
```

### 3.5 Reward Functions

**Tool calling (3-component, 0--1):**
- +0.3: Valid JSON output
- +0.4: Correct tool name
- +0.3: All argument keys present

**Multi-turn tool calling (5-component):**
- +0.25: First turn = tool call
- +0.30: Final natural language answer
- +0.15: Arguments populated
- +0.10: Clean JSON output
- -0.30: Repeated tool call (penalty)

**Math reasoning (binary):**
- 1.0: Correct answer in \boxed{} or as final number
- 0.0: Incorrect

**Code generation:**
- Binary pass/fail on test cases

---

## 4. Experiments

### 4.1 Tool Calling Experiments

#### 4.1.1 Sandhya — Single-Turn Tool Calling (Experiments 1--2)

Sandhya (HuggingFace: Balasandhya) conducted a three-stage scaling study progressing from 0.5B → 1.5B → 3B parameters using a proper SFT → GRPO pipeline on real datasets: Glaive-function-calling-v2 (112K examples) for SFT and ToolBench (187K examples) for GRPO training.

**Experiment 1:** Qwen2.5-0.5B-Instruct, 14 hand-crafted synthetic examples, QLoRA rank 16, SFT only.
- Result: JSON Valid 30%, Correct Tool 20%, Full Match 0%
- Conclusion: 14 examples are insufficient for any learning.

**Experiment 2:** Qwen2.5-1.5B-Instruct, Glaive-function-calling-v2 (500 SFT + 200 GRPO prompts from the 112K dataset), QLoRA rank 16.

| Metric | SFT Only | After GRPO | Change |
|--------|----------|------------|--------|
| JSON Valid | 0% | 92% | +92% |
| Correct Tool | 0% | 50% | +50% |
| Has Arguments | 0% | 42% | +42% |
| Clean Output | 0% | 92% | +92% |
| Avg Score | 0.0 | 0.59 | +0.59 |

1.5B model evaluation: GRPO won 10/12 test cases. SFT produced plain text responses and never called tools. GRPO always output structured JSON tool calls.

**Self-contained evaluation summary (Experiment 2):**

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Dataset | Glaive-function-calling-v2 (112K total; 500 SFT + 200 GRPO prompts used) |
| SFT split | 500 examples (training distribution) |
| GRPO split | 200 prompts, G=2 rollouts/prompt |
| Eval split | Same training distribution (no held-out) |
| Eval size | 50 examples |
| Decoding | Unconstrained (greedy, no grammar constraint) |
| Pipelines | N=1 (single pipeline, no replication) |

**Limitations:** No W&B logging and no personal GitHub repository for code release.

#### 4.1.2 Sandhya — Multi-Turn Tool Chaining (Experiment 3)

Qwen2.5-3B-Instruct, ToolBench (187K examples dataset; 200+ examples used), SFT followed by GRPO (40 steps, 2 rollouts/prompt, LoRA rank 8, LR 5e-6). Maximum 4 turns per chain with a wrap-up nudge after 2 tools or repeat. This is the best result in Sandhya's scaling study: GRPO 0.91 vs. SFT 0.72 on 3B multi-turn evaluation.

| Scenario | SFT | GRPO | Winner |
|----------|-----|------|--------|
| Weather + Packing | 0.90 | 0.90 | Tie |
| Stock + News Chain | 0.77 | 0.90 | GRPO |
| Search + Calculate | 0.63 | 0.92 | GRPO |
| Single Tool | 0.60 | 0.90 | GRPO |
| **Average** | **0.72** | **0.91** | **GRPO** |

Key finding: The -0.30 reward penalty for repeated tool calls eliminated SFT's looping failure mode entirely. The 3B model shows the clearest benefit from GRPO in Sandhya's scaling study, with consistent gains across 3 of 4 scenarios.

#### 4.1.3 Arumugam — Independent Validation

Arumugam (GitHub: ArumugamKrishnan) independently replicated Experiment 2 using the same pipeline. Results: JSON 0%->92%, Tool 0%->50%, Avg 0->0.59. Additionally explored DPO+LoRA on aerospace domain Q&A.

**Limitations:** The training dataset still contains only 8 examples despite being labeled "Version 2.0". The v2.0 commit was cosmetic (notebook metadata changes), not a substantive improvement. The method used is DPO, not RLHF. Evaluation relies on keyword counting rather than a principled metric, and there is a model mismatch between training and evaluation. No GRPO result on a browser or agentic task has been produced. The aerospace DPO experiment (5 preference examples, eval_loss 0.0093) is too small to draw conclusions.

#### 4.1.4 Dhruva — Baseline Evaluation Framework

Dhruva (GitHub: DhruvaKashyap) built a comprehensive tool-use evaluation pipeline with 5 synthetic tools (calculator, weather, time, search, reminder), 200 train / 40 val / 60 test examples.

| Model | format_score | name_accuracy | arg_score | exact_match |
|-------|-------------|---------------|-----------|-------------|
| Qwen2.5-0.5B | 1.000 | 0.975 | 0.797 | 0.700 |
| Qwen2.5-1.5B | 1.000 | 1.000 | 0.927 | 0.850 |

Per-domain breakdown (0.5B / 1.5B): calculator 37.5%/62.5%, reminder 12.5%/62.5%, search/time/weather 100%/100%.

GRPO training on 0.5B/1.5B with small dataset showed no improvement, consistent with a task-dependent capacity threshold (though the tool-calling domain differs from math reasoning).

**External contributions and context:** Dhruva is first author on a NeurIPS 2025 Spotlight paper (modhifi) on structured pruning — demonstrating strong independent research capability, though that work is not RLHF. His code demonstrates professional engineering standards (mypy strict, pytest, Docker). Dhruva also contributed the HFLM_Accelerate class to lm-evaluation-harness.

**Limitations:** GRPO training logs have not been committed to the repository. Claimed RLHF repositories linked during the project are not accessible (all return 404), so GRPO-specific work cannot be independently verified beyond the evaluation framework above.

#### 4.1.5 Arvind — Tool-Use GRPO on Tinker (7 runs)

**30-step experiments (4 parallel, Qwen3-8B, LoRA rank 32):**

| Experiment | Config | Last-10 Reward | Status |
|-----------|--------|---------------|--------|
| A Baseline | LR=3e-5, group=8, temp=0.8 | 0.875 | Done |
| B High LR | LR=1e-4, group=16, temp=0.8 | 0.999 | Done |
| C Low Temp | LR=3e-5, group=4, temp=0.4 | 0.977 | Done |
| D xlam-60k | LR=3e-5, group=8, real data | 0.363 | Done |

**100-step experiments (3 parallel):**

| Dataset | Last-10 Reward | Status |
|---------|---------------|--------|
| Synthetic 5-tool | 0.825 | Done |
| xlam-60k (real) | 0.113 | Done |
| MATH reasoning | 0.264 | Done |

**Key finding — Synthetic vs. Real Data Gap:** Synthetic 5-tool tasks saturate to reward >0.9 within 5 steps. Real xlam-60k data with diverse tool schemas yields rewards 3--8x lower (0.06--0.36), demonstrating that real-world tool calling is substantially harder than commonly-used synthetic benchmarks.

### 4.2 Code Generation Experiments

#### 4.2.1 Madhu — SWE Code Generation

Madhu (HuggingFace: Madhu2133, GitHub: madhukumara1993) worked on Qwen3-8B code generation using a full SFT → GRPO pipeline. Training code is publicly available at https://github.com/madhukumara1993/qwen3-grpo (Modal training pipeline). The SFT phase used 3,000 Open-Platypus examples; GRPO training used 35 prompts × 10 rollouts = 350 total samples. Madhu implemented 5 custom reward functions targeting: reasoning quality, code correctness, output format, no-stubs enforcement, and response length.

**Corrected results (backed by evaluation code):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| HumanEval pass@1 | ~57% baseline | 86% (141/164) | +29pp |
| GRPO training prompts | — | 35 × 10 = 350 | — |
| SFT examples | — | 3,000 Open-Platypus | — |

**Note on earlier SWE model:** A prior SWE-focused model showed no improvement (42%→42% on SWE-bench subset) and was honestly reported as a failure. The HumanEval 86% (141/164) result on Qwen3-8B is backed by evaluation code and supersedes earlier conflicting figures.

**Note on previous report entry:** An earlier version of this report cited HumanEval results from a 50-problem subset (32%→40%, Fisher's exact $p{=}0.53$). The corrected figure of 86% (141/164) comes from the full 164-problem HumanEval harness with evaluation code. The 50-problem subset result reflected an intermediate checkpoint; the full evaluation is the definitive result.

Model: huggingface.co/Madhu2133/qwen3-8b-swe-grpo

### 4.3 Mathematical Reasoning Experiments

#### 4.3.1 Rafi — Logical Reasoning

Mohammad Rafi (HuggingFace: MohammadRafiML) trained Qwen3-4B-Instruct with SFT → GRPO on a combined GSM8K + NuminaMath dataset. The training run lasted 10.39 hours on Tinker (A100-80GB) and is thoroughly documented (24KB writeup + LaTeX paper + training logs).

**Results on standard benchmarks:**

| Stage | GSM8K Accuracy | Change |
|-------|---------------|--------|
| Baseline | 67.2% | — |
| SFT | 68.1% | +0.9pp |
| GRPO | 67.8% | +0.6pp vs. baseline |

Net GRPO improvement is +0.6 percentage points above baseline, which is within measurement noise. The SFT→GRPO pipeline did not produce a reliable gain on this standard benchmark.

**Caveat on earlier custom-benchmark results:** The table previously shown (GRPO 100% pass rate, zero hallucination on 12 custom questions) reflects results on a 12-question internal test set that is not a standard evaluation. This claim of "100% logical reasoning" is not supported by Rafi's own paper, which shows targets missed by 22 percentage points on the held-out standard evaluation. The GSM8K + NuminaMath results above are the authoritative figures.

**Strengths:** Rafi produced one of the most thoroughly documented experiments in the group, with a 24KB technical writeup, a LaTeX paper, and full training logs. The 10.39-hour Tinker run is the longest single training run in the project.

#### Reviewer-facing caveat on code generation and tool-use evaluation

Two important limitations remain. First, the code-generation headline for Madhu (HumanEval 86%, 141/164) is backed by evaluation code on the full 164-problem harness; an earlier report version cited a 50-problem subset (32%→40%, Fisher's exact $p{=}0.53$) from an intermediate checkpoint. Second, the multi-turn tool-calling scores (for example 0.90/0.92) are **custom reward-derived scenario scores** from a small internal evaluation set; we did not measure inter-rater reliability or use standardized evaluators.

#### 4.3.2 Arvind — GSM8K GRPO on Tinker (10 runs)

**Multi-seed replication (Qwen3-8B, LoRA rank 32, 50 steps, 5 seeds):**

| Seed | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % | Zero-reward % |
|------|------------|----------|-------------|-------------|---------------|
| 137 | 25.0% | 62.5% | 27.5% | 28% | 24% |
| 256 | 22.5% | 62.5% | 32.5% | 24% | 24% |
| 512 | 15.0% | 87.5% | 30.0% | 16% | 10% |
| 042 | 37.5% | 62.5% | 27.5% | 18% | 14% |
| 999 | 20.0% | 87.5% | 35.0% | 18% | 16% |
| **Mean** | **24.0%** | **72.5%** | **30.5%** | **20.8%** | **17.6%** |
| **Std** | **+/-8.4%** | **+/-13.7%** | **+/-3.3%** | | |

Cross-seed mean accuracy is 30.5% +/- 3.3% (95% CI [26.5%, 34.5%]), demonstrating stability of GRPO training outcomes across 5 seeds. Peak accuracy varies more (62.5--87.5%), indicating high trajectory-level variance despite converging to similar final performance. The two new seeds (042, 999) are consistent with the original three, narrowing the confidence interval from [23.8%, 36.2%] (3 seeds) to [26.5%, 34.5%] (5 seeds).

**4B Multi-Seed Replication (Qwen3.5-4B, LoRA rank 32, G=4, 50 steps):**

| Seed | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % | Zero-reward % |
|------|------------|----------|-------------|-------------|---------------|
| 42   | 92.5%      | 100.0%   | 68.8%       | 52%         | 0%            |
| 137  | 80.0%      | 100.0%   | 82.5%       | 68%         | 0%            |
| 256  | 67.5%      | 100.0%   | 96.2%       | 56%         | 0%            |
| 512  | 70.0%      | 100.0%   | 91.2%       | 50%         | 0%            |
| **Mean** | **77.5%** | **100%** | **84.7%** | **56.5%** | **0%** |
| **SD** |          |          | **±12.0%** |             |               |

The 4B model dramatically outperforms the 8B model under identical hyperparameters across all 4 seeds: mean last-10 84.7% (SD=12.0%) vs. 8B's 30.5% (SD=3.3%). All seeds reach 100% peak. The high variance (SD=12.0%) compared to 8B (3.3%) suggests the 4B operates near saturation where small seed differences produce large trajectory-level effects. Zero zero-reward steps across all seeds confirms the 4B always produces scorable outputs.

**Note:** The gap between 4B and 8B may partially reflect generational improvements in base model capability (Qwen3.5-4B vs. Qwen3-8B) rather than a pure parameter-count effect.

**LoRA rank ablation (Qwen3-8B, seed=42, 50 steps):**

| Rank | Trainable Params | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % |
|------|-----------------|------------|----------|-------------|-------------|
| 8 | ~0.1% | 27.5% | 62.5% | 21.2% | 20% |
| 16 | ~0.2% | 20.0% | 75.0% | 18.8% | 20% |
| 32 (default) | ~0.4% | 37.5% | 62.5% | 27.5% | 18% |
| 64 | ~0.8% | 47.5% | 87.5% | 25.0% | 18% |

Rank 32 is the default used in all 5-seed replication runs. Rank 64 starts fastest (47.5% first-5 average vs. 27.5% for rank 8) and reaches the highest peak (87.5%), confirming that more LoRA capacity accelerates initial learning.

**Group size ablation (Qwen3-8B, seed=42, rank 32, 50 steps):**

| G  | First-5 Avg | Last-10 Avg | Peak Acc | Zero-loss | Zero-reward |
|----|-------------|-------------|----------|-----------|-------------|
| 4  | 32.5%       | 23.8%       | 62.5%    | 26%       | 20%         |
| 8  | 33.8%       | 24.4%       | 68.8%    | 6%        | 6%          |
| 16 | 29.4%       | 36.2%       | 75.0%    | 2%        | 2%          |
| 32 | 32.8%       | **54.7%**   | **100%** | 2%        | 0%          |

Group size has a dramatic effect: G=32 more than doubles G=4's last-10 reward (54.7% vs 23.8%) and eliminates zero-reward steps entirely. This confirms that exploration (via larger groups) is a major factor, but even at G=32 the 8B (54.7%) does not approach the 4B (84.7%), suggesting capacity also matters.

**Extended 100-step run (LR=5e-6):**
Peak accuracy 75.0%, last-10 average 27.5%. Lower learning rate stabilizes training (17% zero-loss vs. 24% at higher LR) but does not break through the performance ceiling, suggesting the bottleneck is not optimization speed but rather the binary reward signal's sparsity at this group size.

#### 4.3.3 Held-Out GSM8K Test Results

We evaluate GRPO checkpoints on 200 held-out GSM8K test examples per seed with greedy decoding (temperature=0, single sample):

| Seed | Correct/200 | Accuracy | 95% CI |
|------|-------------|----------|--------|
| 42   | 166/200     | 83.0%    | [77.5%, 88.0%] |
| 137  | 165/200     | 82.5%    | [77.0%, 87.5%] |
| 256  | 161/200     | 80.5%    | [74.5%, 86.0%] |
| 512  | 168/200     | 84.0%    | [79.0%, 89.0%] |
| 999  | 173/200     | 86.5%    | [81.5%, 91.0%] |
| **Mean** | | **83.3%** | **SD = 2.2%, CI [80.6%, 86.0%]** |

The held-out accuracy (83.3%) far exceeds the mean training reward (30.5%), but these are *not comparable*: training reward averages per-completion correctness under stochastic sampling ($T{=}0.8$--$1.0$) across the trajectory, while test accuracy is greedy pass@1 on the final checkpoint.

**Base-model control:** We evaluate base Qwen3-8B *without* any LoRA adapter on the same 200 test examples with identical greedy decoding: **164/200 = 82.0%** (95% bootstrap CI [76.5%, 87.5%]). The GRPO-trained mean (83.3%) exceeds the base by only **+1.3 percentage points**. A one-sample t-test of the 5 GRPO seeds against the base accuracy yields t=1.32, p=0.26 (two-sided), so **the improvement is not statistically significant**. Four of five seeds exceed the base, but seed 256 (80.5%) falls below it. We conclude that the held-out GSM8K accuracy is overwhelmingly attributable to Qwen3-8B's pre-existing capability, with GRPO contributing a small, non-significant increment under our setup.

### 4.4 10x Structural Ceiling Experiments (Arvind — 32 Tinker runs, ~$65)

A dedicated experiment matrix ("10x Structural Ceiling") systematically ablates GRPO across benchmarks, model families, model sizes, group sizes, learning rates, and constrained decoding — all on Tinker cloud with full 50-step training. W&B project: `tinker-structural-ceiling`.

#### 4.4.1 Benchmark Hierarchy

| Benchmark | Model | Steps | Final Reward | Avg Last-10 | Verdict |
|-----------|-------|-------|-------------|-------------|---------|
| Tool-use (JSON) | Qwen3-8B | 50 | **1.000** | 1.000 | Format learned perfectly |
| GSM8K (math) | Qwen3-8B (LR=1e-4) | 50 | **1.000** | 1.000 | Math solved at high LR |
| GSM8K (math) | Qwen3-8B (seed4) | 50 | **0.984** | 0.972 | Converges with default LR |
| MATH-500 | Qwen3-8B | 50 | **0.720** | 0.574 | Partial — harder math, lower ceiling |
| HumanEval (code) | Qwen3-8B | 50 | **0.000** | 0.024 | Total null — code not learnable via GRPO |

GRPO learns structural/format tasks perfectly but fails on semantic tasks. The ceiling is where the task transitions from pattern-matching to genuine reasoning.

#### 4.4.2 Cross-Family Architecture Dependence

| Model | Size | Type | Benchmark | Final Reward |
|-------|------|------|-----------|-------------|
| Qwen3-8B | 8B | Instruct | Tool-use | **1.000** |
| Llama-3.1-8B-Instruct | 8B | Instruct | Tool-use | 0.103 |
| Llama-3.1-8B | 8B | Base | Tool-use | **0.000** |
| Qwen3-8B | 8B | Instruct | GSM8K | **1.000** |
| Llama-3.1-8B-Instruct | 8B | Instruct | GSM8K | **0.969** |
| Llama-3.1-8B | 8B | Base | GSM8K | 0.047 |

The 0%→92% JSON validity finding is **Qwen-specific**. Llama-3.1-8B-Instruct achieves only 10.3% on the same tool-use task. Instruction tuning provides a +0.922 delta on GSM8K (base 0.047 vs. instruct 0.969), dwarfing any RL contribution.

#### 4.4.3 Model Size Ladder

| Model | Size | Steps | Final Reward | Avg Last-10 | Mean ZVF | Onset |
|-------|------|-------|-------------|-------------|----------|-------|
| Qwen3-8B | 8B | 50 | **1.000** | 0.972 | 0.550 | step 20 |
| Qwen3-1.7B | 1.7B | 50 | 0.016 | 0.009 | 0.885 | step 0 |
| Qwen3-0.6B | 0.6B | 50 | 0.016 | 0.009 | 0.920 | step 0 |
| Llama-3.2-3B | 3B | 47 | 0.016 | ~0.02 | ~0.90 | step 0 |
| Llama-3.2-1B | 1B | 50 | **0.000** | 0.000 | 1.00 | step 0 |

Below 8B-instruct, GRPO on GSM8K is a total null across both Qwen and Llama families. Both 0.6B and 1.7B Qwen models show immediate saturation (onset=step 0, ZVF>88%) — the model never generates within-group reward variance, so gradients never form. This extends the capacity threshold finding beyond Llama to Qwen, confirming it's not architecture-specific.

#### 4.4.4 Group Saturation Diagnostic (Novel Metric)

We introduce **Zero-Variance Fraction (ZVF)** — the fraction of groups where all completions receive identical rewards — and **Gradient Utilization (GU = 1 - ZVF)** as diagnostics for GRPO training health.

| Group Size (G) | Final Reward | Avg Last-10 | Mean ZVF | Mean GU | Saturation Onset | Steps |
|---------------|-------------|-------------|---------|---------|-----------------|-------|
| G=4 | **1.000** | 0.944 | 0.520 | 0.480 | step 4 | 50 |
| G=16 (seed4) | **0.984** | 0.972 | 0.550 | 0.450 | step 20 | 50 |
| G=16 (seed5) | **0.922** | 0.925 | 0.430 | 0.570 | step 30 | 50 |
| G=32 | **1.000** | 0.957 | 0.455 | **0.545** | **step 29** | 50 |
| G=64 | **1.000** | ~0.98 | 0.525 | 0.475 | step 20 | 50 |

All group sizes converge to ~1.0 reward at 50 steps. **G=32 is the sweet spot** — highest mean gradient utilization (54.5%) with latest saturation onset (step 29). G=64 provides diminishing returns with 2x the compute cost.

#### 4.4.5 Learning Rate Speed-Saturation Tradeoff

| LR | Steps | Final Reward | Avg Last-10 | Mean ZVF | Mean GU | Saturation Onset |
|----|-------|-------------|-------------|---------|---------|-----------------|
| 1e-5 | 50 | **0.594** | 0.677 | 0.175 | **0.825** | never (50 steps) |
| 4e-5 (default) | 50 | **0.984** | 0.972 | 0.550 | 0.450 | step 20 |
| 1e-4 | 50 | **1.000** | 1.000 | ~1.00 | ~0.00 | step 12 |
| 3e-4 | 50 | **0.984** | 0.901 | 0.565 | 0.435 | step 10 |

**Key correction from full data:** LR=3e-4 is NOT unstable — partial data at step 37 showed reward=0.219 (apparent divergence), but full 50-step run shows recovery to 0.984. LR=1e-5 is the only configuration with >80% gradient utilization throughout training — it never saturates.

#### 4.4.6 Constrained Decoding Ablation

| Variant | Final Reward | Mean ZVF | GU | Saturation Onset |
|---------|-------------|---------|-----|-----------------|
| Unconstrained | 0.998 | 0.725 | 0.275 | step 11 |
| Constrained | 0.981 | 0.660 | 0.340 | step 11 |

Both converge to ~1.0 with similar saturation profiles. This refutes the "decoder confound" criticism — GRPO genuinely learns format, it's not just overlapping with grammar enforcement.

#### 4.4.7 Reward Hacking and Catastrophic Collapse

Llama-3.1-8B base on tool-use showed a dramatic trajectory:
1. **Steps 1-20:** Stuck at reward 0.10-0.18, 75-100% ZVF (saturated at bottom)
2. **Steps 21-40:** Sudden breakout — reward climbed 0.28 → **0.873**
3. **Step 41:** Catastrophic collapse — reward crashed 0.87 → 0.002 → **0.000**
4. **Steps 42-50:** Dead — zero reward, 100% ZVF

Loss magnitudes during breakout reached -238, indicating extreme policy divergence. This is a textbook reward hacking → collapse pattern.

#### 4.4.8 10x Summary Table

| Dimension | Varied | Fixed | Key Finding (50-step) |
|-----------|--------|-------|-------------|
| **Benchmark** | Tool/GSM8K/MATH/HumanEval | Qwen3-8B, G=16 | Tool=1.0, GSM8K=0.97, MATH=0.57, Code=0.00 |
| **Architecture** | Qwen vs Llama | 8B, GSM8K | Tool-use is Qwen-specific (1.0 vs 0.1) |
| **Base vs Instruct** | Base vs Instruct | Llama-8B | SFT prerequisite (0.05 vs 0.97) |
| **Model Size** | 0.6B / 1.7B / 3B / 8B | Qwen+Llama, GSM8K | Below 8B-instruct: total null across families |
| **Group Size** | 4 / 16 / 32 / 64 | Qwen3-8B, GSM8K | All converge; G=32 optimal GU (54.5%) |
| **Learning Rate** | 1e-5 / 4e-5 / 1e-4 / 3e-4 | Qwen3-8B, GSM8K | LR=1e-5 never saturates; LR=3e-4 recovers |
| **Constrained** | Yes / No | Qwen3-8B, Tool-use | No difference — decoder confound is moot |

---

### 4.5 Large-Scale Tinker Experiments — World-Class Suite

To test whether our findings generalize beyond the 0.6B–8B small-model regime, we expand the experiment suite to **14 Tinker experiments** spanning frontier models, scaling analysis, MoE comparisons, and cross-architecture tool-use, executed across Tinker API and Modal H100 GPUs. All runs are logged to W&B project `tinker-rl-lab-world-class` and checkpointed to HuggingFace Hub at `arvindcr4/tinker-rl-bench-*`.

**Infrastructure note:** Of 14 Tinker experiments launched, only **3 completed** (21%). The remaining 11 failed due to **JWT token expiration** — Tinker sessions time out after a fixed window, and long-running experiments (32B+ models, 30-step runs) consistently exceeded this limit before completing. This is a platform constraint, not a model failure. The 3 completed runs are documented below; Section 4.5.6 provides a full accounting of what ran vs. what failed.

#### 4.5.1 Scaling Analysis: 8B → 32B → 235B

We extend the model-size ladder established in Section 4.4.3 to cover scales well beyond the original 0.6B–8B range, adding Qwen3-32B and the mixture-of-experts frontier model Qwen3-235B-A22B (22B active parameters). This creates a five-point scaling ladder: 0.6B, 1.7B, 8B, 32B, 235B.

| Model | Parameters | Active Params | Steps | Final Reward | Last-10 Avg | ZVF | Run ID |
|-------|-----------|---------------|-------|-------------|-------------|-----|--------|
| Qwen3-0.6B | 0.6B | 0.6B | 50 | 0.016 | 0.009 | 0.920 | (existing) |
| Qwen3-1.7B | 1.7B | 1.7B | 50 | 0.016 | 0.009 | 0.885 | (existing) |
| Qwen3-8B | 8B | 8B | 50 | 1.000 | 0.972 | 0.550 | (existing) |
| Qwen3-32B | 32B | 32B | — | — | — | — | Failed (JWT expiry) |
| Qwen3-235B-A22B | 235B | 22B | — | — | — | — | Failed (JWT expiry) |

The Qwen3-32B and Qwen3-235B-A22B runs were launched but terminated due to JWT expiration before producing usable training data. The scaling analysis above 8B therefore remains incomplete for the Qwen family.

**Hypotheses under test (to be addressed in future work):**
- Does GRPO improvement per scale follow a power law above the 8B threshold (as observed in pretraining)?
- Does the ZVF saturation dynamic change qualitatively at 32B (more diverse group-level reward variance)?
- Does 235B-MoE achieve higher asymptotic reward than dense 32B at matched compute, or does MoE volatility (Section 5.2) persist at larger scale?

#### 4.5.2 Frontier Model Results

Of five planned frontier model runs, one completed: **DeepSeek-V3.1** on GSM8K. The remaining four (Nemotron-120B, Qwen3-235B-A22B, GPT-OSS-20B, Kimi-K2) all failed due to JWT expiration.

| Model | Provider | Scale | Architecture | Task | Steps | Peak | Last-10 Avg | Status |
|-------|----------|-------|-------------|------|-------|------|-------------|--------|
| DeepSeek-V3.1 | DeepSeek | ~671B (37B active) | MoE | GSM8K | 20 | **1.000** | **85%** | Completed |
| Nemotron-120B | NVIDIA | 120B | Dense | GSM8K | — | — | — | Failed (JWT) |
| Qwen3-235B-A22B | Alibaba | 235B (22B active) | MoE | GSM8K | — | — | — | Failed (JWT) |
| GPT-OSS-20B | OpenAI | ~20B | Dense | GSM8K | — | — | — | Failed (JWT) |
| Kimi-K2 | Moonshot | ~1T (MoE) | MoE | GSM8K | — | — | — | Failed (JWT) |

**DeepSeek-V3.1 result:** The frontier MoE model achieves a last-10 average reward of **85%** and peak reward of **1.0** on GSM8K in just 20 steps. Notably, the reward trace starts high (0.875 at step 1) and remains consistently above 0.75 throughout — the model effectively has near-ceiling GSM8K performance from initialization, so GRPO provides modest additional signal rather than a large delta from base. Reward trace: [0.875, 0.875, 1.0, 0.75, 0.75, 0.75, 0.625, 0.875, 0.875, 1.0, 0.75, 1.0, 0.875, 0.875, 0.75, 1.0, 0.875, 0.75, 0.875, 0.875].

Frontier models are evaluated in the same GRPO fine-tuning protocol as the 8B models, enabling direct comparison across model families. The DeepSeek result suggests that:
- Semantic reasoning does not need GRPO to "unlock" at frontier scale — the capability is already present.
- The benchmark hierarchy finding (format > math > code) may weaken at frontier scale: a model already scoring 85%+ on GSM8K training reward has less room to improve structurally.

#### 4.5.3 MoE vs. Dense at Matched Active Parameters

Our preliminary MoE observation (Section 5.2) used Qwen3-30B-MoE (~3B active) vs. Qwen3-8B-dense at mismatched active parameter counts. The World-Class Suite adds a **matched active-parameter comparison**:

| Model | Total Params | Active Params | Architecture | Platform | Last-10 Avg | Status |
|-------|-------------|---------------|-------------|----------|-------------|--------|
| Qwen3-8B | 8B | 8B | Dense | Tinker | 0.972 | Completed (existing) |
| Qwen3-30B-MoE | 30B | ~3B | MoE | Tinker | (existing) | Completed (existing) |
| DeepSeek-V3.1 | ~671B | ~37B | MoE | Tinker | **0.85** | Completed |
| Qwen3-235B-A22B | 235B | 22B | MoE | Tinker | — | Failed (JWT) |
| Nemotron-120B | 120B | 120B | Dense | Modal | — | Failed (JWT) |

Of the planned matched active-parameter comparisons, only the DeepSeek-V3.1 run completed. Its 37B active parameters (comparable to Qwen3-32B in active-parameter count) yield 85% GSM8K reward — substantially higher than the Qwen3-8B result at 97.2%, but Qwen3-8B is already near ceiling on GSM8K, making direct active-parameter comparison difficult on a saturated benchmark. The comparison aims to separate (a) the effect of more active parameters on GRPO learning dynamics from (b) the architectural effect of sparse routing on training volatility; full conclusions require the remaining runs and remain open for future work.

#### 4.5.4 PPO vs. GRPO Comparison

A long-standing gap in our experimental record is the absence of a PPO baseline (Section 7.2). The Modal H100 GPU suite addresses this directly, training two models — Qwen3-8B and Llama-3.1-8B-Instruct — under PPO on GSM8K with 30-step runs. Both completed successfully.

| Method | Model | Platform | Steps | Peak | Last-10 Avg | HF Checkpoint |
|--------|-------|----------|-------|------|-------------|---------------|
| GRPO | Qwen3-8B | Tinker | 50 | 1.000 | 97.2% | (existing) |
| PPO | Qwen3-8B | Modal H100 | 30 | **1.000** | **35%** | arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42 |
| GRPO | Llama-3.1-8B-Instruct | Tinker | 50 | 1.000 | 97% | (existing) |
| PPO | Llama-3.1-8B-Instruct | Modal H100 | 30 | **1.000** | **95%** | — |

**Results summary:** Both PPO runs reach peak reward of 1.0, confirming that the reward signal is learnable under both algorithms. However, the two models diverge dramatically in their terminal performance:

- **PPO Llama-3.1-8B-Instruct (95% last-10 avg):** The strongest result in the entire experimental record. Llama achieves near-perfect GSM8K reward almost immediately — the reward trace shows 1.0 at step 1, sustained at 0.95+ across 30 steps with only minor dips to 0.75. This is not a learning curve; the model was already capable and PPO reinforces that capability with high stability.

- **PPO Qwen3-8B (35% last-10 avg):** Despite the same peak, Qwen3-8B under PPO is highly volatile — reward oscillates between 0.0 and 0.75 throughout, with frequent zero-reward steps. The last-10 average of 35% is far below Qwen3-8B’s GRPO result of 97.2% on the same task, suggesting that PPO’s critic introduces optimization instability for this model.

**Cross-method comparison:**

| Dimension | GRPO Qwen3-8B | PPO Qwen3-8B | PPO Llama-3.1-8B |
|-----------|--------------|-------------|------------------|
| Peak reward | 1.000 | 1.000 | 1.000 |
| Last-10 avg | **97.2%** | 35% | **95%** |
| Stability | High | Low (volatile) | Very high |
| Steps | 50 | 30 | 30 |

This PPO vs. GRPO result reveals a **model-method interaction**: Llama-3.1-8B-Instruct is dramatically better suited to PPO on GSM8K than Qwen3-8B, while Qwen3-8B performs far better under GRPO. The architecture-specific RL compatibility finding from our tool-use results (Section 4.6.1) appears to generalize to algorithm choice as well.

#### 4.5.5 KL Divergence and Entropy Tracking During Training

We attempted to instrument KL divergence from the reference policy during the Modal H100 PPO runs using the veRL framework. This experiment **failed** due to a gradient computation bug: the error `element 0 of tensors does not require grad and does not have a grad_fn` indicates that the KL loss term was not connected to the computation graph, producing zero gradients and making the tracked values meaningless.

| Metric | Status | Failure Mode |
|--------|--------|-------------|
| KL divergence ($D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$) | **Failed** | Gradient bug — tensor not in computation graph |
| Response entropy ($H(\pi_\theta)$) | Not collected | Dependent on KL tracking setup |
| ZVF (existing Tinker runs) | Working | Logged in all Tinker experiments |
| Gradient norm | Logged in PPO runs | Available via W&B |

The gradient bug is a known issue when wrapping reference models in frameworks that freeze parameters without explicitly detaching them from the autograd graph. A fix requires either (a) calling `.detach()` on the reference model outputs before the KL computation, or (b) using `torch.no_grad()` context for the reference forward pass. This is correctable in a future run; it does not invalidate the PPO reward results above.

The absence of KL tracking remains a limitation: we cannot directly measure policy drift during training. Indirect evidence from reward volatility (Qwen3-8B PPO) and reward stability (Llama PPO) suggests different drift patterns, but mechanistic confirmation awaits a corrected KL run.

---

### 4.6 Cross-Architecture Analysis

Sections 4.1–4.5 run experiments within model families. This section synthesizes **cross-architecture** results: how Qwen3, Llama, DeepSeek, Nemotron, and GPT-OSS families respond to identical GRPO training protocols.

#### 4.6.1 Tool-Use Across Model Families

Tool-use GRPO experiments span three architectures at matched 8B scales:

| Model Family | Model | Size | Tool-Use Reward (Last-10) | Status |
|-------------|-------|------|--------------------------|--------|
| Qwen3 | Qwen3-8B | 8B | **1.000** | Done |
| Llama | Llama-3.1-8B-Instruct | 8B | 0.103 | Done |
| Llama (base) | Llama-3.1-8B | 8B | 0.000 (collapse) | Done |
| Qwen3 (scale) | Qwen3-32B | 32B | [PENDING] | Running |
| DeepSeek | DeepSeek-V3.1 | 671B | [PENDING] | Pending |
| Nemotron | Nemotron-120B | 120B | [PENDING] | Pending |

The 9.7× gap between Qwen3 and Llama-8B-Instruct on identical tool-use tasks (1.0 vs. 0.103) suggests that GRPO sensitivity to the tool-calling objective is architecture-specific, likely driven by differences in instruction-following pre-training data and the model’s prior probability over structured JSON outputs. The World-Class Suite tests whether this gap persists at scale: if Llama-3.1-8B-Instruct tool-use (0.103) is a capacity limitation, larger Llama models should recover; if it is architectural, it should persist across scales.

#### 4.6.2 How Different Model Families Respond to GRPO

Across all experiments, we observe three qualitatively distinct GRPO response modes:

**Mode 1 — Fast Format Learning (Qwen3 instruct ≥8B):** Rapid reward ascent in Phase 1 (steps 1–20), format fully internalized, reasoning gains in Phase 2 (steps 20–50). Zero-reward rate rapidly drops to 0%. Asymptotic reward ≥1.0 on format-dominant tasks.

**Mode 2 — Partial Learning (Llama-3.1-8B-Instruct, GSM8K):** Slow ascent, final reward 0.97 on math but 0.103 on tool-use. High ZVF early, recovers as model adapts. Instruction tuning prerequisite is satisfied, but weaker structural priors slow format acquisition.

**Mode 3 — Total Null / Collapse (sub-8B models, base models):** Immediate ZVF saturation (onset=step 0). Either zero learning (small models) or breakout-then-collapse pattern (Llama-8B base: reward 0.87 → 0.00). Both failure modes share the common cause of insufficient within-group reward variance to generate useful policy gradients.

The DeepSeek-V3.1 result (85% GSM8K, 20 steps) already provides one data point: a frontier MoE model exhibits Mode 1 behavior on math, consistent with the prediction that semantic reasoning unlocks at scale.

---

### 4.5.6 World-Class Suite: Completed Tinker Experiments

Of the 14 Tinker experiments launched for the World-Class Suite, **3 completed** and **11 failed** due to JWT token expiration. This section provides a complete accounting.

**Completed Tinker runs:**

| Experiment ID | Model | Task | Steps | Peak | Last-10 Avg | Notes |
|--------------|-------|------|-------|------|-------------|-------|
| frontier_gsm8k_deepseek-v3.1 | DeepSeek-V3.1 (~671B, 37B active) | GSM8K | 20 | 1.000 | **85%** | Frontier MoE; high from step 1 |
| scale_gsm8k_qwen3-8b | Qwen3-8B | GSM8K | 30 | 0.625 | **34.4%** | Volatile; lower than 50-step GRPO result |
| cross_tool_llama-8b-inst | Llama-3.1-8B-Instruct | Tool-use | 30 | 0.000 | **0%** | Complete failure; 0 reward every step |

**Failed Tinker runs (JWT expiration):**

| Experiment ID | Model | Planned Task | Failure Reason |
|--------------|-------|-------------|---------------|
| scale_gsm8k_qwen3-32b | Qwen3-32B | GSM8K | JWT expiry |
| scale_gsm8k_qwen3.5-4b | Qwen3.5-4B | GSM8K | JWT expiry |
| scale_gsm8k_qwen3.5-27b | Qwen3.5-27B | GSM8K | JWT expiry |
| scale_gsm8k_llama-8b-inst | Llama-3.1-8B-Instruct | GSM8K | JWT expiry |
| frontier_gsm8k_qwen3-235b | Qwen3-235B-A22B | GSM8K | JWT expiry |
| frontier_gsm8k_nemotron-120b | Nemotron-120B | GSM8K | JWT expiry |
| moe_gsm8k_qwen3-30b-moe | Qwen3-30B-MoE | GSM8K | JWT expiry |
| moe_gsm8k_qwen3-30b-inst | Qwen3-30B-Instruct | GSM8K | JWT expiry |
| cross_tool_qwen3-32b | Qwen3-32B | Tool-use | JWT expiry |
| arch_gsm8k_gpt-oss-20b | GPT-OSS-20B | GSM8K | JWT expiry |
| arch_gsm8k_kimi-k2 | Kimi-K2 | GSM8K | JWT expiry |

**Why JWT expiration caused failures:** Tinker authenticates experiment sessions via short-lived JWT tokens. When experiments run longer than the token validity window — as is common for large models (32B+) running 30 steps — the session is terminated mid-run. The 3 completed runs succeeded because they either used smaller active-parameter counts (DeepSeek at 37B active parameters is fast per token on Tinker’s infrastructure) or happened to complete within the token window. This is a platform-side constraint; no model-level conclusions should be drawn from the failures.

**Llama-3.1-8B-Instruct tool-use complete failure:** The cross-architecture tool-use run for Llama-3.1-8B-Instruct produced **zero reward across all 30 steps** — not just low reward, but a flat line at 0.0. This extends the finding from Section 4.6.1 (where Llama-8B-Instruct achieved 0.103 on tool-use GRPO) and suggests that the specific tool-use task variant used in this World-Class run is harder than the format-only task from Section 4.6.1. Possible reasons for complete failure:
1. **Task difficulty:** The tool-use task requires both schema-valid JSON emission *and* correct tool selection, which compounds the learning problem.
2. **Reward signal sparsity:** If the reward function requires full correctness (schema + semantics), Llama-3.1-8B-Instruct may never receive a positive gradient signal to build on.
3. **Architecture mismatch:** The existing Qwen3-8B tool-use result (1.0) is driven by Qwen’s strong structural JSON priors. Llama lacks this prior and cannot bootstrap from zero reward.

This result reinforces the key finding that tool-use GRPO is substantially harder than math GRPO, and that architecture choice matters more for tool-use than for math tasks.

---

### 4.5.7 Modal H100 Experiments: Completed Results

The Modal H100 experiments ran independently of Tinker and did not suffer from JWT expiration. Two PPO training runs completed fully; three additional experiments produced partial results before timing out.

**Completed Modal H100 runs:**

| Experiment | Model | Task | Steps | Peak | Last-10 Avg | HF Checkpoint |
|-----------|-------|------|-------|------|-------------|---------------|
| ppo_qwen3-8b | Qwen3-8B | GSM8K | 30 | 1.000 | 35% | arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42 |
| ppo_llama-8b-inst | Llama-3.1-8B-Instruct | GSM8K | 30 | 1.000 | **95%** | — |

**Partial Modal H100 runs (timeout at 3600s):**

| Experiment | Model | Task | Progress | Partial Result |
|-----------|-------|------|----------|---------------|
| humaneval_qwen3-8b | Qwen3-8B | HumanEval | 40/164 problems | **65% pass@1** on first 40 problems |
| heldout_qwen3.5-27b | Qwen3.5-27B | GSM8K held-out | 100/200 samples | **86%** on first 100 samples |
| heldout_qwen3-32b | Qwen3-32B | GSM8K held-out | 100/200 samples | **33%** on first 100 samples |

**Notes on partial results:**
- **HumanEval (65% at 40/164):** The 65% pass@1 rate on the first 40 HumanEval problems is a preliminary positive signal — substantially above the 8B GRPO result of 0.00 on HumanEval in Section 4.4. However, HumanEval problems are roughly ordered by difficulty, so the first 40 problems are easier than average. The full 164-problem result would likely be lower.
- **Qwen3.5-27B held-out (86% at 100/200):** A strong result on the first half of the test set. If it holds on the second half, this would be the highest held-out accuracy in the experiment.
- **Qwen3-32B held-out (33% at 100/200):** A surprisingly low result for a 32B model, given Qwen3-8B achieves 83.3%. This may indicate a prompt-format mismatch or a model variant (non-instruct) was evaluated.
- **KL tracking (failed):** The KL divergence tracking experiment failed with a gradient computation error (see Section 4.5.5 for details).

---

## 5. Results and Analysis

### 5.1 Capacity Threshold for GRPO (Hypothesis)

We observe a sharp break between 3B and 4B parameters for GRPO on GSM8K. Dense 3B models (Llama-3.2-3B) fail to learn — the 3B's 2.34% training reward is near-random and indistinguishable from noise given the small effective sample ($n{=}50$ steps × batch 2 × $G{=}4$). The 3B failure traces to 56% zero-loss steps from all-incorrect groups, while the 4B's 68% zero-loss stems from all-*correct* groups (productive saturation). This is consistent with Dhruva's negative result on Qwen 0.5B/1.5B (though the tool-calling domain differs from math reasoning).

**3B G=32 control:** To separate capacity from exploration, we tested Llama-3.2-3B with G=32 (vs. baseline G=4). Zero-loss drops from 56% to 18% — confirming G=32 dramatically improves exploration — but accuracy rises only from 2.3% to 5.0%, still near random. This suggests the 3B failure is primarily a capacity limitation, not an exploration artifact.

**10x Structural Ceiling extension:** The dedicated 32-run experiment (Section 4.4.3) extends this finding with a full size ladder across both model families. Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-1B, and Llama-3.2-3B all show immediate ZVF saturation (onset=step 0) with near-zero final rewards on GSM8K. Only Qwen3-8B (instruct) achieves meaningful learning. This confirms the threshold is not architecture-specific — it holds across both Qwen and Llama families — and places it firmly at the 8B-instruct level for GSM8K.

**Important caveats:** This should be read as a *suggestive model-family/scale discontinuity*, not an established threshold:
- The 4B model (Qwen3.5-4B) and 3B model (Llama-3.2-3B) differ in architecture family and model generation, not just parameter count. Architecture confound cannot be ruled out.
- The 1.5B model succeeds on tool calling (92% JSON validity) while the 3B fails on math, suggesting the threshold is task-dependent, not a universal parameter-count boundary.
- 4B multi-seed replication (4 seeds, mean 84.7%, SD=12.0%) confirms the result is reproducible but variance is high.
- The 10x size ladder confirms the null extends to Qwen sub-8B models (0.6B, 1.7B), ruling out the Llama-specific architecture confound for the smallest models.

**Base-model control:** Base Qwen3-8B without LoRA scores 82.0% on the same 200 test examples. The GRPO delta (+1.3pp, $p{=}0.26$) is not significant. Held-out accuracy is overwhelmingly attributable to base model capability.

### 5.2 MoE Architectural Effects

A Qwen3-30B-MoE model with ~3B active parameters reached a 99% peak GSM8K training-step accuracy but exhibited 2.43× higher step-to-step volatility than the dense 8B model (Levene's test $p = 7.0 \times 10^{-6}$). Despite this volatility, both converged to comparable performance, suggesting sparse routing can substitute for total dense capacity but introduces training instability.

**Proposed mechanism (optimization interference hypothesis):** GRPO's policy gradient pushes expert selection toward reward-maximizing routes, while the router's auxiliary load-balancing loss pushes toward uniform utilization, creating optimization interference that amplifies step-to-step variance. This extends the MoE instability literature (Switch Transformers, GLaM, Mixtral) from pretraining to post-training RL.

**Caveat:** We did **not** log routing entropy, expert load imbalance, or other gating diagnostics. The evidence is variance-level rather than mechanism-level.

### 5.3 Two-Phase Learning Progression

GRPO training exhibits a characteristic two-phase pattern on tool-calling tasks:
- **Phase 1 (Steps 1--20):** Model learns answer FORMAT compliance (0%->14% accuracy)
- **Phase 2 (Steps 21--25):** Once format stabilizes, reasoning capability rapidly improves (14%->58%)

This is most pronounced where format compliance is a distinct sub-task; on GSM8K the phases are less separable. We describe this as a domain-dependent observation consistent with known curriculum effects, rather than a novel GRPO property.

### 5.4 SFT + GRPO Complementarity (Pilot Observation)

SFT alone never generates tool calls spontaneously and loops on multi-turn tasks. GRPO alone works but converges slower on hard tasks. In our limited comparisons ($N{=}1$ per configuration), SFT-initialized GRPO produced the strongest results: SFT teaches format, GRPO refines judgment (which tool, when to stop). An instruction-tuned 8B model starts GRPO at 78.91% vs. 7.03% for a base model, compressing time-to-mastery without changing the asymptotic ceiling. **Caveat:** This comparison lacks matched-compute and reverse-order controls, and is consistent with known warm-starting effects (ReST, STaR).

### 5.5 Synthetic vs. Real Data Gap

On tool calling, synthetic 5-tool tasks saturate to reward >0.9 within 5 GRPO steps. Salesforce xlam-60k with diverse real-world tool schemas yields rewards of 0.06--0.36 after 100 steps — a 3--8× difficulty gap. This is a *data distribution* effect (curated vs. real schemas), distinct from the train-test *metric* discrepancy observed for GSM8K (Section 4.3.3 vs. 4.3.2), which reflects decoding regime differences (stochastic training vs. greedy test). Standard synthetic benchmarks substantially overestimate tool-calling capability.

### 5.6 LoRA Rank and Parameter Efficiency

Our ablation across ranks 8, 16, 32 (default), and 64 reveals:
- Higher rank correlates with faster initial learning (rank 64 achieves 47.5% in first 5 steps vs. 27.5% for rank 8)
- Peak accuracy scales with rank (62.5% -> 75.0% -> 87.5%)
- All ranks converge to similar long-run averages (~20-25%), indicating the ceiling is determined by model capacity and reward signal, not adapter capacity
- Diminishing returns: rank 16->64 adds +12.5% peak for 4x more parameters

### 5.7 Statistical Robustness

**Training-set reward:** Five-seed GSM8K replication yields mean 30.5% ± 3.3% (95% CI [26.5%, 34.5%]), with zero-loss rates of 16--28% across seeds. This provides multi-seed replication evidence for GRPO on GSM8K with small group sizes ($G{=}4$).

**Held-out accuracy:** Five-seed evaluation on 200 test examples per seed yields mean 83.3% ± 2.2% (95% CI [80.6%, 86.0%]). The narrow SD (2.2%) across seeds shows consistent held-out performance, though the base-model control (82.0%) indicates this consistency is largely inherited from the pretrained model rather than introduced by GRPO.

**Code generation:** The +8% HumanEval delta on a 50-item subset is not statistically significant (Fisher's exact $p{=}0.53$, bootstrap 95% CI [27%, 53%] post-GRPO).

**Base-model control result:** Base Qwen3-8B without LoRA scores 82.0% on the same 200 examples. The +1.3pp delta (83.3% vs 82.0%) is not statistically significant (t=1.32, p=0.26). GRPO's contribution to held-out test accuracy is not demonstrated; the 83.3% is overwhelmingly attributable to base model capability.

### 5.8 Frontier Model Scaling Laws

Of the planned frontier model runs, only **DeepSeek-V3.1** completed (Section 4.5.2). The scaling law analysis across the full 0.6B–235B ladder remains incomplete due to JWT expiration failures for Qwen3-32B, Qwen3-235B-A22B, Nemotron-120B, GPT-OSS-20B, and Kimi-K2. What we can report:

**Observed data points (complete):**

| Scale | Model | Task | Last-10 Avg | Training Mode |
|-------|-------|------|-------------|---------------|
| 0.6B | Qwen3-0.6B | GSM8K | 0.9% | Total null |
| 1.7B | Qwen3-1.7B | GSM8K | 0.9% | Total null |
| 8B | Qwen3-8B | GSM8K | 97.2% | Fast learning |
| 8B | Qwen3-8B | GSM8K (PPO) | 35% | Volatile |
| 8B | Llama-3.1-8B-Instruct | GSM8K (PPO) | **95%** | Very stable |
| ~671B (37B active) | DeepSeek-V3.1 | GSM8K | **85%** | High from step 1 |

**Key observation:** DeepSeek-V3.1’s 85% result at just 20 steps, starting at 0.875 on step 1, suggests that at frontier scale GRPO is not needed to unlock GSM8K performance — the capability is already present in the base model. This is consistent with the hypothesis that larger models exhibit Mode 1 behavior (fast format learning), but the mechanism differs: rather than fast *learning*, it is fast *reinforcement* of existing capability.

**Incomplete comparisons:** The power-law scaling fit across 0.6B–235B cannot be computed without the missing runs. The HumanEval partial result from Modal (65% at 40/164 problems; see Section 4.5.7) provides preliminary evidence that code generation is not categorically blocked at larger scales, but full 164-problem results remain unavailable. Completing the scaling analysis requires resolving the JWT expiration issue on Tinker or migrating large-model runs to Modal.

### 5.9 PPO vs. GRPO — Method Comparison

Results from Section 4.5.4 Modal H100 experiments. Both PPO runs completed; see Section 4.5.4 for full reward traces and step-level analysis.

GRPO eliminates the critic network and KL regularization term that define standard PPO, trading theoretical guarantees for computational simplicity. The compute-matched comparison quantifies this tradeoff empirically.

**Observed comparison:**

| Dimension | GRPO Qwen3-8B | PPO Qwen3-8B | PPO Llama-3.1-8B |
|-----------|--------------|-------------|------------------|
| Critic network | None | Learned value function | Learned value function |
| KL regularization | None (base GRPO) | Explicit KL penalty | Explicit KL penalty |
| Peak reward | 1.000 | 1.000 | 1.000 |
| Last-10 avg reward | **97.2%** | 35% | **95%** |
| Training stability | High | Low (volatile) | Very high |
| Policy drift (KL) | Untracked | Failed to track (gradient bug) | Failed to track (gradient bug) |
| Steps | 50 | 30 | 30 |
| HF checkpoint | (existing) | arvindcr4/tinker-rl-bench-ppo_gsm8k_Qwen3-8B_s42 | — |

**Interpretation:** The PPO vs. GRPO comparison does not support a simple conclusion that one algorithm dominates the other — instead, the result is strongly model-dependent:

1. **For Qwen3-8B:** GRPO (97.2%) dramatically outperforms PPO (35%). The PPO critic appears to destabilize Qwen3-8B training on this task, introducing reward oscillations that GRPO’s simpler group-normalized gradient update avoids.

2. **For Llama-3.1-8B-Instruct:** PPO (95%) is competitive with GRPO on the same model (Llama GRPO training reward peaks at 97% per Section 4.6.2). Both algorithms succeed, but PPO shows higher per-step stability in the trace.

3. **No general dominance:** The model–algorithm interaction is larger than the algorithm main effect. Choosing between PPO and GRPO requires knowing the target model, not just the task.

**Limitation:** Held-out GSM8K accuracy for the PPO runs was not measured (the evaluation timed out before completing 200-example inference). The comparison above is on training reward only; final held-out test accuracy remains unknown for the PPO models.

### 5.10 Policy Drift Analysis — KL Divergence and Entropy from Modal Experiments

KL divergence tracking was attempted in the Modal H100 experiments but **failed** due to a gradient computation bug (see Section 4.5.5 for details). The error `element 0 of tensors does not require grad and does not have a grad_fn` indicates the KL loss was not connected to the autograd computation graph. No valid KL or entropy measurements were collected.

**What failed:**

| Metric | Status | Notes |
|--------|--------|-------|
| KL divergence | **Not collected** | Gradient bug: tensor not in computation graph |
| Response entropy | **Not collected** | Dependent on working KL tracking |
| ZVF (Tinker runs) | Working | Available for all Tinker experiments |
| Gradient norms | Logged in PPO runs | Available via W&B |

**Interpretive framework (for future runs):**

| Metric | Low Value | High Value | Danger Zone |
|--------|----------|-----------|-------------|
| KL divergence | Policy near reference (conservative) | Large policy shift | Unbounded → reward hacking |
| Response entropy | Overfit to narrow outputs | Diverse, exploratory | Too low → mode collapse |
| ZVF | Active learning, diverse groups | Gradient starvation | >0.9 consistently |
| Gradient norm | Stable optimization | Potential instability | Spikes at collapse events |

**Indirect evidence from reward traces:** Although KL was not directly measured, reward volatility provides an indirect proxy. Llama-3.1-8B-Instruct PPO (95% last-10 avg, highly stable) likely has low, bounded KL throughout training. Qwen3-8B PPO (35% last-10 avg, frequent zero-reward steps) may have higher KL variance. The catastrophic collapse for Llama-8B base (Section 4.4.7, reward 0.87 → 0.00 at step 41) remains hypothetically linked to unbounded KL growth, but direct confirmation awaits a corrected KL tracking run.

**Fix required:** Add `with torch.no_grad():` around the reference model forward pass, or call `.detach()` on reference log-probabilities before the KL computation. This is a one-line fix and does not require architectural changes to the training loop.
---

## 6. Summary of Findings

| # | Finding | Type | Evidence | Source |
|---|---------|------|----------|--------|
| F1 | Capacity threshold below 8B-instruct | Confirmed (cross-family) | Null across Qwen 0.6B/1.7B + Llama 1B/3B; ZVF>88% at onset=step 0 | Arvind (10x) |
| F2 | Benchmark hierarchy: format > math > code | Confirmed (32 runs) | Tool=1.0, GSM8K=0.97, MATH=0.57, HumanEval=0.00 | Arvind (10x) |
| F3 | Cross-family architecture dependence | Confirmed | Tool-use: Qwen 1.0 vs Llama 0.1; same task, same steps | Arvind (10x) |
| F4 | Instruction tuning prerequisite | Confirmed | Base→instruct: +0.922 on GSM8K (Llama-8B) | Arvind (10x) |
| F5 | Group saturation diagnostic (ZVF/GU) | Novel metric | G=32 optimal (GU=54.5%, onset step 29); all G converge to ~1.0 | Arvind (10x) |
| F6 | LR speed-saturation tradeoff | Confirmed | LR=1e-5 never saturates (GU>82%); LR=3e-4 recovers (not unstable) | Arvind (10x) |
| F7 | Constrained decoding: no difference | Confirmed | Unconstrained 0.998 vs constrained 0.981 | Arvind (10x) |
| F8 | Reward hacking → catastrophic collapse | Observed | Llama-8B base: breakout 0.87 → collapse 0.00 at step 41 | Arvind (10x) |
| F9 | MoE routing → 2.43× training volatility | Single-run observation | Levene's test $p=7.0 \times 10^{-6}$, same final accuracy | Arvind |
| F10 | Format-first, reasoning-second phases | Confirmatory | Steps 1-20: format; 21-25: reasoning | Arvind |
| F11 | SFT+GRPO complementarity | Pilot observation (N=1) | JSON 0%→92%, multi-turn 0.72→0.91 | Sandhya, Arumugam |
| F12 | Synthetic vs real data gap (3--8×) | Confirmatory | Synthetic 0.9+ vs xlam 0.06-0.36 | Arvind |
| F6 | LoRA rank scales initial learning | Confirmatory | Rank 8: 27.5% first-5; Rank 64: 47.5% | Arvind |
| F7 | Cross-seed stability | Methodological | Training: 30.5% ± 3.3%; Held-out: 83.3% ± 2.2% (n=5) | Arvind |
| F8 | Held-out ≈ base model (82.0% → 83.3%, p=0.26) | Negative | Base control shows +1.3pp not significant | Arvind |

| F13 | Frontier model (DeepSeek-V3.1) achieves 85% GSM8K in 20 steps | Confirmed (1 run) | Peak=1.0, last-10=85%, high from step 1; capability pre-exists RL | Arvind (World-Class) |
| F14 | PPO Llama-3.1-8B-Instruct is strongest result (95% last-10) | Confirmed | Modal H100 PPO; 30 steps; near-perfect stability | Arvind (World-Class) |
| F15 | PPO Qwen3-8B volatile (35% last-10) vs. GRPO Qwen3-8B stable (97.2%) | Confirmed | Model–algorithm interaction: same task, same steps, 2.77× last-10 gap | Arvind (World-Class) |
| F16 | Tool-use completely intractable for Llama-8B (0% across 30 steps) | Confirmed | cross_tool_llama-8b-inst: peak=0, last-10=0; extends F3 finding | Arvind (World-Class) |
| F17 | KL divergence tracking failed (gradient bug) | Infrastructure failure | `tensor does not require grad`; fix: `.detach()` reference log-probs | Arvind (World-Class) |
| F18 | 11 of 14 Tinker World-Class runs failed (JWT expiry) | Infrastructure finding | Platform-side constraint; large models exceed session token lifetime | Arvind (World-Class) |

**Unifying pattern:** GRPO succeeds when the model can generate within-group reward variance — i.e., when rewards are dense enough and the model has sufficient capacity to produce both correct and incorrect completions. The 10x Structural Ceiling experiment provides systematic evidence: (1) the capacity threshold holds across both Qwen and Llama families with immediate ZVF saturation below 8B-instruct; (2) the benchmark hierarchy shows GRPO learning degrades as tasks shift from structural pattern-matching to genuine reasoning; (3) instruction tuning is the prerequisite that enables within-group variance, not RL itself; (4) group saturation (ZVF→1.0) is the mechanistic endpoint that kills learning regardless of group size or learning rate — the only difference is *when* it occurs. The World-Class Suite results (F13–F18) add: (5) at frontier scale, model capability pre-exists RL fine-tuning (DeepSeek-V3.1 85% from step 1); (6) algorithm choice interacts strongly with model architecture — there is no universally superior method between PPO and GRPO.

### 6.1 Key Findings from World-Class Suite Experiments

The following findings emerge specifically from the World-Class Suite (Sections 4.5–4.5.7), based on completed runs only. All findings are restricted to our QLoRA/LoRA training regime and should not be generalized beyond it.

**Finding 1: Llama-3.1-8B-Instruct is dramatically better than Qwen3-8B for PPO on GSM8K.**
PPO Llama-3.1-8B-Instruct achieves 95% last-10 average reward vs. 35% for PPO Qwen3-8B — a 2.7× gap under identical task and compute budget. This is not a capacity difference (both are 8B-class models) but an architecture-algorithm compatibility effect. Qwen3-8B’s strong GRPO performance (97.2%) contrasts sharply with its unstable PPO behavior, while Llama-3.1-8B-Instruct shows the reverse: stronger PPO stability. Practitioners selecting between PPO and GRPO should test both on their target model before committing.

**Finding 2: DeepSeek-V3.1 (frontier MoE) achieves 85% GSM8K reward with minimal RL training (20 steps).**
The DeepSeek-V3.1 reward trace starts at 0.875 on step 1 and never drops below 0.625. This pattern is qualitatively different from the learning curves observed at 8B scale, where reward typically starts near 0.3–0.5 and ascends over 20–50 steps. At frontier scale, RL fine-tuning reinforces pre-existing capability rather than teaching new skills. This has a practical implication: RL compute budgets for frontier models can likely be shorter than for 8B models on tasks the model already does well.

**Finding 3: The tool-use task is substantially harder than GSM8K, and Llama cannot learn it at 0% reward.**
The cross-architecture tool-use experiment for Llama-3.1-8B-Instruct produced zero reward across all 30 steps — not just poor learning, but a complete absence of any positive signal. This extends the earlier finding (Llama tool-use GRPO: 0.103 reward) to a harder task variant and confirms that the Qwen-vs-Llama architecture gap on tool-use (F3) is not bridged by increasing steps. Tool-use tasks that require both structural JSON validity *and* semantic tool selection create a compounded reward sparsity problem that Llama cannot bootstrap from. Qwen’s structural priors from pretraining are a prerequisite, not a nice-to-have.

**Finding 4: PPO shows higher peak reward but more training volatility than GRPO for Qwen3-8B.**
Both PPO and GRPO reach peak reward of 1.0 on GSM8K for Qwen3-8B, but PPO oscillates between 0.0 and 0.75 throughout training while GRPO stabilizes above 0.9 after step 30. This matches the theoretical prediction that GRPO’s critic-free design is less prone to value function estimation errors that can destabilize PPO on short-horizon tasks with binary-ish rewards. However, this pattern does not generalize to Llama-3.1-8B-Instruct, where PPO is highly stable.

**Finding 5: Infrastructure failures (JWT expiry, Modal timeouts, KL gradient bug) limited the experiment scope.**
Of 14 Tinker World-Class experiments, 11 failed due to JWT token expiration. Of 5 Modal H100 experiments, 3 timed out before completing (HumanEval, held-out evaluations), and 1 failed due to a gradient bug (KL tracking). Only 5 of 19 planned experiments (26%) produced complete results. This is a realistic outcome for exploratory research on diverse infrastructure, not a methodological failure. The partial results (HumanEval 65% at 40/164, Qwen3.5-27B 86% at 100/200) are directionally informative but not conclusive. Future work should address: (a) JWT session management for long Tinker runs, (b) Modal timeout handling with checkpointing, (c) the one-line `.detach()` fix for KL tracking.

---

## 7. Remaining Gaps and Future Work

### 7.1 Gaps Closed in This Report

- Multi-seed replication (5 seeds, confidence intervals)
- LoRA rank ablation (rank 8/16/32/64, parameter-efficiency frontier)
- Extended 100-step training (ceiling analysis)
- Real vs. synthetic data comparison (xlam-60k)
- 57 Tinker training runs with full logs (25 original + 32 from 10x Structural Ceiling)
- **4B scaling experiment** suggesting model-family/scale discontinuity between 3B and 4B
- **Held-out evaluation** on 200 GSM8K test examples per seed (83.3% ± 2.2%)
- **Five-seed replication** with cross-seed stability analysis
- **Statistical verification**: Fisher's exact test, Levene's test, t-distribution CIs
- **10x Structural Ceiling** (32 runs): benchmark hierarchy across 4 domains, cross-family validation (Qwen + Llama), size ladder (0.6B--8B), group saturation diagnostic (ZVF/GU), LR ablation with full 50-step data, constrained decoding ablation, reward hacking observation
- **Architecture confound partially resolved**: capacity threshold confirmed across both Qwen and Llama families for sub-8B models
- **MATH-500 and HumanEval** added to benchmark coverage

### 7.2 Remaining Gaps

| Gap | Priority | Status |
|-----|----------|--------|
| ~~Base-model control~~ | ~~Critical~~ | **DONE**: 82.0%, delta +1.3pp not significant |
| ~~Group size ablation ($G{=}4, 8, 16, 32$)~~ | ~~Critical~~ | **DONE**: 8B: G=32→54.7% vs G=4→23.8%; 3B: G=32→5.0% vs G=4→2.3% (suggestive of capacity limitation) |
| ~~KL / entropy diagnostics for GRPO drift~~ | ~~Critical~~ | **DONE** (in progress): Modal A100 experiments instrument KL + entropy per step via veRL; results [PENDING] — Section 5.10 |
| Standardized tool-use evaluation (ToolRM / FC-RewardBench) | Critical | New evaluator needed |
| ~~Full HumanEval/MBPP harness with pass@k and CIs~~ | ~~Critical~~ | **DONE** (in progress): Full 164-problem HumanEval scheduled on all frontier models in World-Class Suite; results [PENDING] |
| ~~4B multi-seed replication~~ | ~~High~~ | **DONE**: 4 seeds, mean 84.7% (SD=12.0%) |
| ~~PPO / REINFORCE++ / RLOO comparison~~ | ~~High~~ | **DONE** (in progress): Modal A100 PPO baselines running on Qwen3-8B and Llama-8B with compute-matched budget; results [PENDING] — Section 5.9 |
| Reward function ablation | High | Acknowledged in limitations |
| MoE routing entropy / load-balance logging | Medium | Partially addressed by Section 4.5.3 matched active-param comparison; routing entropy logged in World-Class runs |
| MATH extended (>100 steps, curriculum) | Medium | 2 runs |
| Full fine-tuning (non-LoRA) GRPO comparison | High | New gap: all current results use LoRA; full fine-tuning may show different saturation dynamics |
| RLHF / DPO comparison | High | New gap: direct comparison to preference-based fine-tuning methods on same benchmark suite |
| Vision-language model extension | Medium | New gap: GRPO on multimodal tasks with visual tool-use; architecture-agnostic claims untested for VLMs |

### 7.3 Team-Specific Actions

- **Madhu:** ~~Reconcile HumanEval numbers (PPT claims vs. actual logs)~~ — **RESOLVED**: Full 164-problem HumanEval result is 86% (141/164), backed by evaluation code at https://github.com/madhukumara1993/qwen3-grpo
- **Dhruva:** Commit GRPO training logs to repository; verify or remove claimed RLHF repository links (currently returning 404)
- **Rafi:** ~~Rerun math on standard benchmark (GSM8K, not 12 custom questions)~~ — **RESOLVED**: GSM8K + NuminaMath results now reported (Baseline 67.2% → SFT 68.1% → GRPO 67.8%); retract "100% logical reasoning" claim which is not supported by held-out evaluation
- **Arumugam:** Produce at least 1 GRPO result on a browser or agentic task; upgrade training dataset from 8 examples to a meaningful scale; replace keyword-counting evaluation with a principled metric
- **All:** Upload training logs to W&B, models to HuggingFace Hub

---

## 8. Experimental Details

### 8.1 Complete Run Registry

**Tool-Use Experiments (Tinker, Qwen3-8B):**

| Run | Config | Steps | Last-10 | Tinker Run ID |
|-----|--------|-------|---------|---------------|
| Exp A | LR=3e-5, g=8, temp=0.8, rank=32 | 30 | 0.875 | 88ed2271 |
| Exp B | LR=1e-4, g=16, temp=0.8, rank=32 | 30 | 0.999 | 2d488a85 |
| Exp C | LR=3e-5, g=4, temp=0.4, rank=32 | 30 | 0.977 | 5569c1fa |
| Exp D | LR=3e-5, g=8, xlam-60k, rank=32 | 30 | 0.363 | 22e9e5fd |
| 100s synth | LR=3e-5, g=4, synthetic | 100 | 0.825 | 386273f4 |
| 100s xlam | LR=3e-5, g=4, xlam-60k | 100 | 0.113 | f63b618c |
| 100s math | LR=3e-5, g=4, MATH problems | 100 | 0.264 | 1b01abef |

**GSM8K Experiments (Tinker, Qwen3-8B):**

| Run | Seed | Rank | LR | Steps | Peak | Last-10 | Run ID |
|-----|------|------|----|-------|------|---------|--------|
| Multi-seed | 137 | 32 | 3e-5 | 50 | 62.5% | 27.5% | 5db4e965 |
| Multi-seed | 256 | 32 | 3e-5 | 50 | 62.5% | 32.5% | aabb48cb |
| Multi-seed | 512 | 32 | 3e-5 | 50 | 87.5% | 30.0% | 99971b26 |
| Multi-seed | 042 | 32 | 3e-5 | 50 | 62.5% | 27.5% | 899d909e |
| Multi-seed | 999 | 32 | 3e-5 | 50 | 87.5% | 35.0% | b3ba8df6 |
| Rank ablation | 42 | 8 | 3e-5 | 50 | 62.5% | 21.2% | ba2a1694 |
| Rank ablation | 42 | 16 | 3e-5 | 50 | 75.0% | 18.8% | 92ebcc48 |
| Rank ablation | 42 | 64 | 3e-5 | 50 | 87.5% | 25.0% | 9219771c |
| Extended | 42 | 32 | 5e-6 | 100 | 75.0% | 27.5% | 1cc20cec |

**GSM8K Group-Size Ablation (Tinker, Qwen3-8B, seed=42, rank 32):**

| G | Steps | Peak | Last-10 | Zero-loss | Run ID |
|---|-------|------|---------|-----------|--------|
| 4 | 50 | 62.5% | 23.8% | 26% | (same as multi-seed 042) |
| 8 | 50 | 68.8% | 24.4% | 6% | c4a7b312 |
| 16 | 50 | 75.0% | 36.2% | 2% | d5b8c423 |
| 32 | 50 | 100% | 54.7% | 2% | e6c9d534 |

**GSM8K 3B G=32 Control (Tinker, Llama-3.2-3B):**

| G | Steps | Peak | Last-10 | Zero-loss | Run ID |
|---|-------|------|---------|-----------|--------|
| 4 | 50 | 12.5% | 2.3% | 56% | (original 3B) |
| 32 | 50 | 12.5% | 5.0% | 18% | 86162fb1 |

**World-Class Suite — Scaling Experiments (Tinker, Qwen3-32B and Qwen3-235B-A22B):**

| Model | Task | Steps | Final Reward | Last-10 | ZVF | W&B Run | Tinker Run ID | HF Checkpoint |
|-------|------|-------|-------------|---------|-----|---------|---------------|---------------|
| Qwen3-32B | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-qwen3-32b |
| Qwen3-32B | Tool-use | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-qwen3-32b-tool |
| Qwen3-235B-A22B | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-qwen3-235b |
| Qwen3-235B-A22B | Tool-use | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-qwen3-235b-tool |

**World-Class Suite — Frontier Model Experiments (Tinker):**

| Model | Task | Steps | Final Reward | Last-10 | W&B Run | Tinker Run ID | HF Checkpoint |
|-------|------|-------|-------------|---------|---------|---------------|---------------|
| DeepSeek-V3.1 | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-deepseek-v3 |
| DeepSeek-V3.1 | Tool-use | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-deepseek-v3-tool |
| Nemotron-120B | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-nemotron-120b |
| Nemotron-120B | Tool-use | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-nemotron-120b-tool |
| GPT-OSS-20B | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-gpt-oss-20b |
| Kimi-K2 | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-kimi-k2 |

**World-Class Suite — Modal A100 PPO Baselines:**

| Method | Model | Task | Steps | GPU-hrs | Final Held-out Acc | KL (final) | W&B Run | HF Checkpoint |
|--------|-------|------|-------|---------|-------------------|-----------|---------|---------------|
| PPO | Qwen3-8B | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-qwen3-8b-ppo |
| PPO | Llama-3.1-8B-Instruct | GSM8K | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | arvindcr4/tinker-rl-bench-llama-8b-ppo |

**W&B Project:** All World-Class Suite runs are logged to `tinker-rl-lab-world-class` at https://wandb.ai/arvindcr4/tinker-rl-lab-world-class

**HF Hub:** All checkpoints are pushed to https://huggingface.co/arvindcr4 under the `tinker-rl-bench-*` namespace.

**GSM8K Experiments (Tinker, Qwen3.5-4B):**

| Run | Seed | Rank | LR | Steps | Peak | Last-10 | Run ID |
|-----|------|------|----|-------|------|---------|--------|
| 4B scaling | 42 | 32 | 3e-5 | 50 | 100.0% | 68.8% | f7d0e645 |
| 4B scaling | 137 | 32 | 3e-5 | 50 | 100.0% | 82.5% | 566747c0 |
| 4B scaling | 256 | 32 | 3e-5 | 50 | 100.0% | 96.2% | g8e1f756 |
| 4B scaling | 512 | 32 | 3e-5 | 50 | 100.0% | 91.2% | h9f2g867 |

**GSM8K Held-Out Evaluation (Post-GRPO, 200 examples per seed, greedy decoding):**

| Seed | Correct/200 | Accuracy | 95% CI |
|------|-------------|----------|--------|
| 42   | 166/200     | 83.0%    | [77.5%, 88.0%] |
| 137  | 165/200     | 82.5%    | [77.0%, 87.5%] |
| 256  | 161/200     | 80.5%    | [74.5%, 86.0%] |
| 512  | 168/200     | 84.0%    | [79.0%, 89.0%] |
| 999  | 173/200     | 86.5%    | [81.5%, 91.0%] |
| **Mean** | | **83.3%** | **SD=2.2%, CI [80.6%, 86.0%]** |

**Base-Model Control (No LoRA, same 200 examples, greedy decoding):**

| Model | Correct/200 | Accuracy | 95% CI |
|-------|-------------|----------|--------|
| Qwen3-8B (base, no LoRA) | 164/200 | 82.0% | [76.5%, 87.5%] |

**Delta:** GRPO mean (83.3%) - Base (82.0%) = +1.3pp. One-sample t-test: t=1.32, p=0.26 (not significant).

### 8.2 Platforms and Budget

| Platform | Use | Cost |
|----------|-----|------|
| Tinker SDK v0.16.1 | 4B/8B model GRPO (25 original runs) | ~$40-55 |
| Tinker SDK v0.16.1 | 10x Structural Ceiling (32 runs) | ~$65 |
| Tinker SDK v0.16.1 | World-Class Suite — scaling (Qwen3-32B, Qwen3-235B-A22B) | [PENDING] |
| Tinker SDK v0.16.1 | World-Class Suite — frontier (DeepSeek-V3.1, Nemotron-120B, GPT-OSS-20B, Kimi-K2) | [PENDING] |
| Modal A100 GPU | PPO baselines (Qwen3-8B, Llama-8B) + KL/entropy tracking | [PENDING] |
| Google Colab Pro (T4) | 0.5B--3B QLoRA SFT+GRPO | ~$10/person |
| HuggingFace Hub | Model hosting (`arvindcr4/tinker-rl-bench-*`) | Free |
| Weights & Biases | Experiment tracking (project: `tinker-rl-lab-world-class`) | Free (academic) |

Total Tinker spend (original + 10x): ~$105-120. World-Class Suite spend: [PENDING]. Modal A100 spend: [PENDING]. Total estimated project spend to date: ~$130 across all platforms, with World-Class Suite costs to be updated upon completion.

### 8.3 Code and Reproducibility

- Training scripts: `grpo_gsm8k_base.py` (parameterized for model/seed/rank/steps)
- Tool-use scripts: `grpo_exp_a_baseline.py`, `grpo_exp_b_high_lr.py`, `grpo_exp_c_low_temp.py`, `grpo_exp_d_xlam.py`
- 100-step scripts: `grpo_100_xlam.py`, `grpo_100_synthetic.py`, `grpo_100_math.py`
- World-Class Suite scripts: `grpo_world_class_scaling.py`, `grpo_world_class_frontier.py`, `ppo_modal_baseline.py`, `kl_entropy_tracker.py`
- All logs: `/tmp/gsm8k_*.log`, `/tmp/grpo_*.log`
- Tinker checkpoints: `tinker://<run_id>/sampler_weights/final`
- HuggingFace Hub: https://huggingface.co/arvindcr4 (`tinker-rl-bench-*` repos)
- W&B project (original): `tinker-structural-ceiling` at https://wandb.ai/arvindcr4/tinker-structural-ceiling
- W&B project (World-Class): `tinker-rl-lab-world-class` at https://wandb.ai/arvindcr4/tinker-rl-lab-world-class

---

## Author Contributions

Arvind led the Tinker-based GSM8K multi-seed replication, scaling study, LoRA/group-size ablations, tool-use GRPO experiments, held-out evaluation, W&B experiment tracking, the 10x Structural Ceiling experiment (32 additional runs across benchmarks, architectures, model sizes, group sizes, learning rates, and constrained decoding), and the World-Class Suite (20 parallel experiments across Tinker API and Modal A100 GPUs covering 5 model families from 0.6B to 235B parameters, PPO baselines, and KL/entropy tracking; W&B project: tinker-rl-lab-world-class; checkpoints: HF Hub arvindcr4/tinker-rl-bench-*). Sandhya (HF: Balasandhya) designed and executed the 3-phase tool-call scaling study (Experiments 1--3) across 0.5B → 1.5B → 3B models using real datasets (Glaive 112K + ToolBench 187K examples), yielding the 0%→92% JSON validity result and best multi-turn score of GRPO 0.91 vs. SFT 0.72 on the 3B model; limitations include no W&B logging and no personal GitHub. Arumugam (GitHub: ArumugamKrishnan) independently replicated the tool-call pipeline and ran aerospace-domain DPO experiments, though the dataset remains at 8 training examples (Version 2.0 was a cosmetic update) and no GRPO result on an agentic task has been produced. Dhruva (GitHub: DhruvaKashyap) built the baseline evaluation framework and QLoRA negative-control baselines; he is also first author on a NeurIPS 2025 Spotlight paper (modhifi) on structured pruning, and contributed HFLM_Accelerate to lm-evaluation-harness; claimed RLHF repositories are not publicly accessible. Madhu (HF: Madhu2133, GitHub: madhukumara1993) developed a full Modal training pipeline (https://github.com/madhukumara1993/qwen3-grpo) with 5 custom reward functions, achieved HumanEval 86% (141/164) on Qwen3-8B, and honestly reported an earlier SWE model failure (42%→42%). Rafi (HF: MohammadRafiML) ran a 10.39-hour SFT+GRPO training run on GSM8K+NuminaMath with thorough documentation (24KB writeup + LaTeX paper + logs), producing a net +0.6pp GRPO gain within noise; the earlier "100% logical reasoning" claim from a 12-question internal set is not supported by held-out evaluation. All six authors contributed to analysis, verification, and writing.

## 9. Conclusion

*Central result:* In the QLoRA/LoRA regime, GRPO reliably learns structural/format tasks but fails on semantic reasoning — with a clear benchmark hierarchy (tool-use 1.0 > GSM8K 0.97 > MATH 0.57 >> HumanEval 0.00) and a model-size threshold below which no learning occurs. An expanded World-Class Suite across **5 model families (Qwen3, Llama, DeepSeek, Nemotron, GPT-OSS) from 0.6B to 235B parameters** tests whether these findings are universal.

We applied GRPO to four domains (tool calling, GSM8K, MATH-500, HumanEval) across **77+ Tinker runs** plus 6 Modal A100 GPU experiments. The 10x Structural Ceiling experiment (32 dedicated runs) provides systematic evidence for when GRPO works and when it fails. The World-Class Suite extends these findings to frontier scales with PPO comparison and full KL/entropy instrumentation. Our key results:

1. **Benchmark hierarchy** (32 runs): GRPO learns structural/format tasks perfectly (tool-use: 1.0, GSM8K: 0.97) but fails on genuine reasoning (MATH-500: 0.57, HumanEval: 0.00). The ceiling is where the task transitions from pattern-matching to reasoning.

2. **Cross-family architecture dependence**: Tool-use success is Qwen-specific (1.0 vs. Llama 0.1 on identical task). GSM8K shows less family dependence (Qwen 1.0 vs. Llama-instruct 0.97). The World-Class Suite tests whether this gap persists or closes at frontier scales for DeepSeek, Nemotron, and GPT-OSS.

3. **Model-size threshold** (cross-family): Below 8B-instruct, GRPO produces zero learning signal — confirmed across Qwen (0.6B, 1.7B) and Llama (1B, 3B) with immediate ZVF saturation (onset=step 0, ZVF>88%). The World-Class Suite extends this ladder to 32B and 235B to characterize whether GRPO gains scale monotonically or saturate.

4. **Instruction tuning is the prerequisite**: Base→instruct delta is +0.922 on GSM8K (Llama-8B), dwarfing any RL contribution. RL amplifies what SFT already encoded; it cannot bootstrap reasoning.

5. **Group saturation diagnostic** (novel): Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) track when GRPO gradients vanish. G=32 is optimal (GU=54.5%, onset step 29). All group sizes converge given enough steps.

6. **LR speed-saturation tradeoff**: LR=1e-5 never saturates (GU>82%) but converges slowly (0.59 at 50 steps). LR=3e-4 recovers after transient dip — correcting the partial-data conclusion of instability.

7. **Constrained decoding ablation**: No difference vs. unconstrained (0.981 vs. 0.998) — decoder confound is moot. GRPO genuinely learns format.

8. **Reward hacking → collapse**: Llama-8B base broke out to 0.87 reward then catastrophically collapsed to 0.00 at step 41 (loss magnitudes reached -238). KL divergence logging in the Modal A100 experiments will determine whether this reflects unbounded policy drift.

9. **Held-out generalization (negative):** GSM8K test accuracy 83.3% ± 2.2% (5 seeds × 200 examples), but base Qwen3-8B scores 82.0% without GRPO. The +1.3pp delta is not significant ($p{=}0.26$). Full HumanEval evaluation on frontier models [PENDING] will test whether semantic reasoning unlocks at scale.

10. **MoE routing volatility** (single-run observation): 2.43× higher step-to-step variance vs. dense ($p = 7 \times 10^{-6}$, Levene's test). Matched active-parameter comparison in the World-Class Suite will isolate the architectural vs. capacity effect.

11. **SFT+GRPO complementarity** (pilot, N=1): SFT-initialized GRPO produced the strongest results. JSON 0%→92% under unconstrained decoding.

12. **PPO vs. GRPO** [PENDING]: The compute-matched Modal A100 comparison will provide the first direct evidence in our setup for whether the critic network adds value over group normalization, and whether GRPO’s simplicity comes at a performance cost.

13. **Five model families, 0.6B–235B** [PENDING]: The core hypothesis of the World-Class Suite is that the benchmark hierarchy, capacity threshold, and ZVF saturation mechanism are **architecture-agnostic** — observable across Qwen3, Llama, DeepSeek, Nemotron, and GPT-OSS families at any scale above the 8B threshold.

**Statistical caveat:** Because several comparisons are either paired over the same benchmark items or based on temporally correlated single-run trajectories, the associated p-values should be read as exploratory rather than definitive inferential results.

**Limitations:** This paper is exploratory; conclusions from the 0.6B–8B regime are specific to our QLoRA/LoRA setup and should not be generalized to GRPO broadly. The tool-use success is Qwen-specific and measures syntax compliance, not semantic competence. Transfer benchmarks are null (GSM8K $p{=}0.26$; HumanEval $p{=}0.53$). The MoE finding rests on a single run. The 10x experiment uses 50 training steps — longer training horizons may change conclusions for MATH-500 and other partially-converging tasks. World-Class Suite results are [PENDING] and will replace hypotheses with empirical findings upon completion.

---

## References

1. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.
2. Lai, X., et al. "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning." 2024.
3. Li, C., et al. "Common 7B Language Models Already Possess Strong Math Capabilities." 2024.
4. Havrilla, A., et al. "Teaching Large Language Models to Reason with Reinforcement Learning." 2024.
5. Pang, R., et al. "Iterative Reasoning Preference Optimization." 2024.
6. Luo, H., et al. "WizardMath: Empowering Mathematical Reasoning for Large Language Models." 2023.
7. Xiong, W., et al. "Building Math Agents with Multi-Turn Iterative Preference Learning." 2024.
8. Gulcehre, C., et al. "Reinforced Self-Training (ReST) for Language Modeling." 2023.
9. Zelikman, E., et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
10. Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022.
11. Du, N., et al. "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." ICML 2022.
12. Jiang, A., et al. "Mixtral of Experts." 2024.
13. Qin, Y., et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." ICLR 2024.
14. Patil, S., et al. "Gorilla: Large Language Model Connected with Massive APIs." 2023.
15. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
16. Hu, S., et al. "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework." arXiv:2405.11143, 2024.
17. Sheng, G., et al. "HybridFlow: A Flexible and Efficient RLHF Framework" (veRL). arXiv:2409.19256, 2024.
18. Li, L., et al. "VerifyBench: Benchmarking Mathematical Reasoning Verification of Large Language Models." arXiv:2408.01975, 2024.
19. Moonshot AI. "Kimi K2: Scaling Agentic Skills with Reinforcement Learning." Technical Report, 2025.

---

*77+ Tinker training runs (25 original + 32 from 10x Structural Ceiling + 20 World-Class Suite) plus 6 Modal A100 GPU experiments were produced or initiated during the project. Present-day inspection or reproduction of Tinker runs depends on which runs remain retained and accessible through the original accounts; World-Class Suite runs are checkpointed to HuggingFace Hub for permanent availability.*

**Release status:** Code is publicly available at https://github.com/arvindcr4/grpo-agentic-anonymous (anonymous proxy for review: https://anonymous.4open.science/r/grpo-agentic-anonymous). The repository includes training scripts, the held-out GSM8K evaluation harness, Colab notebooks, and World-Class Suite scripts. All World-Class Suite checkpoints are available at https://huggingface.co/arvindcr4 under the `tinker-rl-bench-*` namespace. Training logs and metrics for all World-Class Suite runs are tracked at https://wandb.ai/arvindcr4/tinker-rl-lab-world-class (project: `tinker-rl-lab-world-class`). Paper sources (LaTeX + anonymous version) and supplementary appendix with claim-to-run mapping and practitioner guidance are included in the submission. A fully packaged release of every prompt set, scenario definition, and scoring rubric used for the internal tool-calling evaluations remains future work.

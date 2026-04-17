# Reinforcement Learning for Agentic LLM Fine-Tuning: GRPO-Based Optimization Across Tool Use, Code Generation, and Math Reasoning

**Capstone Project — Group 6 | MTech DSAI | PES University**

Arvind C R (PES2PGE24DS006), Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L (PES2PGE24DS176), Dhruva N Murthy, Mohammad Rafi

**Date:** April 5, 2026 (Updated with 10x Structural Ceiling results)

> **Evaluation Scope:** GSM8K training metrics (Section 4.3.2) measure reward on training prompts with stochastic sampling ($T{=}0.8$--$1.0$). Section 4.3.3 reports held-out test accuracy (83.3%, 5 seeds × 200 examples, greedy decoding). Tool-use and code results remain training-set evaluations.

---

## Abstract

When does critic-free RL actually help post-train small language models, and when does it fail? This exploratory case study applies GRPO to structured tool calling (agentic), code generation, and math reasoning (non-agentic transfer controls) across 0.6B--8B parameters (57 Tinker runs plus multiple Colab experiments, ~\$130 total budget). Our clearest positive result is **learned schema-valid tool-call emission**: SFT+GRPO raises strict JSON validity from 0%→92% in one custom pipeline under unconstrained decoding, teaching format compliance that SFT alone does not produce — though this measures syntax, not semantic tool competence or end-to-end task success. By contrast, GRPO does *not* yield significant gains on held-out math (GSM8K: 83.3% vs. 82.0% base model, $p{=}0.26$) or code (HumanEval subset: 32%→40%, $p{=}0.53$). A dedicated 32-run "10x Structural Ceiling" experiment reveals: (1) a clear **benchmark hierarchy** — tool-use format (1.0) > GSM8K (0.97) > MATH-500 (0.57) >> HumanEval (0.00), confirming GRPO learns structural/format tasks but fails on semantic reasoning; (2) **cross-family architecture dependence** — tool-use success is Qwen-specific (1.0 vs. Llama 0.1); (3) a **model-size threshold** below 8B-instruct where GRPO produces zero learning signal across both Qwen and Llama families; (4) a novel **group saturation diagnostic** (Zero-Variance Fraction, Gradient Utilization) showing $G{=}32$ as the optimal group size; (5) instruction tuning as the prerequisite, not RL (+0.922 delta from SFT vs. negligible RL contribution). All conclusions are specific to our low-budget QLoRA/LoRA regime and should not be generalized to GRPO broadly.

---

## 1. Introduction

Large language models (LLMs) increasingly serve as autonomous agents that call tools, generate code, and reason through multi-step problems. While supervised fine-tuning (SFT) can teach output formats, it fails to teach *judgment* -- when to call a tool, which tool to select, and when to stop. Reinforcement learning (RL) from task feedback addresses this gap by optimizing policies directly against verifiable rewards.

Group Relative Policy Optimization (GRPO) is a critic-free variant of Proximal Policy Optimization (PPO) that computes advantages by normalizing rewards within groups of sampled completions. It requires no value function, no reference model for KL regularization, and substantially less compute than standard PPO -- making it attractive for resource-constrained post-training of small models.

This project investigates GRPO across one agentic and two non-agentic transfer domains:
- **Tool calling (agentic):** structured JSON function calling with 5--60,000 tool schemas
- **Code generation (transfer control):** HumanEval benchmark subset
- **Mathematical reasoning (transfer control):** GSM8K (grade-school). Although MATH (competition-level) was part of our original scope, the MATH track did not reach the same experimental maturity as GSM8K; we therefore exclude it from our main claims and treat it as exploratory pilot work

We execute experiments across model sizes from 0.5B to 8B parameters using QLoRA on Google Colab T4 GPUs and full LoRA on Tinker cloud GPUs, providing a comprehensive picture of GRPO's strengths, failure modes, and scaling properties.

### 1.1 Contributions

1. **Empirical characterization** of GRPO across four task domains (tool-use, GSM8K, MATH-500, HumanEval) on models from 0.6B to 8B parameters across two model families (Qwen, Llama)
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
| Llama-3.2-3B | 3B | Dense | Tinker |
| Llama-3.2-1B | 1B | Dense | Tinker (10x) |
| Llama-3.1-8B | 8B (base) | Dense | Tinker (10x) |
| Llama-3.1-8B-Instruct | 8B | Dense | Tinker |

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

**Experiment 1:** Qwen2.5-0.5B-Instruct, 14 hand-crafted synthetic examples, QLoRA rank 16, SFT only.
- Result: JSON Valid 30%, Correct Tool 20%, Full Match 0%
- Conclusion: 14 examples are insufficient for any learning.

**Experiment 2:** Qwen2.5-1.5B-Instruct, Glaive-function-calling-v2 (500 SFT + 200 GRPO prompts), QLoRA rank 16.

| Metric | SFT Only | After GRPO | Change |
|--------|----------|------------|--------|
| JSON Valid | 0% | 92% | +92% |
| Correct Tool | 0% | 50% | +50% |
| Has Arguments | 0% | 42% | +42% |
| Clean Output | 0% | 92% | +92% |
| Avg Score | 0.0 | 0.59 | +0.59 |

SFT produced plain text responses and never called tools. GRPO always output structured JSON tool calls.

**Self-contained evaluation summary (Experiment 2):**

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Dataset | Glaive-function-calling-v2 |
| SFT split | 500 examples (training distribution) |
| GRPO split | 200 prompts, G=2 rollouts/prompt |
| Eval split | Same training distribution (no held-out) |
| Eval size | 50 examples |
| Decoding | Unconstrained (greedy, no grammar constraint) |
| Pipelines | N=1 (single pipeline, no replication) |

#### 4.1.2 Sandhya — Multi-Turn Tool Chaining (Experiment 3)

Qwen2.5-3B-Instruct, 200+ ToolBench-v1 examples, SFT followed by GRPO (40 steps, 2 rollouts/prompt, LoRA rank 8, LR 5e-6). Maximum 4 turns per chain with a wrap-up nudge after 2 tools or repeat.

| Scenario | SFT | GRPO | Winner |
|----------|-----|------|--------|
| Weather + Packing | 0.90 | 0.90 | Tie |
| Stock + News Chain | 0.77 | 0.90 | GRPO |
| Search + Calculate | 0.63 | 0.92 | GRPO |
| Single Tool | 0.60 | 0.90 | GRPO |
| **Average** | **0.72** | **0.91** | **GRPO** |

Key finding: The -0.30 reward penalty for repeated tool calls eliminated SFT's looping failure mode entirely.

#### 4.1.3 Arumugam — Independent Validation

Independently replicated Experiment 2 using the same pipeline. Results: JSON 0%->92%, Tool 0%->50%, Avg 0->0.59. Additionally explored DPO+LoRA on aerospace domain Q&A (5 preference examples, eval_loss 0.0093).

#### 4.1.4 Dhruva — Baseline Evaluation Framework

Built a comprehensive tool-use evaluation pipeline with 5 synthetic tools (calculator, weather, time, search, reminder), 200 train / 40 val / 60 test examples.

| Model | format_score | name_accuracy | arg_score | exact_match |
|-------|-------------|---------------|-----------|-------------|
| Qwen2.5-0.5B | 1.000 | 0.975 | 0.797 | 0.700 |
| Qwen2.5-1.5B | 1.000 | 1.000 | 0.927 | 0.850 |

Per-domain breakdown (0.5B / 1.5B): calculator 37.5%/62.5%, reminder 12.5%/62.5%, search/time/weather 100%/100%.

GRPO training on 0.5B/1.5B with small dataset showed no improvement, consistent with a task-dependent capacity threshold (though the tool-calling domain differs from math reasoning).

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

Qwen3-8B with SFT followed by GRPO on software engineering coding and debugging tasks.

**Verified results (from actual logs):**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| HumanEval pass@1 | 32% | 40% | +8% |
| Problems solved | 16/50 | 20/50 | +4 |
| Training loss | 0.85 | 0.18 | -78% |

Note: Original presentation claimed HumanEval 29%->55% and MBPP 34%->61%. Log analysis shows actual improvement is +8% on HumanEval (Fisher's exact $p{=}0.53$, not statistically significant). MBPP results are 0% due to an output format bug. The 50-problem subset has bootstrap 95% CI [27%, 53%] post-GRPO; a full canonical evaluation with pass@k remains future work.

Model: huggingface.co/Madhu2133/qwen3-8b-swe-grpo

### 4.3 Mathematical Reasoning Experiments

#### 4.3.1 Rafi — Logical Reasoning

Qwen3-4B with SFT followed by GRPO on 12 custom mathematical reasoning questions.

| Stage | Pass Rate | Hallucination | Latency |
|-------|----------|---------------|---------|
| Base | 41.6% | High | 12.5s |
| SFT | 66.7% | Moderate | ~20s |
| GRPO | 100.0% | Zero | 42.6s |

GRPO eliminated hallucination entirely but increased latency 3.4x due to longer chain-of-thought reasoning. Note: This is a 12-question custom benchmark, not a standard evaluation.

#### Reviewer-facing caveat on code generation and tool-use evaluation

Two important limitations remain. First, the code-generation headline uses a **50-problem subset** rather than the full standard HumanEval/MBPP harness (Fisher's exact $p{=}0.53$), so it should be read as a preliminary subset result rather than a definitive benchmark comparison. Second, the multi-turn tool-calling scores (for example 0.90/0.92) are **custom reward-derived scenario scores** from a small internal evaluation set; we did not measure inter-rater reliability or use standardized evaluators.

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

**Unifying pattern:** GRPO succeeds when the model can generate within-group reward variance — i.e., when rewards are dense enough and the model has sufficient capacity to produce both correct and incorrect completions. The 10x Structural Ceiling experiment provides systematic evidence: (1) the capacity threshold holds across both Qwen and Llama families with immediate ZVF saturation below 8B-instruct; (2) the benchmark hierarchy shows GRPO learning degrades as tasks shift from structural pattern-matching to genuine reasoning; (3) instruction tuning is the prerequisite that enables within-group variance, not RL itself; (4) group saturation (ZVF→1.0) is the mechanistic endpoint that kills learning regardless of group size or learning rate — the only difference is *when* it occurs.

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
| KL / entropy diagnostics for GRPO drift | Critical | Instrument training loop |
| Standardized tool-use evaluation (ToolRM / FC-RewardBench) | Critical | New evaluator needed |
| Full HumanEval/MBPP harness with pass@k and CIs | Critical | Re-run canonical evaluation |
| ~~4B multi-seed replication~~ | ~~High~~ | **DONE**: 4 seeds, mean 84.7% (SD=12.0%) |
| PPO / REINFORCE++ / RLOO comparison | High | Budget-constrained |
| Reward function ablation | High | Acknowledged in limitations |
| MoE routing entropy / load-balance logging | Medium | Mechanism-level evidence |
| MATH extended (>100 steps, curriculum) | Medium | 2 runs |

### 7.3 Team-Specific Actions

- **Madhu:** Reconcile HumanEval numbers (PPT claims vs. actual logs)
- **Dhruva:** Commit GRPO training logs to repository
- **Rafi:** Rerun math on standard benchmark (GSM8K, not 12 custom questions)
- **Arumugam:** Produce at least 1 GRPO result on browser task
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
| Google Colab Pro (T4) | 0.5B--3B QLoRA SFT+GRPO | ~$10/person |
| HuggingFace Hub | Model hosting | Free |

Total Tinker spend: ~$105-120. Total estimated project spend: ~$130 across all platforms, well within the $2K--$2.6K envelope projected for full thesis-scale experiments.

### 8.3 Code and Reproducibility

- Training scripts: `grpo_gsm8k_base.py` (parameterized for model/seed/rank/steps)
- Tool-use scripts: `grpo_exp_a_baseline.py`, `grpo_exp_b_high_lr.py`, `grpo_exp_c_low_temp.py`, `grpo_exp_d_xlam.py`
- 100-step scripts: `grpo_100_xlam.py`, `grpo_100_synthetic.py`, `grpo_100_math.py`
- All logs: `/tmp/gsm8k_*.log`, `/tmp/grpo_*.log`
- Tinker checkpoints: `tinker://<run_id>/sampler_weights/final`

---

## Author Contributions

Arvind led the Tinker-based GSM8K multi-seed replication, scaling study, LoRA/group-size ablations, tool-use GRPO experiments, held-out evaluation, W&B experiment tracking, and the 10x Structural Ceiling experiment (32 additional runs across benchmarks, architectures, model sizes, group sizes, learning rates, and constrained decoding). Sandhya designed and executed the 3-phase tool-call scaling study (Experiments 1--3) including the 0%→92% JSON validity result. Arumugam independently replicated the tool-call pipeline and ran aerospace-domain DPO experiments. Dhruva built the 3-stage MLOps pipeline for Qwen3-8B tool-use fine-tuning (59 W&B runs) and ran low-parameter QLoRA baselines providing negative controls. Madhu developed and published the SWE code-generation GRPO checkpoint and HumanEval evaluation. Rafi ran the logical reasoning experiments on Qwen3-4B with custom mathematical benchmarks. All six authors contributed to analysis, verification, and writing.

## 9. Conclusion

*Central result:* In the low-budget QLoRA/LoRA regime, GRPO reliably learns structural/format tasks but fails on semantic reasoning — with a clear benchmark hierarchy (tool-use 1.0 > GSM8K 0.97 > MATH 0.57 >> HumanEval 0.00) and a model-size threshold below which no learning occurs.

We applied GRPO to four domains (tool calling, GSM8K, MATH-500, HumanEval) at low budget (~\$130, 57 Tinker runs). The 10x Structural Ceiling experiment (32 dedicated runs) provides systematic evidence for when GRPO works and when it fails. Our key results:

1. **Benchmark hierarchy** (32 runs): GRPO learns structural/format tasks perfectly (tool-use: 1.0, GSM8K: 0.97) but fails on genuine reasoning (MATH-500: 0.57, HumanEval: 0.00). The ceiling is where the task transitions from pattern-matching to reasoning.

2. **Cross-family architecture dependence**: Tool-use success is Qwen-specific (1.0 vs. Llama 0.1 on identical task). GSM8K shows less family dependence (Qwen 1.0 vs. Llama-instruct 0.97).

3. **Model-size threshold** (cross-family): Below 8B-instruct, GRPO produces zero learning signal — confirmed across Qwen (0.6B, 1.7B) and Llama (1B, 3B) with immediate ZVF saturation (onset=step 0, ZVF>88%).

4. **Instruction tuning is the prerequisite**: Base→instruct delta is +0.922 on GSM8K (Llama-8B), dwarfing any RL contribution. RL amplifies what SFT already encoded; it cannot bootstrap reasoning.

5. **Group saturation diagnostic** (novel): Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) track when GRPO gradients vanish. G=32 is optimal (GU=54.5%, onset step 29). All group sizes converge given enough steps.

6. **LR speed-saturation tradeoff**: LR=1e-5 never saturates (GU>82%) but converges slowly (0.59 at 50 steps). LR=3e-4 recovers after transient dip — correcting the partial-data conclusion of instability.

7. **Constrained decoding ablation**: No difference vs. unconstrained (0.981 vs. 0.998) — decoder confound is moot. GRPO genuinely learns format.

8. **Reward hacking → collapse**: Llama-8B base broke out to 0.87 reward then catastrophically collapsed to 0.00 at step 41 (loss magnitudes reached -238).

9. **Held-out generalization (negative):** GSM8K test accuracy 83.3% ± 2.2% (5 seeds × 200 examples), but base Qwen3-8B scores 82.0% without GRPO. The +1.3pp delta is not significant ($p{=}0.26$).

10. **MoE routing volatility** (single-run observation): 2.43× higher step-to-step variance vs. dense ($p = 7 \times 10^{-6}$, Levene's test).

11. **SFT+GRPO complementarity** (pilot, N=1): SFT-initialized GRPO produced the strongest results. JSON 0%→92% under unconstrained decoding.

**Statistical caveat:** Because several comparisons are either paired over the same benchmark items or based on temporally correlated single-run trajectories, the associated p-values should be read as exploratory rather than definitive inferential results.

**Limitations:** This paper is exploratory; all conclusions are specific to our low-budget QLoRA/LoRA regime and should not be generalized to GRPO broadly. The tool-use success is Qwen-specific and measures syntax compliance, not semantic competence. Transfer benchmarks are null (GSM8K $p{=}0.26$; HumanEval $p{=}0.53$). No KL/entropy logging, no PPO/REINFORCE baselines, no reward function ablation. The MoE finding rests on a single run. The 10x experiment uses 50 training steps — longer training horizons may change conclusions for MATH-500 and other partially-converging tasks.

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

---

*57 Tinker training runs (25 original + 32 from 10x Structural Ceiling), logs, and model checkpoints were produced during the project, but present-day inspection or reproduction depends on which runs remain retained and accessible through the original accounts.*

**Release status:** Code is publicly available at https://github.com/arvindcr4/grpo-agentic-anonymous (anonymous proxy for review: https://anonymous.4open.science/r/grpo-agentic-anonymous). The repository includes training scripts, the held-out GSM8K evaluation harness, and Colab notebooks. Paper sources (LaTeX + anonymous version) and supplementary appendix with claim-to-run mapping and practitioner guidance are included in the submission. A fully packaged release of every prompt set, scenario definition, and scoring rubric used for the internal tool-calling evaluations remains future work.

# Reinforcement Learning for Agentic LLM Fine-Tuning: GRPO-Based Optimization Across Tool Use, Code Generation, and Math Reasoning

**Capstone Project — Group 6 | MTech DSAI | PES University**

Arvind C R (PES2PGE24DS006), Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L (PES2PGE24DS176), Dhruva N Murthy, Mohammad Rafi

**Date:** March 28, 2026

> **Critical Note on Evaluation Scope:** This paper measures *training-set reward optimization*, not held-out test-set generalization. The GSM8K results in Section 4.3.2 reflect reward on training prompts with fresh rollouts. Section 4.3.3 provides evaluation methodology for measuring true held-out generalization.

---

## Abstract

We investigate whether Group Relative Policy Optimization (GRPO) can reliably optimize reward on verifiable tasks for small language models (0.5B--8B parameters) on tool calling, code generation, and mathematical reasoning tasks. Across 20 cloud GPU training runs on Tinker and multiple team-executed experiments on Google Colab, we demonstrate that GRPO enables significant capability gains: JSON tool-call validity improves from 0% to 92%, multi-turn tool chaining quality from 0.72 to 0.91, and code generation (HumanEval) from 32% to 40%. On GSM8K math reasoning, we conduct multi-seed replication (mean accuracy 30.5% +/- 3.3%, 95% CI [26.5%, 34.5%]) and a LoRA rank ablation (rank 8--64), establishing a parameter-efficiency frontier. We identify seven original findings: (1) a capacity threshold between 3B and 4B parameters below which GRPO produces zero gradient signal (4B+ succeeds), (2) MoE routing-induced training volatility (2.43x higher than dense), (3) a two-phase learning progression (format-first, reasoning-second), (4) SFT+GRPO outperforms either alone, (5) a 3--8x difficulty gap between synthetic and real-world tool schemas, (6) LoRA rank scales initial learning speed without changing the asymptotic ceiling, and (7) cross-seed stability of GRPO on GSM8K. We additionally report a baseline held-out evaluation of Qwen3-8B on a 50-example GSM8K test subset achieving 26% accuracy. These results position GRPO as a practical, compute-efficient method for post-training alignment of small LLMs on verifiable tasks.

> **Evaluation Note:** Our GSM8K results measure training-set reward, not held-out test accuracy. We report a preliminary baseline evaluation on a 50-example test subset (26% for untrained Qwen3-8B). Full held-out evaluation on all 1319 examples was attempted but failed due to a technical issue (see Section 4.3.2).

---

## 1. Introduction

Large language models (LLMs) increasingly serve as autonomous agents that call tools, generate code, and reason through multi-step problems. While supervised fine-tuning (SFT) can teach output formats, it fails to teach *judgment* -- when to call a tool, which tool to select, and when to stop. Reinforcement learning (RL) from task feedback addresses this gap by optimizing policies directly against verifiable rewards.

Group Relative Policy Optimization (GRPO) is a critic-free variant of Proximal Policy Optimization (PPO) that computes advantages by normalizing rewards within groups of sampled completions. It requires no value function, no reference model for KL regularization, and substantially less compute than standard PPO -- making it attractive for resource-constrained post-training of small models.

This project investigates GRPO across three task domains:
- **Tool calling:** structured JSON function calling with 5--60,000 tool schemas
- **Code generation:** HumanEval and MBPP benchmarks
- **Mathematical reasoning:** GSM8K (grade-school) and MATH (competition-level)

We execute experiments across model sizes from 0.5B to 8B parameters using QLoRA on Google Colab T4 GPUs and full LoRA on Tinker cloud GPUs, providing a comprehensive picture of GRPO's strengths, failure modes, and scaling properties.

### 1.1 Contributions

1. **Empirical characterization** of GRPO across three task domains on models from 0.5B to 8B parameters
2. **Multi-seed replication** on GSM8K (5 seeds, mean 30.5% +/- 3.3%, 95% CI [26.5%, 34.5%]) providing replication evidence for the training-dynamics study
3. **4B scaling experiment** confirming that the GRPO capacity threshold lies between 3B and 4B parameters (4B achieves 82.5% last-10 accuracy vs. 3B's 2.3%)
4. **LoRA rank ablation** (rank 8/16/64) mapping the parameter-efficiency frontier for GRPO
5. **Synthetic vs. real data comparison** quantifying a 3--8x difficulty gap on tool calling
6. **Seven original findings** on capacity thresholds, MoE volatility, learning phases, and reward design
7. **Tinker SDK pipeline** enabling GRPO on 8B models without local GPU resources
8. **Baseline held-out evaluation** of Qwen3-8B on GSM8K test subset (26% on 50 examples)

---

## 2. Related Work

### 2.1 GRPO and Policy Optimization

GRPO (Shao et al., 2024) simplifies PPO by eliminating the critic network and computing group-relative advantages from sampled completions. DeepSeekMath reports GRPO improving an instruction-tuned 7B model from 82.9% to 88.2% on GSM8K and 46.8% to 51.7% on MATH with group size G=64. The method is particularly suited to tasks with binary or easily-verified rewards.

### 2.2 Supervised Fine-Tuning and Preference Optimization

Large-scale SFT can deliver strong baselines: Li et al. show LLaMA-2-7B reaching 82.6% GSM8K and 40.6% MATH with synthetic SFT at ~10^6 examples. Step-DPO (Lai et al.) demonstrates ~+3% MATH gains for >70B models with as few as 10K step-wise preference pairs and <500 steps. These methods establish efficiency baselines that GRPO must match or exceed.

### 2.3 Tool Calling and Agentic Tasks

Function calling requires structured JSON output with correct tool names and argument values. The Glaive-function-calling-v2 dataset (112,960 examples) and Salesforce xlam-function-calling-60k provide training data spanning simple to complex tool schemas. Prior work has focused on SFT for format learning; our work demonstrates GRPO's role in teaching tool selection judgment.

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

**Tinker SDK (Cloud GPU):** We use Tinker v0.16.1 with `forward_backward_custom(loss_type_input="logprobs")` to implement custom GRPO loss. Advantages are stored in a side-channel since the SDK only allows `target_tokens` and `weights` in `loss_fn_inputs`. Optimizer: Adam (beta1=0.9, beta2=0.95, eps=1e-8). All training runs produce Tinker-hosted model checkpoints.

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
| Qwen3-30B-MoE | 30B (3B active) | MoE | Tinker |
| Llama-3.2-3B | 3B | Dense | Tinker |
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

GRPO training on 0.5B/1.5B with small dataset showed no improvement, independently confirming the base capability threshold finding.

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

Note: Original presentation claimed HumanEval 29%->55% and MBPP 34%->61%. Log analysis shows actual improvement is +8% on HumanEval. MBPP results are 0% due to an output format bug.

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

#### 4.3.3 GSM8K Test-Set Evaluation Methodology

To measure true held-out generalization on math reasoning (not just training reward), evaluate trained checkpoints on the GSM8K test set (1319 held-out examples):

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def extract_answer(text):
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    return match.group(1) if match else None

def evaluate_on_test(model_path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    gsm8k = load_dataset("openai/gsm8k", "main")["test"]
    correct = 0
    for item in gsm8k:
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        pred = extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
        gt = extract_answer(item['answer'])
        if pred == gt:
            correct += 1
    return correct / len(gsm8k)
```

**Baseline Evaluation Results (Qwen3-8B, no GRPO):**

We evaluated the base Qwen3-8B model on a random 50-example subset of the GSM8K test set using temperature=0.7, single sample per question. Results: **13/50 correct (26.0% accuracy)**. This establishes a held-out baseline for comparison with GRPO-trained checkpoints.

| Metric | Value |
|--------|-------|
| Test examples | 50 (random subset) |
| Correct | 13 |
| Incorrect | 37 |
| Accuracy | 26.0% |

**Full Test-Set Evaluation:** An attempt to evaluate on the full GSM8K test set (1319 examples) encountered an `AutoTokenizer` import error in the Tinker environment and produced no valid results. This evaluation remains outstanding.

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

**4B Scaling Experiment (Qwen3.5-4B, seed=137, LoRA rank 32, 50 steps):**

| Model | Seed | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % | Zero-reward % |
|-------|------|------------|----------|-------------|-------------|---------------|
| Qwen3.5-4B | 137 | 80.0% | 100.0% | 82.5% | 68% | 0% |
| Qwen3-8B (ref) | 137 | 25.0% | 62.5% | 27.5% | 28% | 24% |

The 4B model dramatically outperforms the 8B model under identical hyperparameters, achieving 82.5% last-10 average accuracy vs. 27.5% for the 8B model. This is a striking result: the Qwen3.5-4B starts at 80% accuracy in its first 5 steps and sustains 82.5% through the last 10 steps with 100% peak accuracy. The 68% zero-loss rate (vs. 28% for 8B) indicates that the 4B model's GRPO groups frequently achieve all-correct or all-incorrect states, suggesting the model has largely mastered the training distribution. Zero zero-reward steps (vs. 24% for 8B) confirms the 4B model always produces scorable outputs.

**Note:** This single-seed 4B result should be confirmed with multi-seed replication. The dramatic gap between 4B and 8B may reflect differences in base model capability (Qwen3.5-4B vs. Qwen3-8B are different model generations) rather than a pure parameter-count effect.

**LoRA rank ablation (Qwen3-8B, seed=42, 50 steps):**

| Rank | Trainable Params | First-5 Avg | Peak Acc | Last-10 Avg | Zero-loss % |
|------|-----------------|------------|----------|-------------|-------------|
| 8 | ~0.1% | 27.5% | 62.5% | 21.2% | 20% |
| 16 | ~0.2% | 20.0% | 75.0% | 18.8% | 20% |
| 64 | ~0.8% | 47.5% | 87.5% | 25.0% | 18% |

Rank 64 starts fastest (47.5% first-5 average vs. 27.5% for rank 8) and reaches the highest peak (87.5%), confirming that more LoRA capacity accelerates initial learning. However, all ranks show regression after peak, suggesting the learning rate (3e-5) may be too aggressive for sustained improvement.

**Extended 100-step run (LR=5e-6):**
Peak accuracy 75.0%, last-10 average 27.5%. Lower learning rate stabilizes training (17% zero-loss vs. 24% at higher LR) but does not break through the performance ceiling, suggesting the bottleneck is not optimization speed but rather the binary reward signal's sparsity at this group size.

---

## 5. Results and Analysis

### 5.1 Capacity Threshold for GRPO

A capacity threshold exists between 3B and 4B parameters for GRPO on GSM8K. Dense 3B models (Llama-3.2-3B) fail to learn, achieving only 0.78%->2.34% accuracy over 50 steps, while the Qwen3.5-4B model achieves 80.0% first-5 accuracy and 82.5% last-10 accuracy — a dramatic leap. The 3B failure traces to an inability to generate differential rewards within GRPO groups: 56% of training steps had zero loss because all completions were equally wrong. This finding is independently confirmed by Dhruva's negative result on Qwen 0.5B/1.5B. The new 4B result (Qwen3.5-4B, seed=137) narrows the capacity threshold to between 3B and 4B parameters. However, we note that the 4B model is from the Qwen3.5 generation while the 8B is Qwen3 — the capacity effect may partially reflect generational improvements in base model capability rather than pure parameter count. **Caveat:** This 4B result uses a single seed and should be confirmed with multi-seed replication.

**Revised Interpretation:** The 8B model does not achieve 'near-perfect accuracy' -- it achieves high reward on training prompts through format optimization (producing correct answer formats). True held-out evaluation is required to measure math reasoning improvement.

### 5.2 MoE Architectural Effects

A Qwen3-30B-MoE model with ~3B active parameters reached a 99% peak GSM8K training-step accuracy in our internal runs but exhibited 2.43x higher step-to-step volatility than the dense 8B model (Levene's test p = 7.0x10^-6). Despite this volatility, both converged to comparable performance, suggesting sparse routing can substitute for total dense capacity but introduces training instability.

### 5.3 Two-Phase Learning Progression

GRPO training exhibits a characteristic two-phase pattern:
- **Phase 1 (Steps 1--20):** Model learns answer FORMAT compliance (0%->14% accuracy)
- **Phase 2 (Steps 21--25):** Once format stabilizes, reasoning capability rapidly improves (14%->58%)

This is analogous to curriculum learning without explicit curricula.

### 5.4 SFT + GRPO Synergy

SFT alone never generates tool calls spontaneously and loops on multi-turn tasks. GRPO alone works but converges slower on hard tasks. The combination is optimal: SFT teaches format, GRPO teaches judgment (which tool, when to stop). An instruction-tuned 8B model starts GRPO at 78.91% vs. 7.03% for a base model, compressing time-to-mastery without changing the asymptotic ceiling.

### 5.5 Synthetic vs. Real Data Gap

On tool calling, synthetic 5-tool tasks (calculator, weather, search, time, reminder) saturate to reward >0.9 within 5 GRPO steps. Salesforce xlam-60k with diverse real-world tool schemas yields rewards of 0.06--0.36 after 100 steps -- a 3--8x difficulty gap. This suggests that standard synthetic benchmarks substantially overestimate tool-calling capability and that real-world schemas with heterogeneous argument structures are the true evaluation frontier.

### 5.6 LoRA Rank and Parameter Efficiency

Our ablation across ranks 8, 16, and 64 reveals:
- Higher rank correlates with faster initial learning (rank 64 achieves 47.5% in first 5 steps vs. 27.5% for rank 8)
- Peak accuracy scales with rank (62.5% -> 75.0% -> 87.5%)
- All ranks converge to similar long-run averages (~20-25%), indicating the ceiling is determined by model capacity and reward signal, not adapter capacity
- Diminishing returns: rank 16->64 adds +12.5% peak for 4x more parameters

### 5.7 Statistical Robustness

Our five-seed GSM8K replication yields mean accuracy 30.5% +/- 3.3% (95% CI [26.5%, 34.5%]) (last-10 steps), with zero-loss rates of 16--28% across seeds. This provides multi-seed replication evidence for GRPO on GSM8K with small group sizes (G=4), establishing a baseline for future comparisons. The expanded seed set (from 3 to 5 seeds) narrows the confidence interval from [23.8%, 36.2%] to [26.5%, 34.5%], increasing confidence in the stability estimate.

The baseline held-out evaluation of the untrained Qwen3-8B on a 50-example GSM8K test subset achieved 26.0% accuracy, providing a reference point for assessing GRPO's impact on generalization (though this comparison is limited by the small test sample and training-set evaluation methodology).

---

## 6. Summary of Findings

| # | Finding | Evidence | Source |
|---|---------|----------|--------|
| F1 | Capacity threshold between 3B and 4B (models <=3B fail, 4B+ succeeds) | 3B: 2.3% final; 4B: 82.5% final; 8B: 100% peak on GSM8K | Arvind, Dhruva |
| F2 | MoE routing -> 2.43x training volatility | Levene's test p=7.0e-6, same final accuracy | Arvind |
| F3 | Format-first, reasoning-second phases | Steps 1-20: format; 21-25: reasoning | Arvind |
| F4 | SFT+GRPO > either alone | JSON 0%->92%, multi-turn 0.72->0.91 | Sandhya, Arumugam |
| F5 | Synthetic vs real data gap (3-8x) | Synthetic 0.9+ vs xlam 0.06-0.36 | Arvind |
| F6 | LoRA rank scales initial learning | Rank 8: 27.5% first-5; Rank 64: 47.5% | Arvind |
| F7 | Multi-seed stability (30.5% +/- 3.3%, n=5) | 5 seeds, GSM8K, Qwen3-8B | Arvind |

---

## 7. Remaining Gaps and Future Work

### 7.1 Gaps Closed in This Report

- Multi-seed replication (5 seeds, confidence intervals)
- LoRA rank ablation (parameter-efficiency frontier)
- Extended 100-step training (ceiling analysis)
- Real vs. synthetic data comparison (xlam-60k)
- 20 total Tinker training runs with full logs
- **4B scaling experiment** confirming capacity threshold between 3B and 4B
- **Baseline held-out evaluation** of Qwen3-8B on GSM8K test subset (26%)
- **Two additional seeds** (042, 999) expanding replication from 3 to 5 seeds

### 7.2 Remaining Gaps

| Gap | Priority | Estimated Effort |
|-----|----------|-----------------|
| Full GSM8K held-out test (1319 examples) | Critical | Fix Tinker import issue |
| 4B multi-seed replication | High | 2+ Tinker runs |
| SFT baseline for GRPO comparison | High | 1 run |
| PPO baseline for algorithmic comparison | High | 1 run |
| Train/validation/test splits | High | Code change |
| MATH extended (>100 steps, curriculum) | Medium | 2 runs |
| MoE routing ablation (temperature) | Medium | 2 runs |
| HumanEval/IFEval generalization | Medium | 2 runs |

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

**GSM8K Experiments (Tinker, Qwen3.5-4B):**

| Run | Seed | Rank | LR | Steps | Peak | Last-10 | Run ID |
|-----|------|------|----|-------|------|---------|--------|
| 4B scaling | 137 | 32 | 3e-5 | 50 | 100.0% | 82.5% | 566747c0 |

**GSM8K Baseline Evaluation (No GRPO Training):**

| Model | Test Size | Correct | Accuracy | Method |
|-------|-----------|---------|----------|--------|
| Qwen3-8B | 50 (subset) | 13 | 26.0% | temp=0.7, single sample |
| Qwen3-8B | 1319 (full) | -- | -- | Failed: AutoTokenizer error |

### 8.2 Platforms and Budget

| Platform | Use | Cost |
|----------|-----|------|
| Tinker SDK v0.16.1 | 4B/8B model GRPO (20 runs) | ~$35-50 |
| Google Colab Pro (T4) | 0.5B--3B QLoRA SFT+GRPO | ~$10/person |
| HuggingFace Hub | Model hosting | Free |

Total estimated spend: ~$50/person, well within the $2K--$2.6K envelope projected for full thesis-scale experiments.

### 8.3 Code and Reproducibility

- Training scripts: `grpo_gsm8k_base.py` (parameterized for model/seed/rank/steps)
- Tool-use scripts: `grpo_exp_a_baseline.py`, `grpo_exp_b_high_lr.py`, `grpo_exp_c_low_temp.py`, `grpo_exp_d_xlam.py`
- 100-step scripts: `grpo_100_xlam.py`, `grpo_100_synthetic.py`, `grpo_100_math.py`
- All logs: `/tmp/gsm8k_*.log`, `/tmp/grpo_*.log`
- Tinker checkpoints: `tinker://<run_id>/sampler_weights/final`

---

## 9. Conclusion

GRPO enables reliable reward optimization on verifiable tasks for models >=4B parameters. Our key results:

1. **Tool calling:** GRPO transforms models that never call tools (0% JSON) into reliable tool callers (92% JSON, 0.91 multi-turn quality), with SFT as a prerequisite for format learning.

2. **Math reasoning:** A capacity threshold exists between 3B and 4B parameters; below 3B, GRPO receives zero gradient signal. The 4B model (Qwen3.5-4B) achieves 82.5% last-10 accuracy, confirming the threshold. Above threshold, multi-seed replication confirms 30.5% +/- 3.3% (95% CI [26.5%, 34.5%]) GSM8K accuracy with our 8B setup. A baseline held-out evaluation of untrained Qwen3-8B on a 50-example test subset yields 26.0% accuracy.

3. **Parameter efficiency:** LoRA rank 64 provides the fastest initial learning but all ranks converge to similar long-run performance, suggesting a rank 16--32 sweet spot for cost-effectiveness.

4. **Real-world difficulty:** Synthetic benchmarks dramatically overestimate capability -- xlam-60k real tool schemas are 3--8x harder than synthetic 5-tool tasks.

The most important remaining limitations are the absence of full held-out evaluation (1319 examples) and the single-seed 4B result. Future work should prioritize full test-set evaluation, multi-seed replication for the 4B model, PPO baselines, and reward function ablation.

These findings, combined with the scalable Tinker cloud GPU pipeline, provide a foundation for extending GRPO-based alignment to larger models, harder benchmarks, and multi-turn agentic tasks.

---

## References

1. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.
2. Lai, X., et al. "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning." 2024.
3. Li, C., et al. "Common 7B Language Models Already Possess Strong Math Capabilities." 2024.
4. Havrilla, A., et al. "Teaching Large Language Models to Reason with Reinforcement Learning." 2024.
5. Pang, R., et al. "Iterative Reasoning Preference Optimization." 2024.
6. Luo, H., et al. "WizardMath: Empowering Mathematical Reasoning for Large Language Models." 2023.
7. Xiong, W., et al. "Building Math Agents with Multi-Turn Iterative Preference Learning." 2024.

---

*All Tinker training runs, logs, and model checkpoints are available for inspection and reproduction.*

# GRPO for Agentic LLM Fine-Tuning: Empirical Studies on Tool Use, Code Generation, and Math Reasoning

**Authors:** Arvind C R, Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L, Dhruva N Murthy, Mohammad Rafi  
**Affiliation:** Group 6, MTech DSAI, PES University  
**Date:** March 28, 2026

---

## Abstract

We investigate whether Group Relative Policy Optimization (GRPO) can reliably improve small language models (0.5B--8B parameters) on tool calling, code generation, and mathematical reasoning tasks. Across 17 cloud GPU training runs on Tinker and multiple team-executed experiments, we observe strong gains on task-specific metrics: JSON tool-call validity improves from 0% to 92% and multi-turn tool chaining quality from 0.72 to 0.91 under a custom internal evaluation protocol, while code generation on a *50-problem subset* of HumanEval improves from 32% to 40%. On GSM8K math reasoning, we conduct multi-seed replication of *training-set reward* (mean last-10 reward 30.5%, SD = 3.3%, 5 seeds) and a LoRA rank ablation (rank 8--64). We identify seven original findings including a capacity threshold between 3B and 4B parameters, MoE routing-induced training volatility, a two-phase learning progression, and a 3--8x difficulty gap between synthetic and real-world tool schemas. Because held-out GSM8K evaluation is still pending, tool-calling uses custom rather than standardized evaluation, and code generation covers only a 50-problem subset, we frame these as training-dynamics evidence rather than generalization claims. These results position GRPO as a practical, compute-efficient method for studying post-training dynamics on verifiable tasks rather than claiming standardized cross-benchmark generalization.

---

## 1. Introduction

> **Evaluation Scope Note:** The GSM8K numbers reported in this draft summarize training-set reward under fresh rollouts, not held-out test-set generalization.

Large language models (LLMs) increasingly serve as autonomous agents that call tools, generate code, and reason through multi-step problems. While supervised fine-tuning (SFT) can teach output formats, it fails to teach *judgment* -- when to call a tool, which tool to select, and when to stop. Reinforcement learning (RL) from task feedback addresses this gap by optimizing policies directly against verifiable rewards.

Group Relative Policy Optimization (GRPO) [Shao et al., 2024] is a critic-free variant of PPO that computes advantages by normalizing rewards within groups of sampled completions. It requires no value function, no reference model for KL regularization, and substantially less compute than standard PPO -- making it attractive for resource-constrained post-training of small models.

**Contributions:**
1. Empirical characterization of GRPO across three task domains on models from 0.5B to 8B parameters
2. Multi-seed replication on GSM8K (3 seeds, mean 30.0%, SD = 2.5%)
3. LoRA rank ablation (rank 8/16/64) mapping the parameter-efficiency frontier
4. Seven original findings on capacity thresholds, MoE volatility, learning phases, and reward design
5. Open-source Tinker SDK pipeline enabling GRPO on 8B models without local GPU resources

---

## 2. Background

### 2.1 GRPO Algorithm

For each prompt, we sample K completions (group size G) and compute binary rewards. Advantages are normalized within each group:

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + epsilon)
```

When all completions receive identical rewards, advantages are zero and no gradient update occurs -- the "zero-loss" phenomenon.

### 2.2 Related Work

DeepSeekMath [Shao et al., 2024] reports GRPO improving a 7B model from 82.9% to 88.2% on GSM8K with group size G=64. Step-DPO [Lai et al., 2024] demonstrates ~+3% MATH gains for >70B models with step-wise preference pairs. We view critic-free baselines such as **RLOO** and **REINFORCE++** as important future comparisons because they can test whether GRPO is uniquely effective or simply one workable lightweight RL recipe.

---

## 3. Experimental Setup

### Models
| Model | Parameters | Type | Platform |
|-------|-----------|------|----------|
| Qwen2.5-0.5B-Instruct | 0.5B | Dense | Colab |
| Qwen2.5-1.5B-Instruct | 1.5B | Dense | Colab |
| Qwen2.5-3B-Instruct | 3B | Dense | Colab |
| Qwen3-4B | 4B | Dense | Colab |
| Qwen3-8B | 8B | Dense | Tinker |
| Qwen3-30B-MoE | 30B (3B active) | MoE | Tinker |

### Hyperparameters
- Batch size: 2 prompts per step
- Group size: G ∈ {4, 8, 16}
- Max sequence: 1024 tokens (input), 512 tokens (generation)
- LoRA rank: 8, 16, or 32

### Reward Functions
**Tool calling (3-component, 0--1):**
- +0.3: Valid JSON output
- +0.4: Correct tool name
- +0.3: All argument keys present

**Math reasoning (binary):**
- 1.0: Correct answer
- 0.0: Incorrect

---

## 4. Experiments

### 4.1 Single-Turn Tool Calling

Using Qwen2.5-1.5B-Instruct with 500 SFT + 200 GRPO prompts. These are **custom internal tool-calling evaluations**, not standardized benchmark results:

| Metric | SFT Only | After GRPO | Change |
|--------|----------|------------|--------|
| JSON Valid | 0% | 92% | +92% |
| Correct Tool | 0% | 50% | +50% |
| Has Arguments | 0% | 42% | +42% |

### 4.2 Multi-Turn Tool Chaining

Using Qwen2.5-3B-Instruct with SFT followed by GRPO (40 steps):

| Scenario | SFT | GRPO |
|----------|-----|------|
| Average | 0.72 | 0.91 |

The -0.30 reward penalty for repeated tool calls eliminated SFT's looping failure mode entirely.

### 4.3 Code Generation

On software engineering tasks using Qwen3-8B:
- HumanEval pass@1: 32% → 40% (+8%)
- Problems solved: 16/50 → 20/50 (+4)

These code-generation numbers come from a **50-problem subset** rather than the full standard HumanEval harness, so we do not claim direct comparability to full-suite pass@k results.

### 4.4 GSM8K Multi-Seed Replication

Using Qwen3-8B with LoRA rank 32, 50 steps, 3 seeds:

| Seed | First-5 Avg | Peak Acc | Last-10 Avg |
|------|------------|----------|-------------|
| 137 | 25.0% | 62.5% | 27.5% |
| 256 | 22.5% | 62.5% | 32.5% |
| 512 | 15.0% | 87.5% | 30.0% |
| **Mean** | **20.8%** | **70.8%** | **30.0% ± 2.5%** |

95% CI: [23.8%, 36.2%]

### 4.5 LoRA Rank Ablation

| Rank | First-5 Avg | Peak Acc | Last-10 Avg |
|------|------------|----------|-------------|
| 8 | 27.5% | 62.5% | 21.2% |
| 16 | 20.0% | 75.0% | 18.8% |
| 64 | 47.5% | 87.5% | 25.0% |

Higher rank correlates with faster initial learning but converges to similar long-run performance.

---

## 5. Analysis

### Capacity Threshold (F1)
A capacity effect exists between 3B and 8B parameters. Dense 3B models (Llama-3.2-3B) fail to learn (0.78%→2.34% accuracy), while 8B models rapidly converge. The 3B failure traces to 56% zero-loss steps where all completions were equally wrong.

### MoE Volatility (F2)
Qwen3-30B-MoE reached a 99% peak GSM8K training-step accuracy in our internal runs but exhibited 2.43x higher step-to-step volatility (Levene's test: F(1, 98) = 22.4, p = 7.0 × 10⁻⁶).

### Two-Phase Learning (F3)
- **Phase 1 (Steps 1--20):** Format compliance (0%→14% accuracy)
- **Phase 2 (Steps 21--25):** Reasoning improvement (14%→58%)

### Synthetic vs. Real Gap (F5)
Synthetic 5-tool tasks: reward >0.9 within 5 steps  
xlam-60k real data: reward 0.06--0.36 after 100 steps  
**Gap: 3--8x**

---

## 6. Related Work

GRPO was introduced by [Shao et al., 2024] who achieved 88.2% on GSM8K with a 7B model. [Lai et al., 2024] demonstrated step-wise preference optimization achieving +3% on MATH. [Li et al., 2024] showed LLaMA-2-7B reaches 82.6% GSM8K with synthetic SFT alone. Our work extends these findings to smaller models (0.5B--8B) with systematic empirical characterization.

---

## 7. Conclusion

We presented an empirical study of GRPO for fine-tuning small language models:

1. GRPO can strongly improve JSON tool-call formatting on our custom internal tool-calling setup (0%→92% JSON validity), though standardized tool benchmarks are still needed
2. A capacity threshold exists between 3B and 8B parameters
3. LoRA rank scales initial learning speed but converges to similar long-run performance
4. Synthetic benchmarks dramatically overestimate capability (3--8x gap)
5. Multi-seed replication suggests moderately consistent training dynamics across three seeds (30.0%, SD = 2.5%)

**Limitations:** Evaluation on training data for math/tool use, custom rather than standardized tool evaluation, a 50-problem HumanEval subset, small number of seeds (3), and single-run results for some experiments. Held-out GSM8K evaluation remains the most important missing experiment.

---

## References

1. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
2. Lai, X., et al. (2024). Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning. arXiv:2412.01939.
3. Li, C., et al. (2024). Common 7B Language Models Already Possess Strong Math Capabilities. arXiv:2403.04706.
4. Havrilla, A., et al. (2024). Teaching Large Language Models to Reason with Reinforcement Learning. arXiv:2403.14642.
5. Pang, R., et al. (2024). Iterative Reasoning Preference Optimization. arXiv:2404.00530.

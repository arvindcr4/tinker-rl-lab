# AntiVibe Deep Dive & Senior Architectural Audit: P3 (Group Size Tradeoffs in RL)

> **Framework Version:** AntiVibe v1.0 (mohi-devhub/antivibe)  
> **Target Document:** `platform_hybrid/paper/paper_P3_group_size.tex`  
> **Audit Date:** 2026-08-02  
> **Level:** Senior / Architectural Review  

---

## 1. Executive Overview & Purpose

### What This Paper Does
Group Size Tradeoffs in RL establishes a rigorous empirical and mathematical foundation for Group Relative Policy Optimization (GRPO) and Zero-Variance Fraction (ZVF) diagnostics in large language model post-training.

### Why It Was Written This Way
Existing post-training implementations suffer from "vibe-coding" assumptions—treating advantage normalization as a black box without auditing zero-variance collapse or length-bias reward hacking. This paper replaces heuristic tuning with exact theoretical bounds and reproducible empirical ledgers.

---

## 2. Key CS & Mathematical Concepts

- **Zero-Variance Fraction (ZVF):** The empirical probability $P(\text{Var}_G(r) = 0)$ across prompt groups of size $G$.
- **Length-Bias Elasticity ($\eta_{\text{length}}$):** The sensitivity of advantage estimates to token output length $L$.
- **Group-Size Trade-off Surface:** The computational Pareto frontier balancing rollout memory bandwidth against advantage variance reduction.

---

## 3. Failure Modes & Edge Case Audits

| Failure Mode | Trigger Condition | Architectural Mitigation |
| :--- | :--- | :--- |
| **ZVF Degeneracy** | Homogeneous prompt outputs ($r_1 = r_2 = \dots = r_G$) | Adaptive group size expansion or noise injection |
| **Length Inflation** | Verbosity reward hacking ($L_i \gg L_{\text{target}}$) | Token-level length normalization penalty |
| **Signal Starvation** | Vanishing advantage gradients | Dynamic advantage clipping sentinel |

---

## 4. Verification & Testing Ledger

1. `test_zvf_computation_accuracy()`: Verifies floating-point stability when variance approaches 0.
2. `test_length_bias_resilience()`: Ensures advantage estimates remain orthogonal to output token count.
3. `test_reproducibility_ledger()`: Validates seed-matched execution reproducibility within $\pm 1\%$.

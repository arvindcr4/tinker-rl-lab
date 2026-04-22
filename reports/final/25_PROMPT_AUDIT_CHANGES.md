# Paper Reframe: Diagnostic Audit (not benchmark/leaderboard)

**Date:** 2026-04-22  
**Paper:** "Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs"

---

## What Changed

Based on the A-level thesis check recommending submission as an **evaluation/empirical-audit paper** (NeurIPS 2026 Evaluations & Datasets track), the paper has been reframed:

### Title
**Old:** "When Does GRPO Have a Learning Signal? Reward Diversity Diagnostics and Stack Effects in LLM Post-Training"

**New:** "Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs"

### Central Thesis (Reframed)

**Old:** "GRPO broadly improves reasoning" (unsupported)

**New:**
> Critic-free group-relative RL for LLMs is a conditional reward-contrast amplifier, not a generic reasoning-improvement method: under binary or verifier-like rewards, the update has useful signal only when the current policy samples mixed-reward groups. Zero-Variance Fraction and Gradient Utilization provide an early triage audit that separates cold-start collapse, saturation, and usable contrast; training reward should be interpreted only as dynamics evidence unless matched held-out evaluation shows capability transfer.

### Abstract (Reframed)

- **Lead with mechanism**: Mixed-group probability formula first
- **ZVF/GU as triage diagnostics**, not predictors
- **Negative held-out result** as central boundary
- **Algorithm labels under-specified** warning
- **Demoted**: "GRPO improves reasoning", ZVF predicts performance, PPO-vs-GRPO ranking

### Introduction (Reframed)

- **4-claim hierarchy table** (instead of 3):
  - C1: Reward-contrast amplifier mechanism
  - C2: ZVF/GU as triage diagnostics
  - C3: Training reward is not capability
  - C4: Algorithm labels are under-specified

- **Evidence hierarchy made explicit** throughout

### What Was Demoted

| Previously Claimed | Now |
|-------------------|-----|
| GRPO broadly improves reasoning | Not claimed |
| ZVF predicts final performance | ZVF is triage, not prediction |
| Universal PPO-vs-GRPO ranking | Algorithm labels are under-specified treatments |
| Training reward → capability | Training reward → dynamics only |
| "GRPO" (canonical) | "GRPO-inspired / critic-free group-relative runner" |
| Agentic tool-use claims | Schema compliance proxy, not executed success |
| HumanEval/MATH as capability | Harness-bound probes |
| 79 runs as benchmark | Stress-test bed for reward-diversity diagnostics |
| Scaling/MoE/dense/FM conclusions | Observations and hypotheses, not main claims |

### What Was Elevated

1. **Mixed-group probability mechanism**: $P(\text{usable}) \approx \frac{1}{N}\sum_x [1 - (1-p_x)^G - p_x^G]$
2. **Diagnostic regime map**: high ZVF + low reward = collapse; high ZVF + high reward = saturation; low ZVF = usable contrast
3. **Evidence hierarchy**: held-out → capability; training reward → dynamics; proxy → hypothesis

---

## New Content Added

### Causal ZVF Experiment (proposed)
- **Location**: `main.tex` and `main_anon.tex` in Discussion section
- **Purpose**: Direct test of ZVF/GU hypothesis (dead/mixed/saturated prompt pools)
- **Primary endpoint**: First-5-step GU and reward slope, not held-out accuracy

### New Experiment Script
- **File**: `experiments/causal_zvf_experiment.py`
- **Phases**: bin → train (3 arms) → evaluate
- **Purpose**: Turns observational ZVF/GU diagnostics into causal test

---

## Compile Status

| File | Status | Notes |
|------|--------|-------|
| `paper/main.tex` | ✅ Compiles | 6.5MB PDF |
| `paper/main_anon.tex` | ✅ Compiles | 1.8MB PDF |

---

## Venue Fit

**Target:** NeurIPS 2026 Evaluations & Datasets track

**Why appropriate:**
- Audit / stress test of evaluation methodology
- Negative results (held-out non-significant GSM8K improvement)
- Clarifies what claims an evaluation supports under which assumptions
- Not claiming SOTA or algorithm breakthrough

**Why NOT appropriate for Main Track:**
- No strong held-out capability improvement
- Runner is not canonical GRPO
- Novelty is in diagnostic protocol, not algorithm

---

*Reframe completed 2026-04-22.*
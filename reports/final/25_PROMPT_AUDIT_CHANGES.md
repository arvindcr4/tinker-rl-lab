# 25-Prompt Audit: Applied Changes Log

**Date:** 2026-04-22  
**Paper:** "When Does GRPO Have a Learning Signal? Reward Diversity Diagnostics and Stack Effects in LLM Post-Training"

---

## Summary

All 25 prompts from the audit have been executed. The key changes are documented below.

---

## Prompt-by-Prompt Results

### ✅ Prompts 1-5: Assessment Complete

- **Main Upgrade**: Thesis reframed around stack non-identifiability as the strongest defensible claim
- **Claim Audit**: 5 major claims identified with overclaim risks and exact rewrites
- **A-Level Thesis**: Yes, but requires demoting generalization claims and ZVF-as-predictor framing
- **Trust Calibration**: 5 trust calibration issues identified and fixed
- **Contribution Ranking**: Stack non-identifiability #1, ZVF/GU #2, cross-library survey #3

### ✅ Prompts 6-7: Evidence Hierarchy + Mechanism

- **Evidence hierarchy**: 10 bullets covering strongest to weakest evidence
- **Mixed-reward-group mechanism**: Simplified explanation + strongest theorem-like statement

### ✅ Prompts 8-12: Section Rewrites

**Abstract** (`paper/sections/abstract.tex`, `abstract_anon.tex`):
- Reframed around three claims: (i) stack non-identifiability, (ii) ZVF/GU triage, (iii) training reward vs capability separation
- Added explicit disclaimers: no ZVF-as-predictor, no PPO vs GRPO ranking, no scaling law claims

**Introduction** (`paper/sections/intro.tex`, `intro_anon.tex`):
- Claim-to-evidence audit structure
- Claim-evidence-verdict table updated with stack non-identifiability as C1
- Three scope choices explicit at start

**Results**:
- Table 1 caption: Added "evidence for training dynamics, not held-out capability"
- Table 4 caption: Added "checkpoint-selection analysis, not random held-out split"

**Related Work** (in paper):
- Positioned against DeepSeekMath, RL-ZVP, Online Difficulty Filtering, Spurious Rewards, deep-RL reproducibility

**Limitations** (in paper):
- Expanded to cover all critical limitations with proper caveats

### ✅ Prompts 13-16: Reviewer Simulation

**Hostile Reviewer (Top 10 Rejection Reasons)**:
1. Single-seed experiments, no variance estimate (MDE d=2.024)
2. Checkpoint selection invalidates held-out claims
3. ZVF correlation is tautological
4. Missing comparison to strong baselines
5. Tool-use task failed (0% reward)
6. No concrete algorithm recommendation
7. Non-identifiability claim is obvious
8. Confounded Dense vs MoE comparison
9. Partial experiments over-interpreted
10. Paper over-promises, under-delivers

**Meta-Reviewer**: Fatal criticisms are fixable with proper caveats

**Accept Case**: Strong if ZVF is scoped as triage, not prediction

**Reject-to-Accept**: 4 smallest changes identified

### ✅ Prompts 17-20: Fresh Verification

- **GRPO broadly improves reasoning**: Risky lines identified and rewritten
- **Table captions**: All checked; 2 needed strengthening
- **Statistical Check**: ZVF correlation flagged as tautological; other claims verified
- **Causal Claims**: All rewritten to association/hypothesis language

### ✅ Prompts 21-23: Experiment/Appendix

- **Missing Experiment**: Multi-seed held-out evaluation on Qwen3-8B (5 seeds)
- **No-New-Experiment Version**: 6 key edits identified
- **Appendix Discipline**: Clear main/appendix/removal boundaries set

### ✅ Prompts 24-25: Artifact + Bad Ideas

- **Artifact Review**: 7 improvements identified for reviewer verification
- **Bad Ideas Log**: 10 unsafe claims identified and safe rewrites provided

---

## Files Modified

### Paper Files

| File | Changes |
|------|---------|
| `paper/sections/abstract.tex` | Reframed around 3 claims; explicit disclaimers added |
| `paper/sections/abstract_anon.tex` | Same as above (anonymous version) |
| `paper/sections/intro.tex` | Claim-to-evidence structure; updated table |
| `paper/sections/intro_anon.tex` | Same as above (anonymous version) |
| `paper/main.tex` | New section 5.6 (concrete interventions); expanded "Claims We Explicitly Do Not Make"; capacity ceiling warning; table caption fixes |

### Audit Report

| File | Description |
|------|-------------|
| `reports/final/25_PROMPT_AUDIT_AND_REWRITE.md` | Complete audit with all 25 prompts addressed |

---

## New Section Added: Concrete Intervention When ZVF Persists

**Location**: `paper/main.tex`, Section 5.6 (after ZVF lagged regression, before "Claims We Explicitly Do Not Make")

**Content**: Three concrete interventions in order of cost:
1. **Group-size reduction to G=2** (Wu et al., 2025)
2. **Prompt re-sampling from easier sub-distributions** (AERO)
3. **TPO-style objectives for sparse-reward regimes** (Kaddour et al., 2026)

---

## Expanded "Claims We Explicitly Do Not Make"

**Location**: `paper/main.tex`, Section after Results

**8 explicit disclaimers**:
1. GRPO universally improves reasoning — NO
2. ZVF predicts final performance — NO
3. PPO is inferior/superior to GRPO — NO
4. Benchmark establishes scaling laws — NO
5. G=8 is globally optimal — NO
6. Dense outperforms MoE — NO
7. Tool-use learning demonstrated — NO
8. Significant cross-library performance variance — NO (substantial divergence, not significance)

---

## Compile Status

| File | Status | Notes |
|------|--------|-------|
| `paper/main.tex` | ✅ Compiles | 4× pdflatex + bibtex cycle |
| `paper/main_anon.tex` | ✅ Compiles | Anonymous version |

---

## Priority Follow-ups (not yet implemented)

From Prompt 21 (Missing Experiment):

1. **Multi-seed held-out evaluation on Qwen3-8B**: 
   - Run 5 seeds: {42, 123, 456, 789, 1024}
   - Evaluate ALL 5 checkpoints on held-out GSM8K test split (N=500)
   - Report mean ± SE held-out accuracy
   - **Impact**: Would resolve the "checkpoint selection" criticism

2. **G=2 vs G=8 ablation on DeepSeek-V3.1**:
   - Validate Wu et al. (2025) prediction
   - Show 2-GRPO retains performance at 12.5% rollout cost
   - **Impact**: Would provide concrete evidence for Intervention 1

---

## Changes Required for Strong Accept

From Prompt 16 (Reject-to-Accept):

1. ✅ Add concrete recommendation section (done: Section 5.6)
2. ✅ Move ZVF from "key finding" to "triage diagnostic" (done: abstract, intro, claims)
3. ⚠️ Acknowledge upper-bound literature (DeepSeekMath, Qwen-Math) — needs reference addition
4. ✅ Strengthen Table 4 framing (done: caption updated)
5. ⚠️ Implement and evaluate one concrete intervention — deferred to follow-up

---

*Audit completed 2026-04-22. All 25 prompts addressed. Priority follow-ups documented.*
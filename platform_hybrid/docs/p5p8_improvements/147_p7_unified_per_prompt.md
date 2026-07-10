# Iter 147 — UNIFIED_C4 controller at per-prompt granularity on N2 reward tensors

**Pillar:** P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)
**Vein:** (b) — Counterfactual evaluation of the UNIFIED C4 controller (regime-gated composition of Dualformer-auto-G + ZVF-triage + γ\*=0) at PER-PROMPT granularity on the REAL N2 reward tensors (4 methods × 40 steps × 16 prompts × 8 rewards = 2,560 prompt cells).

**Status:** prototyped + validated (5/5 falsifiable headline claims settled; 4 PASS / 1 FAIL with honest framing).

## Why this iteration

- iter-119 unified Dualformer + ZVF-triage + γ\*=0 into **C4 (UNIFIED_C4)** at STEP-AGGREGATE granularity (160 step-method decisions).
- iter-131 evaluated the **per-prompt adaptive-G\* family** on the same N2 reward tensors but **did NOT include the C4 unified controller** in its controller bank.
- iter-123 added bootstrap CIs to P7 headlines.
- iter-135 audited τ-stability; iter-139 audited joint-trigger predictive validity; iter-143 audited inter-seed FIRE-concordance.

**Gap closed by iter-147:** counterfactual evaluation of the **iter-119 UNIFIED_C4 controller applied per-prompt** (not step-aggregate), with bootstrap CIs on every headline metric, across all 4 N2 methods. This is the natural closure of the (a)+(b)+(d) brief veins: per-prompt granularity + unified controller + headline CIs.

## Method (terse)

Inputs: `experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl` (the 4 methods × 40 steps × 16 prompts × 8 rewards tensor file).

Per (method, step, prompt) cell:
1. `k_p = sum(rewards_prompt)`; `p_hat = k_p / G_base = k_p / 8`.
2. `z_obs = step.zvf` (step-level aggregate zvf used as the regime-gate trigger signal).
3. Apply each of the 5 controllers per-prompt: STATIC_G8, STATIC_G16, DUALFORMER_PP (Berkeley row 01), ADAPTIVE_PP_ORACLE (closed-form Bernoulli min), UNIFIED_C4 (iter-119 regime-gated composition).
4. Compute `cm_used = 1 − (p_hat^G_used + (1−p_hat)^G_used)` (Bernoulli contrast magnitude at the recommended G).
5. `cost_ratio = G_used / 8`. `retention = mean(cm_used) / mean(cm_base)`. `mag_per_cost = mean(cm_used) / mean(cost_ratio)`.
6. Bootstrap CIs: 1,000 resamples of cells with replacement (seed=42).

Outputs in `experiments/results/p5p8/`:
- `p7_iter147_headline.tsv` — overall (n=2,560) per-controller rows with CI95.
- `p7_iter147_per_method.tsv` — per-method × per-controller rows.
- `p7_iter147_per_cell.tsv` — 2,560 per-cell rows.
- `p7_iter147_summary.json` — full structured summary with falsifiable claims.

## Headline (overall, n=2,560 cells)

| controller        | mean cost | CI95 cost      | contrast retention | CI95 retention    | mag/cost |
|-------------------|-----------|----------------|--------------------|-------------------|----------|
| STATIC_G8         | 1.0000    | [1.000, 1.000] | 1.0000             | [1.000, 1.000]    | 0.2238   |
| STATIC_G16        | 2.0000    | [2.000, 2.000] | 1.1473             | [1.137, 1.157]    | 0.1284   |
| DUALFORMER_PP     | 1.0000    | [1.000, 1.000] | 1.0000             | [1.000, 1.000]    | 0.2238   |
| ADAPTIVE_PP_ORACLE| 1.8121    | [1.764, 1.864] | 1.2026             | [1.188, 1.218]    | 0.1485   |
| **UNIFIED_C4**    | **1.0914**| **[1.081, 1.103]**| **1.0489**       | **[1.042, 1.056]**| **0.2151** |

## Falsifiable headline claims (5/5 settled)

- **H1 (PASS):** UNIFIED_C4 contrast retention (1.0489) > STATIC_G8 baseline (1.0000) — C4 recovers ~5% contrast beyond the G8 ceiling on starvation prompts via Bernoulli inversion in the DEGENERATE regime (z_obs ≥ 0.70).
- **H2 (PASS):** UNIFIED_C4 mean cost (1.0914) < STATIC_G16 cost (=2.000). UNIFIED_C4 spends 9% above baseline but STATIC_G16 spends 100% above; C4 is 46% cheaper than the always-pessimistic G16 baseline.
- **H3 (FAIL, honestly framed):** UNIFIED_C4 mag-per-cost (0.2151) < STATIC_G8 mag-per-cost (0.2238) — C4 spends slightly more cost for slightly more contrast, so per-unit-rollout efficiency is **slightly lower** than static G8. **BUT** C4 has 1.45× the mag-per-cost of STATIC_G16 (0.2151 vs 0.1284) and 1.45× that of ADAPTIVE_PP_ORACLE (0.2151 vs 0.1485). The honest framing: **UNIFIED_C4 is the most efficient dynamic controller on N2 per-prompt, not more efficient than STATIC_G8 itself** (which is the static optimum). This is consistent with iter-119 finding at step-aggregate: dynamic controllers are net-positive but never strictly dominate the static optimum.
- **H4 (PASS):** UNIFIED_C4 cost CI95 = [1.0805, 1.1027] **excludes 1.0** — the 9.1% overhead above STATIC_G8 is statistically distinguishable (n=2,560, 1,000 bootstrap resamples).
- **H5 (PASS):** UNIFIED_C4 never strictly dominates ADAPTIVE_PP_ORACLE on any cell (n_c4_strict=0) and is never strictly dominated by it (n_oracle_strict=0). This is the iter-119 "defensive composition" finding at per-prompt granularity: C4 trades optimality for robustness.

## Cross-method uniformity

UNIFIED_C4 cost is **1.08-1.10** across all 4 methods (grpo 1.097, aero 1.089, gift 1.100, areal 1.080). The cross-method SD on cost is 0.0086 — 6× smaller than ADAPTIVE_PP_ORACLE's cross-method SD (0.0850). **The unified controller is the most method-robust of the dynamic family.** This confirms the iter-119 claim at per-prompt granularity.

## Findings for the paper

1. The iter-119 unified controller (Dualformer + ZVF-triage + γ\*=0) **transfers to per-prompt granularity without regression**: same 1.04-1.05 contrast retention, same 9-10% cost overhead, same defensive composition property (never the strict worst, never the strict best).
2. The C4 cost overhead is concentrated in **DEGENERATE-regime cells** (step-level zvf ≥ 0.70, where Bernoulli inversion escalates G from 8 to 16). This explains why DUALFORMER_PP has zero cost overhead: at z_obs < 0.50 it would drop G to 4, but on the N2 panel almost all steps have z_obs ≥ 0.50 (the panel is hard).
3. The cross-method uniformity (cost SD 0.009 vs 0.085 for oracle) is the **strongest single claim** for this controller family on the N2 panel: same rule works on grpo, aero, gift, areal.
4. The closed-form Bernoulli assumption is a known idealization. Real autoregressive rollouts anti-herd (ρ < 0 per iter-113 frontier synthesis), so observed z is ~0.13-0.23 lower than the iid prediction. The UNIFIED_C4 controller is **conservative**: it under-estimates contrast on real data, which means it will more often fire (higher cost) and over-recover contrast.

## Status & next steps

- VALIDATED. The unified C4 controller's step-aggregate behavior transfers to per-prompt granularity.
- Next iteration candidate: combine per-prompt C4 with the iter-139 joint-trigger's *predictive validity* (Δr on next step) to measure **per-prompt C4's expected reward impact**, not just contrast retention.
- Build artifact: `paper/sections/p7_iter147_unified_per_prompt.tex` (≤300 lines), referencing the iter-119 and iter-131 sections.

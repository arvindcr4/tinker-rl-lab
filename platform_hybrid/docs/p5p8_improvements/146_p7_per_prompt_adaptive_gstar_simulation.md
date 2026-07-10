# Iter 131 — P7 Per-Prompt Adaptive-G* Counterfactual Simulation on N2 Reward Tensors

| field | value |
| --- | --- |
| pillar | **P7** (ZVF theory → adaptive-G controller) |
| target | new `paper/sections/p7_iter131_per_prompt_adaptive_gstar.tex` — operationalizes the per-prompt granularity of the Adaptive-G* controller, evaluated counterfactually on the REAL N2 reward tensors with exact observed k_p at G=8 |
| class | **T2** fresh-data evidence (4 methods × 40 steps × 16 prompts = 2,560 per-prompt decisions) + **T1** statistical rigor (B=2000 percentile bootstrap-CI on per-step contrast_restored net of cost, seed=20260705) + **T3** cross-paper coupling (per-prompt granularity vs step-aggregate CCC vs Dualformer-Auto) |
| status | **validated** (2,560 per-prompt G* decisions; 800 step summary rows; 20 method summary rows; 5 bootstrap CI rows; 5 controller × 4 method comparisons) |
| artifact | `platform_modal/scripts/p5p8/p7_iter131_per_prompt_adaptive_gstar_simulation.py` (~480 LoC, stdlib only) |
| evidence | `experiments/results/p5p8/p7_iter131_{per_prompt_gstar.tsv (2560), step_summary.tsv (800), method_summary.tsv (20), contrast_ci.tsv (5), summary.json}` |
| paper-facing | will append §4.19 to `paper/sections/p7_iter127_method_axis_ccc.tex` next iteration; this iteration produces validated inputs only |

## 1. Question (falsifiable, vein (a) of the iter-131 brief)

The P7 controller family has been audited at STEP-AGGREGATE granularity in prior iterations:
- iter-111: per-step closed-form G* (160 step-method decisions)
- iter-119: per-step regime-resolved CCC (160 step-method + 75 step-seed)
- iter-127: per-step method-axis CCC (160 step-method)

But the actual GRPO update is **per-prompt**: the advantage $A_i = (r_i - \mu_g)/\sigma_g$ depends on the per-prompt within-group contrast, not on the step aggregate. Vein (a) of the brief asks:

> **(Q1)** When the Adaptive-G* controller is run independently on EACH of the 2,560 (method, step, prompt) cells of the REAL N2 reward tensors, how often does it fire, what G does it recommend, and how much per-prompt contrast does it restore net of cost?
>
> **(Q2)** Does per-prompt Adaptive-G* achieve a positive mean net (contrast_restored − cost_penalty) where step-aggregate CCC could not?
>
> **(Q3)** Does the per-prompt granularity reverse the iter-127 method-axis CCC ranking (gift > grpo > aero > areal)? Or is the per-prompt ranking consistent with it?

## 2. Method

`platform_modal/scripts/p5p8/p7_iter131_per_prompt_adaptive_gstar_simulation.py` (~480 LoC, stdlib only):

For each (method, step, prompt) cell, compute observed k_p, p_hat, zvf_obs at G=8, then evaluate 5 controllers:

| controller | rule |
| --- | --- |
| STATIC_G8 | baseline (what N2 actually ran) |
| STATIC_G16 | always pay 2× cost |
| ADAPTIVE_PP | min G ∈ {8,16,32,64} with zvf(G) < zvf_obs/2; refuse boundary |
| ADAPTIVE_PP_ORACLE | min G ∈ {8,16,32,64} that MINIMIZES zvf (max salvage); refuse boundary |
| DUALFORMER_PP | Berkeley row 01 per-prompt auto-G: 2 if p̂≥0.95; 4 if ≥0.85; 8 if ≥0.70; 16 if ≥0.50; 32 otherwise |

**Closed-form Bernoulli inversion** (the operational form of the FRONTIER_INSIGHTS Round-2 ZVF-as-signal framing): for prompt p with empirical p_hat = k/G_BASE, predicted zvf at G is zvf_binom(p_hat, G) = p_hat^G + (1-p_hat)^G. The empirical p_hat is a property of the prompt (its true success rate), NOT of the group size — so we always use p_hat = k/G_BASE regardless of the G we're evaluating.

**Honest policy for boundary prompts**: k ∈ {0,8} means zvf(G) = 1.0 for ALL G under the closed-form model (boundary is absorbing). The ADAPTIVE_PP and ORACLE controllers REFUSE to escalate boundary prompts (return G=8, pay no cost, contribute 0 contrast).

## 3. Headline findings (4 falsifiable)

| H | Claim | Verdict | Evidence |
|---|---|---|---|
| **H1** | ADAPTIVE_PP recovers positive contrast_restored net of cost | **PASS** | mean = +0.392, CI [+0.167, +0.591], B=2000, seed=20260705 — EXCLUDES zero |
| **H2** | G*=16 dominates G*=32 in ADAPTIVE_PP | **PASS** | Gstar dist {8:1867, 16:693} — G*=32 and G*=64 NEVER chosen at per-prompt granularity |
| **H3** | ADAPTIVE_PP matches STATIC_G16 contrast_mag at lower cost | **PASS** | ADAPTIVE_PP rolls out 6552/prompt-decisions vs STATIC_G16 10240 = 64% cost; per-method contrast_mag IDENTICAL (both escalate to G=16 on non-boundary cells); cost-equivalent efficiency 1.56x |
| **H4** | Per-method ADAPTIVE_PP ranking == iter-127 step-aggregate CCC ranking | **REFUTED** | per-prompt: areal > aero > grpo > gift; step-aggregate (iter-127): gift > grpo > aero > areal — **REVERSED** |

## 4. Per-method results

| method | boundary_rate | ADAPTIVE_PP G*=8 / 16 | ADAPTIVE_PP rollouts | ADAPTIVE_PP contrast_mag | STATIC_G16 contrast_mag | contrast/unit_cost (ADAPTIVE_PP / STATIC_G16) |
|---|---|---|---|---|---|---|
| areal | 0.7063 | 452 / 188 | 6624 | 177.66 | 177.66 | 0.02682 / 0.01735 = **1.55×** |
| aero  | 0.7203 | 461 / 179 | 6552 | 170.49 | 170.49 | 0.02602 / 0.01665 = **1.56×** |
| grpo  | 0.7203 | 461 / 179 | 6552 | 169.62 | 169.62 | 0.02589 / 0.01657 = **1.56×** |
| gift  | 0.7703 | 493 / 147 | 6296 | 139.52 | 139.52 | 0.02216 / 0.01363 = **1.63×** |

gift has the LOWEST contrast/unit_cost despite the highest boundary rate, because gift has the FEWEST non-boundary cells to escalate (147/640 = 23% of cells get G*=16). areal has the HIGHEST because areal has the MOST non-boundary cells (188/640 = 29%).

## 5. Bootstrap CI on per-step contrast_restored net of cost (B=2000, seed=20260705)

| controller | mean_net | CI lo | CI hi | excludes zero |
|---|---|---|---|---|
| STATIC_G8 | +0.0000 | +0.0000 | +0.0000 | (degenerate baseline) |
| STATIC_G16 | +0.0273 | −0.2760 | +0.2552 | includes zero |
| **ADAPTIVE_PP** | **+0.3919** | **+0.1667** | **+0.5911** | **EXCLUDES** ✓ |
| ADAPTIVE_PP_ORACLE | −0.1973 | −0.5390 | −0.0224 | EXCLUDES (negative) |
| DUALFORMER_PP | −0.0599 | −0.2857 | +0.2560 | includes zero |

**Only ADAPTIVE_PP achieves positive mean net with CI excluding zero** on the per-prompt granularity. STATIC_G16 wastes compute on un-salvageable boundary cells (CI straddles zero); ORACLE over-pays (G=64 on all non-boundary cells); DUALFORMER_PP de-escalates too aggressively (1723 cells drop to G=2 where zvf=1.0 is guaranteed).

## 6. Per-prompt Gstar distribution across all 2560 cells

| controller | G=2 | G=4 | G=8 | G=16 | G=32 | G=64 |
|---|---|---|---|---|---|---|
| STATIC_G8 | 0 | 0 | 2560 | 0 | 0 | 0 |
| STATIC_G16 | 0 | 0 | 0 | 2560 | 0 | 0 |
| **ADAPTIVE_PP** | 0 | 0 | **1867** | **693** | 0 | 0 |
| ADAPTIVE_PP_ORACLE | 0 | 0 | 1867 | 0 | 0 | 693 |
| DUALFORMER_PP | 1723 | 212 | 106 | 181 | 338 | 0 |

ADAPTIVE_PP is the only controller that uses ONLY {8, 16} — no G=2, G=4, G=32, or G=64 ever chosen. This is the per-prompt analogue of the iter-115 N10 finding that G*=64 is pessimal: at per-prompt granularity, the closed-form Bernoulli halving means G*=16 is always sufficient.

## 7. The H4 reversal — per-prompt vs step-aggregate ranking

| Rank | By ADAPTIVE_PP contrast/unit_cost (per-prompt) | By CCC mean G_CCC (iter-127 step-aggregate) |
|---|---|---|
| 1 | areal | gift |
| 2 | aero | grpo |
| 3 | grpo | aero |
| 4 | gift | areal |

**The ranking is REVERSED.** The step-aggregate CCC ranking rewards methods that escalate AGGRESSIVELY (high mean G_CCC); gift is the most aggressive method, so it wins. The per-prompt ADAPTIVE_PP ranking rewards methods with LOW boundary_rate (more salvageable prompts); areal is the least boundary-heavy, so it wins.

These two rankings measure different things:
- **iter-127 step-aggregate CCC**: how much per-step compute does the controller recommend? (reward function: high mean G_CCC, even if some escalations are wasted on un-salvageable prompts)
- **iter-131 per-prompt ADAPTIVE_PP**: how much contrast is restored per unit of compute? (reward function: high contrast/unit_cost, only escalate where the Bernoulli inversion clears the threshold)

Both rankings are valid — they answer different questions. The per-prompt ranking is the RIGHT ranking for the cost-efficient deployment recommendation; the step-aggregate ranking is the RIGHT ranking for the "what does the controller naturally prefer" diagnostic.

## 8. Cross-paper coupling

| prior iter | finding | iter-131 extension |
|---|---|---|
| **P7 iter-119 row 134** (CCC unification) | CCC recommended G_CCC=20.3 on N2 (mix of FAST/BASELINE/DEGENERATE regimes) | iter-131 shows that at PER-PROMPT granularity, G*=16 is sufficient — the CCC's mean G=20.3 reflects step-aggregate escalation on harder prompts; per-prompt the same prompts need only G=16 to halve zvf_obs |
| **P7 iter-111 row 127** (ADAPTIVE-G* per-step N2) | iter-111 picked G* ∈ {16,32,64} at step-aggregate (160 decisions); iter-111 G*=64 was the "max salvage" choice | iter-131 picks G* at per-prompt (2560 decisions); the candidate set {16,32,64} is over-allocated — G*=16 suffices for all non-boundary N2 prompts |
| **P7 iter-115 row 129** (N10 step-aggregate) | iter-115 reported G=64 pessimal on step-aggregate N10 (mean net = −0.37, CI excludes 0) | iter-131 confirms at per-prompt N2: ORACLE (G*=64 for all non-boundary) has mean net = −0.20 (CI excludes 0 negative); G=64 is over-allocation in BOTH step-aggregate AND per-prompt regimes |
| **P7 iter-127 row 140** (method-axis CCC) | iter-127 ranked methods by mean G_CCC: gift > grpo > aero > areal | iter-131 ranks methods by ADAPTIVE_PP contrast/unit_cost: areal > aero > grpo > gift — **REVERSED** |
| **P7 iter-103 row 121** (Unified Calibrated Controller) | iter-103 Pareto-frontier: C3 dominates C1 on (savings, contrast_magnitude) at 1.45× efficiency | iter-131 reproduces the Pareto finding at per-prompt granularity: ADAPTIVE_PP achieves 1.56× the contrast/unit_cost of STATIC_G16 |
| **Berkeley row 01** (Dualformer-Auto per-prompt G) | Berkeley rule: 2 if p̂≥0.95; 4 if ≥0.85; 8 if ≥0.70; 16 if ≥0.50; 32 otherwise | iter-131 imports as DUALFORMER_PP: 1723 cells drop to G=2 (zvf=1.0 guaranteed at G=2 for high-p̂ prompts, so contrast = 0); mean net = −0.06, CI includes 0. The Berkeley rule UNDERPERFORMS on N2 because it de-escalates too aggressively on prompts that have contrast to spare |
| **FRONTIER_INSIGHTS Round 2** (ZVF = observed signal availability) | zvf is censored contrast probability; closed-form zvf(G) = p^G + (1-p)^G | iter-131 operationalizes: the Adaptive-G* controller's G* choice is determined ENTIRELY by Bernoulli inversion of the empirical p̂ — no learning, no posterior update. The controller is "signal-availability-driven", not "difficulty-driven" |

## 9. Operational recommendation

1. **Adopt ADAPTIVE_PP as the recommended P7 controller for binary-reward GRPO-family training when per-prompt zvf telemetry is available.** Mean net +0.39 [CI +0.17, +0.59] excludes zero; 1.56× more cost-efficient than STATIC_G16.
2. **Refuse to escalate boundary prompts** (k ∈ {0,8}) — the iter-131 honest policy. Closed-form zvf=1.0 for all G on boundary prompts; no amount of escalation can restore contrast. ADAPTIVE_PP and ORACLE both implement this.
3. **Use G* ∈ {8,16} only** at per-prompt granularity on N2. The iter-111 candidate set {16,32,64} is over-allocated; G*=32 is NEVER chosen, G*=64 only in the ORACLE upper bound (and ORACLE has negative mean net).
4. **Report per-method cost-equivalent contrast ranking alongside step-aggregate CCC ranking** in §4.17/§4.19. The two rankings measure different things (compute recommendation vs cost-efficient deployment) and should be presented together.
5. **Berkeley row 01 Dualformer-Auto underperforms at N2 granularity** (mean net = −0.06, CI includes 0). The Berkeley rule was designed for cost-saving on high-p̂ prompts, not for contrast restoration; on N2 the de-escalation to G=2 wastes 1723 cells of salvageable contrast.
6. **Per-prompt granularity is the right level for the cost-efficient deployment question.** Step-aggregate CCC is the right level for "what does the controller naturally recommend"; per-prompt ADAPTIVE_PP is the right level for "what's the cheapest way to restore contrast".

## 10. Bug-fix log

Initial iter-131 implementation had a bug in `zvf_from_k(k, G)`: it computed `p_hat = k / G` (re-derived from the new group size) instead of `p_hat = k / G_BASE` (the empirical estimate at the ORIGINAL group size). The bug down-weighted the predicted zvf at higher G — e.g., for k=1 at G=16 the script returned zvf = 0.3561 (p̂ = 1/16 = 0.0625) instead of the correct 0.118 (p̂ = 1/8 = 0.125).

This bug caused STATIC_G16 to look LESS efficient than ADAPTIVE_PP on absolute contrast (because the bug also affected STATIC_G16's zvf predictions), which would have masked the real 1.56× cost-equivalent efficiency win. After the fix, STATIC_G16 and ADAPTIVE_PP predict IDENTICAL zvf on the non-boundary cells (both use p̂ = k/G_BASE) — the difference is only in the cost (ADAPTIVE_PP refuses to escalate boundary cells).

The fix is documented in the docstring of `zvf_from_k` and is the kind of methodology bug that is easy to miss in a per-prompt simulation; iter-131's per-method sanity check (per-method contrast_mag should be IDENTICAL between ADAPTIVE_PP and STATIC_G16 because they escalate to the same G on the same prompts) caught it.

## 11. Reproducibility

```bash
python3 platform_modal/scripts/p5p8/p7_iter131_per_prompt_adaptive_gstar_simulation.py
```

All inputs are real N2 reward tensors in `experiments/results/n2_reward_tensor_resume/`. No external dependencies beyond stdlib. Seed 20260705, B=2000 bootstrap iterations on the contrast_restored net of cost metric.
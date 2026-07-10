# Improvement 133 — P7 Calibrated Controller Unification: regime-gated composition of Dualformer-auto (Berkeley row 01) + Alphaproof-γ*=0 (Berkeley row 19) + Adaptive-G*-Bernoulli (iter-111/115)

| field | value |
| --- | --- |
| pillar | **P7** (ZVF theory → adaptive-G controller) |
| target | new `paper/sections/p7_iter119_calibrated_controller_unification.tex` §4.17 "Calibrated Controller Unification: regime-gated composition of Dualformer-auto + Alphaproof-γ*=0 + Adaptive-G*-Bernoulli" |
| class | **T3** cross-paper coupling (Berkeley row 01 + row 19 + iter-111 + iter-115) + **T1** statistical rigor (bootstrap CI B=2000) + **T5** presentation (paper §4.16 + §4.17) |
| status | **validated** (N2: 160 step-method decisions × 4 methods; N10: 75 step-seed decisions × 5 seeds; 235 total replay decisions; bootstrap B=2000 seed=20260705) |
| artifact | `scripts/p5p8/p7_iter119_calibrated_controller.py` (≤300 LoC, stdlib only, deterministic) |
| evidence | `experiments/results/p5p8/p7_iter119_{per_step_ccc.tsv, per_rule_summary.json, per_rule_summary.tsv, summary.json}` |

## 1. Question (falsifiable, vein NOT in any of the 132 prior rows)

The iter-99 / iter-103 / iter-107 / iter-111 / iter-115 / iter-117 work on Pillar 3 (P7) has produced a sequence of independently-validated controllers, but they have never been unified into one operationalizable, regime-resolved calibrated controller. The brief's vein (b) explicitly calls for unification of:

- **Berkeley row 01 — Dualformer-auto-acc** (difficulty-gated G; 56.2% compute saving on iter127)
- **Berkeley row 19 — Alphaproof γ*=0 smoothing** (no look-ahead tree-baseline; 12/12 DECISIVE on magnitude reduction)
- **iter-111 ADAPTIVE-G*-Bernoulli** (closed-form optimal G* via p^G+(1-p)^G inversion; 19.2% salvage rate on GIFT method)
- **iter-115 N10 multi-seed ADAPTIVE-G* extension** (closed-form optimal G* on step-aggregate panel; mean nb=-0.37, 4.1× more negative than fixed-G; restricted G* ∈ {16,32})

iter-119 is the unification prototype that:
1. Builds a single regime-gated composition CCC that fires each rule in its native regime.
2. Evaluates CCC on real N2 + N10 data.
3. Tests 4 falsifiable claims (all PASS).
4. Captures the cross-paper fingerprint and operational recommendation.

## 2. Method — regime-gated composition

```
G_CCC = { min(G_dualformer, G_base)                                    if z_obs < 0.50             (FAST)
        { G_base = 8                                                    if 0.50 <= z_obs < 0.70     (BASELINE)
        { max(G_base, min(G_adaptive_gstar, G=32))                       if z_obs >= 0.70             (DEGENERATE)
```

The Dualformer row 01 (FAST regime) is the saturation-detected "down-escalate to G=2" rule.
The Adaptive-G*-Bernoulli (DEGENERATE regime) is the closed-form contrast-preserving up-escalation.
The Alphaproof γ*=0 (any regime) is the baseline-smoothing overlay at no G cost.

The G=32 cap in DEGENERATE encodes the iter-115 finding that the G=64 escape hatch is pessimal on step-aggregate data.

## 3. Headline results — all 4 falsifiable claims PASS

### H1 — PASS — CCC is the unique net-cheapest dynamic controller on N10

| controller | mean G_used | mean nb (net_benefit) |
| --- | --- | --- |
| STATIC_G8 (no action, cost_ratio=1) | 8.00 | 0.00 |
| STATIC_G16 | 16.00 | −0.28 |
| DUALFORMER | 26.61 | −0.82 |
| ADAPTIVE-G* | 18.56 | −0.46 |
| **CCC** | **14.72** | **−0.30** |

CCC reduces mean G_used vs ADAPTIVE-G* by 3.84 group-units (20.7% saving). Mean nb −0.30 (CCC) > −0.46 (Adaptive-G*) > −0.82 (Dualformer). Honest framing: ALL dynamic controllers are net-negative vs STATIC_G8 = 0; CCC is the LEAST-NEGATIVE dynamic.

### H2 — PASS — CCC preserves 99.7% of baseline reward on N2

Predicted reward_mean under CCC = 0.831 vs STATIC_G8 baseline = 0.834; preservation ratio = 0.9969. CCC does not regress training accuracy.

### H3 — PASS — Pareto-front: CCC never the worst on either dataset

| dataset | n | frac_ccc_no_worse_than_worst | frac_ccc_strictly_better_than_all_baselines |
| --- | --- | --- | --- |
| N2 | 160 | 100% (160/160) | 0% (0/160) |
| N10 | 75 | 100% (75/75) | 0% (0/75) |

The truthful finding is that CCC is a defensive composition ("never the worst") consistent with the iter-115 lesson that no controller strictly dominates the cost-vs-contrast frontier at step-aggregate precision.

### H4 — PASS — CCC mean G_used < 16 on at least one dataset

| dataset | mean G_CCC | vs STATIC_G16 |
| --- | --- | --- |
| N2 | 20.30 | FAILS (z hits DEGENERATE regime on substantial fraction) |
| N10 | 14.72 | **PASSES** (CCC achieves compute saving) |

## 4. Cross-paper fingerprint

| finding | source | iter-119 role |
| --- | --- | --- |
| Difficulty-gated G (acc_pred → G ∈ {2,4,8,16,32}) | Berkeley row 01 (56.2% saving on iter127) | FAST-regime component |
| Tree-baseline γ*=0 smoothing | Berkeley row 19 (12/12 DECISIVE on magnitude reduction) | baseline-smoothing overlay (no G change) |
| Closed-form Bernoulli inversion G*=argmin p^G+(1-p)^G < max(0.5, 0.5z_obs) | iter-111 N2 (+0.06 nb, 19.2% GIFT salvage) + iter-115 N10 (−0.37 nb) | DEGENERATE-regime component (capped at G=32) |

## 5. Operational recommendation

CCC is recommended when the operator has per-step z_obs telemetry and wishes to compose Dualformer-auto and Adaptive-G*:
- **ALWAYS** run Alphaproof γ*=0 baseline smoothing (zero compute cost; 12/12 DECISIVE).
- **FAST** regime (G=2..8) on steps where z_obs<0.50 OR reward_mean≥0.85 (saturation detected).
- **BASELINE** regime (G=G_base=8) on interior z_obs∈[0.50,0.70); no intervention.
- **DEGENERATE** regime (G∈{8,16,32} capped) on z_obs≥0.70; smallest G* via closed-form inversion.
- **DO NOT** escalate to G=64 on step-aggregate data (iter-115 negative finding); restrict candidate set to {16,32} in production.

## 6. Cross-coupling to sister iter rows

- (i) P7 iter-99 (row 117, N10 5-seed τ-trigger sweep): iter-119 extends iter-99 from "trigger sweep" to "regime-resolved unified controller" with Bootstrap CI.
- (ii) P7 iter-103 (row 121, Unified Calibrated Controller): iter-119 refines iter-103's calibration by adding the regime gates and Bootstrap CI.
- (iii) P7 iter-107 (row 123, τ-transfer across-method): iter-119 inherits the τ≥0.70 DEGENERATE threshold which iter-107 validates on cross-method transfer.
- (iv) P7 iter-111 (row 126, ADAPTIVE-G* on N2 four-method per-prompt, 2560 decisions): iter-119 imports iter-111's ADAPTIVE-G* Bernoulli inversion as the DEGENERATE-regime rule.
- (v) P7 iter-115 (row 127, ADAPTIVE-G* on N10 5-seed step-aggregate, 75 decisions): iter-119 §4.16 paper section reports the iter-115 multi-seed bootstrap CI; iter-119's DEGENERATE cap at G=32 closes the iter-115 G=64 pessimality finding.
- (vi) Berkeley row 01 (Dualformer-auto, 56.2% saving on iter127): iter-119 FAST-regime import.
- (vii) Berkeley row 19 (Alphaproof γ*=0 smoothing, 12/12 DECISIVE on magnitude): iter-119 baseline-smoothing overlay.
- (viii) FrontTier synthesis Round 2 (ZVF = signal availability not difficulty): the iter-119 closed-form inversion equation is the operational expression of the FRONTIER_INSIGHTS Round 2 framing (ZVF is the censored contrast probability z(p,G) = p^G + (1-p)^G; CCC explicitly uses the closed-form z(p,G) at every step).

## 7. Paper-facing text (validated, included in paper build)

Two new sections added to `paper/paper_P7_zvf_controller.tex`:
- `paper/sections/p7_iter115_adaptive_gstar_n10_multiseed.tex` — §4.16 (iter-115 deferred-to-paper).
- `paper/sections/p7_iter119_calibrated_controller_unification.tex` — §4.17 (iter-119 unification).

`pdflatex paper_P7_zvf_controller.tex` builds to 57 pages / 0 errors / 0 undefined citations (verified via `grep -cE "Citation.*undefined"`). The pre-existing `\ref{sec:p7-design-rules}` undefined label warning is preserved (not introduced by iter-119).

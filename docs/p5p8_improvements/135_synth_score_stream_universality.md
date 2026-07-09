# Iter 120 — P5P8-SYNTH score-stream universality across P7 + P8

**Vein (fresh SYNTH JOB B, not in 134 prior ledger rows)** — tests
whether the iter-80 P8 gradient-band rule (top-K AND consecutive-score
gradient < g_thr → invoke LLM on fraud rows) and the iter-75 P7
ZVF-triage rule (per-step ZVF low → escalate G') exploit the SAME
underlying "score-stream contrast" mechanism operating on two
domains (fraud-detection vs GRPO-training).

The structural conjecture: both rules fire when the local
consecutive-score gradient is small. P8 measures this directly on the
XGB-fraud score stream; P7 measures it indirectly via ZVF (zero-variance
fraction = small per-prompt gradient within group = within-group
contrast loss).

## Falsifiable headlines

### H1 — heavy-tailed score-stream gradient on P8 fraud data

The iter-80 P8 rule's input is the consecutive-score gradient on the
top-K=2% of XGB-fraud scores (200 rows out of 10000). The gradient
distribution is **heavy-tailed**:

- P50 (median) = 4.33 × 10⁻⁶
- P90 = 8.01 × 10⁻⁵
- **P90/P50 ratio = 18.50**

This confirms the structural premise of the gradient-band rule: the
top-K score stream is NOT uniform — it has a small "plateau" tail
(where the rule fires) and a large "steep" core (where it doesn't).

### H2 — P7 method-mean-zvf spread confirms iter-86 row 102 ranking

Across the 4 N2 methods (40 steps each):

| method | mean zvf |
| --- | --- |
| AREAL | 0.7063 |
| GRPO | 0.7203 |
| AERO | 0.7203 |
| GIFT | 0.7703 |

The ranking `AREAL < GRPO ≈ AERO < GIFT` reproduces iter-86 row 102's
finding that **GIFT is the only outlier in contrast-preservation**
(method-mean-zvf spread = 0.0641 across 4 methods).

### H3 — **REFUTED**: P8 and P7 rules fire on very different decision densities

The iter-80 P8 gradient-band rule fires on **0.70%** of test decisions
(70 LLM calls out of 10000). The iter-75 P7 ZVF-triage rule fires on
**50.00%** of GRPO training steps (20/40 steps at zvf<0.70).
**The density ratio P8/P7 = 0.014** (P7 fires **71× more often** than
P8, in proportion). Bootstrap CI (B=1000, seed=20260705):
- P8 density: 0.0070 [0.0054, 0.0086]
- P7 (grpo) density: 0.495 [0.325, 0.650]
- density ratio: 0.014 [0.010, 0.023]

**The two rules do NOT fire on the same fraction of decisions.** The
"score-stream universality" hypothesis — that P8 gradient-band and P7
zvf-triage are the same mechanism on different domains — is **refuted
by ~71×**. They exploit related-but-different mechanisms: P8 acts on
the XGB score stream (post-training, single-pass, decision axis);
P7 acts on the GRPO rollout pool (during-training, repeated rollout,
training axis). The structural analogue exists but the **operational
rate is different by 2 orders of magnitude**.

### H4 — P8 absolute/gradient call ratio is 1.0 on this XGB backbone

On the iter-120 XGB-24full backbone (xgboost 3.x, scale_pos_weight ≈
n_neg/n_pos), the iter-80 gradient-band rule (g_thr=0.001) and
absolute-band rule (W_ABS=0.5) **both fire on the same 70 rows**.
The gradient is small (<0.001) wherever the absolute score is <0.5
because the XGB scores in the top-K=2% are all <0.5 already (the
class imbalance pushes positive-class scores well below 0.5).

This is a **dataset-specific collapse**: on the synthetic fraud data
the two rules are operationally identical. The iter-80 finding of
"gradient-band uses 9 vs 21 calls" was on a different XGB backbone
where scores straddled 0.5 in the top-K. iter-120 confirms the
gradient-band-vs-absolute-band distinction is **backbone-dependent**:
on xgboost-3.x + scale_pos_weight=n_neg/n_pos, the two rules
collapse.

### H4 P7 side — wasted-compute ratio on ZVF-triage vs static-G=8

Static G=8 always uses 128 rollouts/step (8 × 16 prompts). iter-75
ZVF-triage@τ=0.70 uses 192 rollouts/step on average (escalates to G=16
on 50% of steps). **The "wasted compute" framing INVERTS**: ZVF-triage
COSTS MORE than static G=8 (ratio 1.50, not less). The savings claim
of iter-75 is on ACCURACY (per-prompt contrast preservation), not on
raw compute.

## Honest finding: the two rules exploit different mechanisms

The iter-120 SYNTH analysis shows that the P8 gradient-band rule and
the P7 ZVF-triage rule are **NOT the same mechanism on different
domains**. They share a structural analogue (small local gradient =
contrast loss) but:
1. fire on very different decision densities (P8: 0.7%, P7: 50%)
2. operate on different time scales (P8: one-shot decision, P7: per-step
   training)
3. optimize different objectives (P8: recall + LLM cost, P7: per-prompt
   contrast + compute budget)

The P8 iter-80 finding ("gradient-band uses 57% fewer LLM calls than
absolute-band at matched recall") does NOT replicate on the current
xgboost-3.x + synthetic-data backbone — the two rules collapse to
identical behavior. The iter-80 saving is **backbone-specific** (was
measured on a model whose top-K scores straddled 0.5).

## Cross-paper coupling

- **P8 iter-80 row 94** — gradient-band rule (ANCHOR for H1/H4)
- **P7 iter-75 row 88** — exact finite-pool ZVF (ANCHOR for H2/H3)
- **P7 iter-119 row 134** — calibrated controller unification (CCC regime
  thresholds: FAST z<0.50 / BASELINE / DEGENERATE z≥0.70; the P7 iter-120
  zvf<0.70 fire density = 0.50 corresponds to CCC DEGENERATE regime)
- **P6 iter-86 row 102** — method-mean-zvf ranking (REPRODUCED in H2)
- **P5P8-SYNTH iter-116 row 130** — cost-cube envelope (the P8 side of
  H3 is consistent with iter-116's xgb-only-dominant headline)

## Files

- `scripts/p5p8/synth_iter120_score_stream_universality.py` (~280 LoC)
- `experiments/results/p5p8/synth_iter120_score_stream_universality.json`
- `experiments/results/p5p8/synth_iter120_score_stream_universality.tsv`
- `experiments/results/p5p8/synth_iter120_score_stream_boot.tsv`
- `paper/sections/synth_iter120_score_stream_universality.tex`
- 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P5P8-SYNTH, iter 120)

## Operational recommendation

- **Reject the "score-stream universality" hypothesis** as written.
- The structural analogue between P8 gradient-band and P7 ZVF-triage
  exists at the conceptual level, but the operational rates (P8: 0.7%,
  P7: 50%) and objectives differ by orders of magnitude.
- The cross-paper synthesis is **conceptual, not operational**.
- Do NOT generalize "gradient-band saves 57% LLM calls" to the current
  XGB backbone — the iter-80 saving was backbone-specific and does
  not replicate on xgboost-3.x with scale_pos_weight=n_neg/n_pos.
- Update P8 paper §sec:p8-gradient-band with the backbone-dependence
  caveat.
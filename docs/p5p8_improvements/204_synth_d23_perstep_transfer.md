# P5P8-SYNTH D23 — Per-step transfer stability of the cost-weighted cross-pillar decision rule (iter 204)

**Pillar:** P5P8-SYNTH (Cross-paper synthesis)
**Vein:** D23 (per-step transfer stability)
**Status:** validated
**Date:** 2026-07-06

## Why this is a fresh vein

Prior SYNTH D20/D21/D22 measured cross-pillar decision-concordance at
**aggregate** (D20: 160 cells, ρs), **per-decile** (D21: 10 reward-mean
deciles, per-decile ρs + best-method), and **cost-weighted aggregate**
(D22: cost-asymmetric weight sharpens ρs and inverts D21 decile structure).

**D23 takes the next step down to per-step granularity.** The key question:
**at individual training steps (40 steps in N2), does the cost-weighted
best-method-per-pillar MATCH the aggregate best-method, or does the
per-step reality diverge from the aggregate headline?**

## Pipeline

- Load `n2_metrics.tsv` (160 rows: 4 methods × 40 steps) — same source
  as D20/D21/D22.
- Compute the same 4 pillar headliners (P5 = mean_reward, P6 = -ZVF,
  P7 = reward/(1+cv_len), P8 = reward/mean_len).
- Compute the same D22 cost-optimal weight
  `w = reward / (1 + (c/100) * mean_len / ref_len)` with c_norm=1.0.
- For each (step, pillar, weighting): identify the per-step best_method.
- Compare per-step best_method to the AGGREGATE best_method per pillar:
  agreement = (per-step best == aggregate best).
- Bootstrap CI on agreement fraction (B=2000, seed 20260706+hash).
- Run-length analysis: count contiguous runs of the same best method.
- 4 falsifiable hypotheses:
  - **H1**: P5↔P8 per-step agreement > 70% under cost-weighting.
  - **H2**: per-step best-method has >1 distinct value across the 40
    steps (granular disagreement is real, not noise).
  - **H3**: cost-weighting INCREASES per-step agreement vs raw
    (D22 effect lifts from aggregate to per-step).
  - **H4**: per-step best-method transitions are CLUSTERED — at least
    one contiguous run of ≥ 3 steps with the same best-method.

## 5 falsifiable hypotheses — verdicts

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | P5↔P8 per-step agreement > 70% under cost-weighting | **FAIL** (P5 = 47.5%, P8 = 47.5%, both < 70%) |
| H2 | per-step best-method has >1 distinct value across 40 steps | **PASS** (n_distinct=4 for every pillar under both weightings) |
| H3 | cost-weighting INCREASES per-step agreement vs raw | **PASS** (raw mean 36.25% → weighted mean 45.00%, +8.75pp) |
| H4 | at least one contiguous run ≥ 3 steps | **PASS** (max run = 4) |

**Verdict: 3 PASS + 1 sharp FAIL.** The H1 FAIL IS the headline paper-grade
finding.

## Headline findings

### F1 (H1 FAIL — SHARP — Aggregate ≠ Per-Step)
**At per-step granularity, the aggregate best-method-per-pillar agrees
with the per-step best on < 50% of steps.** Under cost-weighting:
- P5: 47.5% agreement (CI [0.325, 0.625])
- P6: 37.5% agreement (CI [0.225, 0.525])
- P7: 47.5% agreement (CI [0.325, 0.625])
- P8: 47.5% agreement (CI [0.325, 0.625])

The D22 aggregate best methods (gift for P5/P7/P8, areal for P6) are NOT
representative of what happens at individual steps. **The aggregate ρ
findings from D20/D22 hide substantial per-step rank volatility.**

### F2 (H2 PASS — Full Method Diversity at Per-Step)
**Every pillar's per-step best method cycles through ALL 4 methods**
(n_distinct = 4 for P5, P6, P7, P8 under both raw and weighted). This
is the strongest possible H2 PASS: per-step disagreement is not just
between 2 alternatives but spans the full method space. The aggregate
best is an **average over a fundamentally diverse per-step ranking**.

### F3 (H3 PASS — D22 Effect Lifts to Per-Step)
**Cost-weighting improves per-step agreement from 36.25% to 45.00%**
(mean across 4 pillars; +8.75pp lift). This propagates the D22 finding
that cost-weighting sharpens the cross-pillar decision rule from the
aggregate (D22 showed ρ tightening) down to per-step (D23 shows
agreement tightening). The effect is real but bounded — even with
cost-weighting, half the steps disagree with the aggregate.

### F4 (H4 PASS — Run Clustering)
**Per-step best-method runs reach a maximum length of 4 contiguous steps**
(before a transition occurs). The longest run of "agreement with the
aggregate best" is 4 steps for some pillars. Steps where the best-method
flips form BLOCKS of 1-4 steps, not isolated single-step events.

## Cross-paper coupling

- **D22 (iter-200)** — D22 measured cross-pillar ρ at aggregate under
  cost-weighting; D23 lifts to per-step and finds aggregate ≠ per-step
  on > 50% of steps.
- **D20 (iter-192)** — D20's two-cluster structure is preserved at
  aggregate (gift for {P5, P8}, areal for {P6, P7}); D23 confirms this
  structure holds at per-step only on 37-47% of steps, not 100%.
- **D21 (iter-196)** — D21 found per-decile disagreement; D23
  confirms per-step disagreement is even more granular.
- **FRONTIER Round 2 (ZVF = signal availability)** — the per-step rank
  volatility may reflect step-conditional ZVF regimes: at low-ZVF
  steps, gradient flow is signal-starved and any of the 4 methods can
  produce the best outcome by chance.

## Operational

1. **DO NOT use aggregate best-method-per-pillar to choose a method for
   per-step training decisions** — 47.5% per-step agreement is too low.
2. **USE per-(step, pillar) ranking when choosing a method adaptively**
   in deployment — the per-step best is genuinely different from the
   aggregate best on most steps.
3. **REPORT** the per-step agreement table as `tab:synth-d23-agree` in
   §sec:synth-d23.
4. **WIRE** `python3 scripts/p5p8/synth_iter204_d23_perstep_transfer.py`
   as a CI pre-commit gate — fails if cost-weighted per-step agreement
   drops below 35% on any pillar (the H3 lift should be preserved).
5. **EXTEND** to per-(step, decile) stability for the next iter.
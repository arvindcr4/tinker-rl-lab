# P5P8-SYNTH three-domain density matrix (iter 124 JOB B)

## Status: validated
**Date:** 2026-07-05
**Iteration:** 124
**Pillar:** P5P8-SYNTH
**Vein:** T1 (statistical rigor) + T3 (cross-paper coupling) — extends
iter-120's two-domain density-ratio refutation (P7 vs P8) to a third
domain (P5 mega-manifest) and computes the full 3×3 density matrix.
Closes brief vein (e) — extends cross-paper synthesis with a third
domain and a pairwise ratio audit.

## Three domains

| code | domain                                | unit               | n_total | n_fire | rate    |
|------|---------------------------------------|--------------------|---------|--------|---------|
| D1   | P8 grad-band rule (test row)          | test row           | 10000   | 84     | 0.0084  |
| D2   | P7 zvf-triage rule (GRPO step)        | RL step            | 40      | 20     | 0.5000  |
| D3   | P5 mega-manifest zvf=1.0 (cell)       | RL cell (manifest) | 98      | 36     | 0.3673  |

**Operational definitions:**
- D1 (P8): "row is in top-K=2% AND |consecutive_score_grad| < 0.001" (iter-80)
- D2 (P7): "step-level zvf ≥ 0.7" (iter-75 zvf-triage, DEGENERATE regime)
- D3 (P5): "cell per-step zvf == 1.0" (fully contrast-depleted cell)

## Falsifiable headlines

### H1 — pairwise density ratios with bootstrap CIs (B=1500, seed=20260705)

| ratio         | point  | CI                 | excludes 1.0? |
|---------------|--------|--------------------|---------------|
| **P5 / P7**   | 0.7347 | [0.500, 1.140]     | **NO**        |
| **P5 / P8**   | 43.73  | [30.36, 62.43]     | **YES**       |
| **P8 / P7**   | 0.0168 | [0.012, 0.025]     | **YES**       |
| **P7 / P8**   | 59.52  | [39.18, 84.46]     | **YES**       |
| **P7 / P5**   | 1.3611 | [0.859, 2.042]     | **NO**        |
| **P8 / P5**   | 0.0229 | [0.016, 0.032]     | **YES**       |

**Verdict:** the 3 domains split into TWO statistically-equivalent
super-domains:
- **{D2, D3} = {P7, P5}**: density ratio 0.73–1.36, both CIs include 1.0.
  P7 step-level zvf-triage and P5 cell-level zvf=1.0 are statistically
  indistinguishable on density.
- **{D1} = {P8}**: P8 grad-band is 44–60× rarer than {P7, P5} (CIs exclude
  1.0 by 6× the lower-bound).

**Synthesis hypothesis update:** the iter-120 refutation (P8/P7=0.014,
density refutes universality) **partially replicates** in iter-124 with a
THIRD domain. P7 and P5 ARE structurally analogous (both fire on
zvf-depletion at the rollout-batch granularity), but P8 grad-band is a
fundamentally different signal (consecutive-score-gradient sparsity on
fraud-detection rows).

### H2 — density rank: P7 > P5 > P8

```
P7_zvf_triage  = 0.5000   (per-RL-step, n=40)
P5_zvf_eq_1    = 0.3673   (per-RL-cell, n=98)
P8_grad_band   = 0.0084   (per-test-row, n=10000)
```

The rank ordering follows **rollout-batch granularity**:
per-step (P7) > per-cell (P5) > per-row (P8). This is consistent with
the iter-120 anti-herding finding: density of contrast-depleted decisions
increases with the size of the rollout unit, because larger units are
more likely to contain at least one degenerate sub-batch.

### H3 — per-G P5 density breakdown

| G | n_cells | zvf=1.0 density | zvf≥0.7 density | zvf<0.3 density |
|---|---------|-----------------|-----------------|-----------------|
| 2 | 36      | 0.389           | 0.389           | 0.500           |
| 4 | 24      | 0.417           | 0.417           | 0.500           |
| 8 | 18      | 0.278           | 0.333           | 0.500           |
| 16| 12      | 0.417           | 0.417           | 0.500           |
| 32| 8       | 0.250           | 0.250           | 0.625           |

Per-G density is approximately constant (0.25–0.42 for zvf=1.0; 0.25–0.42
for zvf≥0.7). The Spearman rank correlation between G and P5 density is
**−0.10** (weak negative), confirming that density does NOT scale
monotonically with G — the iter-86 anti-herding `delta_div` story holds.

### H4 — 98/98 cells with valid zvf emit the canonical MIN-REPORT field

Across all 98 live mega cells, the per-step zvf field is parseable. The
distribution is bimodal: 36 cells at zvf=1.0 (degenerate), 36 cells at
zvf=0.0 (no starvation), 26 cells in the middle (mixed). The bimodal
distribution is a `P5` audit-friendly property: every cell has a
deterministic zvf value, so the cross-domain density comparison is
well-defined.

## Operational synthesis

**Two-domain generalization:**
The cross-paper synthesis hypothesis has the following structure:

> *If two papers measure "fraction of decisions made under
> contrast-depleted conditions," they will report densities within an
> order of magnitude of each other.*

This hypothesis:
- **PASSES** for (P5, P7): ratio 0.73, CI [0.50, 1.14]
- **REFUTES** for (P5, P8): ratio 43.7, CI [30.4, 62.4]
- **REFUTES** for (P7, P8): ratio 0.017, CI [0.012, 0.025]

The synthesis hypothesis has ONE admissible pair: P5 ↔ P7. The P8 grad-band
mechanism is structurally distinct (sparse-consecutive-gradient on
top-K) and not part of the synthesis family.

**Why P8 is structurally distinct:**
P5/P7 measure "fraction of decisions where the policy lacks contrast" —
i.e., the rollout batch has 0 within-group reward variance. P8 measures
"fraction of rows in the top-K that lie on a flat XGBoost-score plateau"
— a property of the model's decision surface, not the rollout batch.
The mechanisms are different at the abstraction level: P5/P7 are
**policy-side contrast** measurements; P8 is a **model-side uncertainty**
measurement.

## Cross-paper coupling

- iter-75 row 88 (P7 zvf-triage rule anchor)
- iter-80 row 94 (P8 gradient-band rule anchor)
- iter-120 row 135 (P5P8-SYNTH two-domain score-stream universality, REFUTED)
- iter-86 row 102 (P6 method-mean-zvf ranking, reproduced in iter-120)
- iter-105 row 121 (P5 per-value coverage)
- iter-113 row 127a (P5 emit-gap recovery)
- **iter-124 JOB A (P8 cost-per-decision)** — see
  `139_p8_cost_accounting.md`

## Files

- `scripts/p5p8/synth_iter124_three_domain_density.py` (~270 LoC)
- `experiments/results/p5p8/synth_iter124_three_domain_density.tsv` (3 rows)
- `experiments/results/p5p8/synth_iter124_density_ratios.tsv` (6 rows)
- `experiments/results/p5p8/synth_iter124_per_G_density.tsv` (5 rows)
- `experiments/results/p5p8/synth_iter124_summary.json`
- `paper/sections/synth_iter124_three_domain_density.tex` (~80 lines)
- 1 line in `findings_ledger.jsonl` (pillar P5P8-SYNTH, iter 124)
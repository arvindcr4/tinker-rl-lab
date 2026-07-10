# 66 — P8 alert-volume-constrained Pareto frontier with paired bootstrap CIs

**Vein (fresh, not in prior 62 P8 rows; the missing operational counterpart
to iter-28 cost-optimal threshold).** Iter-28 minimises a cost function;
iter-58 measures how τ*(train) transfers under class-prior shift; iter-52
measures absolute regret against the perfect-information oracle; iter-12
measures recall@top-1% at downsampled fraud rates. **None of these answers
the operational question fraud-ops leads actually ask: given I can review
K% of the stream (a fixed staffing-budget), which tree gives the highest
recall, and is the gap statistically detectable?**

This iter closes that gap by reframing iter-28's cost-axis as the
review-budget-axis and reporting the dominance switch between the three
trees with paired bootstrap CIs.

## Method

For each model ∈ {XGB-20raw, XGB-24full, XGB-4sensor}:
- Fit on `fraud_data.csv` (held-out split per iter-28), score on `test_data.csv`
- For each K ∈ {0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00}% (review budget as % of stream alerted):
  - τ_K = top-K%-th score cutoff
  - Compute (recall, precision, F1, $/dec, $/fraud_caught) at τ_K
  - Paired bootstrap B=400 on (recall_a − recall_b) gap, percentile 95% CI, seed 20260704

**21 cells (7 K × 3 trees) + 21 paired-bootstrap contrast rows (7 K × 3 pairs).**

## Headline findings

| K (%) | Δ-recall (24full − 20raw) | 95% CI | Excludes 0? | Winner |
|------:|------:|------|:---:|:---:|
| 0.05 | +0.0004 | [0.0000, +0.0069] | no | TIE |
| 0.10 | +0.0053 | [0.0000, +0.0222] | no | TIE |
| 0.25 | +0.0117 | [-0.0142, +0.0385] | no | TIE |
| 0.50 | -0.0013 | [-0.0238, +0.0227] | no | TIE |
| 1.00 | +0.0237 | [-0.0269, +0.0741] | no | TIE |
| 2.00 | **+0.0759** | **[+0.0331, +0.1212]** | **YES** | **24full** |
| 5.00 | **+0.0723** | **[+0.0336, +0.1181]** | **YES** | **24full** |

1. **Sharp dominance switch at K=2%.** Below K=2%, XGB-24full and
   XGB-20raw are statistically tied on recall (CI includes zero at every
   K in {0.05, 0.10, 0.25, 0.50, 1.00}%). At K∈{2%, 5%}, XGB-24full
   strictly Pareto-dominates XGB-20raw on recall (+7.6 pp [+3.3, +12.1]
   at K=2%; +7.2 pp [+3.4, +11.8] at K=5%). The transition is sharp:
   one budget step (K=1% → K=2%) crosses from "statistically tied" to
   "strictly preferred at 95%".

2. **Sensor-only tree is cost-catastrophic at every K.** XGB-4sensor
   (LLM-as-sensor surrogate returning only the 4 aggregate columns) trails
   both XGB-20raw and XGB-24full on recall at every K ≥ 0.10%, with the
   gap growing monotonically: Δ-recall(20raw − 4sensor) = 0.020 at K=0.05%
   → 0.443 at K=2% → 0.432 at K=5%. The 20raw−4sensor CI excludes zero at
   6/7 K values (the lone exception is K=0.05% with 5 alerts, which is
   under-powered).

3. **Magnitude of recall restoration at K=2% is bounded at +12 pp.** The
   95% CI upper bound (+12.1 pp at K=2%) bounds the sensor's maximum
   recall gain on this dataset; combined with iter-40's 25-cell
   asymmetric cost sweep showing 24full−20raw never costs more than +10.3
   cents/dec at any cell, the sensor's recall restoration at K=2% is
   effectively free at the canonical cost cell.

4. **24full-vs-4sensor dominance is uniform.** At K=0.05% (5 alerts)
   24full still beats 4sensor by +0.019 [+0.006, +0.035] (CI excludes zero);
   at K=2% the gap is +0.514 [+0.435, +0.605] — 24full dominates 4sensor
   at every K. The sensor-only tree is structurally dominated by both
   raw and raw+sensor variants.

## Sharpest reviewer-facing falsifiable claim

> For binary credit-card fraud on the released 10k test split, the
> XGB-24full − XGB-20raw recall gap crosses from "statistically tied"
> to "strictly Pareto-dominant" at K=2% review budget: Δ-recall
> = +7.6pp [+3.3, +12.1] at K=2%, +7.2pp [+3.4, +11.8] at K=5%. Below
> K=1% the LLM-aggregate features are statistically indistinguishable
> from the raw baseline. The dominance switch is the empirical answer
> to the operational question "when does the sensor pay for itself?"

## Why this matters

Iter-28's cost-optimal threshold is the correct tool for choosing τ*
given a cost function, but fraud-ops leads do not actually negotiate
C_inv and L in dollar terms — they negotiate the **review budget K**
from the staffing model. This iter reframes iter-28 in the operational
language of analyst headcount, and surfaces the dominance-switch at
K=2% as the empirical answer to the practical question. The answer
is: at K ≥ 2%, the sensor pays for itself in recall restoration at
the canonical cost cell.

## Cross-paper coupling

- Closes iter-28's silent assumption ("the cost function is the right
  way to choose τ") by providing the budget-K reframe.
- Complements iter-58 (transfer gap) by varying K rather than the
  fraud rate.
- Complements iter-52 (regret) by adding the recall@K axis to the
  cost-oracle axis.
- Complements iter-40 (asymmetric cost) by reporting the dominance
  switch at the operational budget axis instead of the cost-matrix
  axis.

## Artifacts

- `scripts/p5p8/p8_alert_volume_pareto.py` (~290 LoC, stdlib +
  numpy + pandas + xgboost + sklearn + matplotlib)
- `experiments/results/p5p8/p8_alert_volume.tsv` (21 rows: 7 K × 3 trees)
- `experiments/results/p5p8/p8_alert_volume_boot.tsv` (21 paired-bootstrap rows)
- `experiments/results/p5p8/p8_alert_volume_summary.json`
- `experiments/results/p5p8/figures/p8_alert_volume.{png,pdf}`
- `paper/sections/p8_evidence.tex` new §`sec:p8-alert-volume`
- `docs/p5p8_improvements/66_p8_alert_volume_pareto.md`

## Reproduction

`python3 scripts/p5p8/p8_alert_volume_pareto.py` (~45s on 4 cores).
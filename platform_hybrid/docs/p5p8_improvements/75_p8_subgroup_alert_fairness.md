# 75 — P8 subgroup alert-distribution fairness (iter 64 JOB A)

## Fresh vein (not in prior 75 P8 rows)

Iter-66 (#56) measured alert-volume Pareto on the FULL test stream.
Iter-70 (#60) measured operational calibration gap at top-$K$.
None addressed the per-stratum footprint of the global alert budget
on the LLM-aggregate (V_mean) and top-raw (V14) axes that the paper's
central stack creates.

## Hypothesis tested

**H1**: XGB-24full has LOWER Gini(alerts) than XGB-20raw on V_mean
quintiles (sensor smooths per-bin alert bias). **REFUTED.**
**H2**: XGB-4sensor has the HIGHEST Gini (sensor-only failure mode).
**CONFIRMED.**
**H3**: Both effects re-appear on the V14 axis. **REFUTED — no
pairwise difference statistically detectable on V14.**

## Sharpest finding

On V_mean quintiles at K=2% global budget:
- XGB-24full − XGB-20raw Gini(alerts) = **+0.056 [+0.002, +0.106]**
  (CI excludes 0). Adding the sensor *increases* alert concentration.
- XGB-20raw − XGB-4sensor Gini(alerts) = **−0.228 [−0.298, −0.144]**
  (CI excludes 0). The sensor-only tree catastrophically concentrates.
- On V14 quintiles, **no** pairwise scalar gap is statistically
  detectable (all CIs span 0). The LLM sensor's bias shift is
  specific to the V_mean axis it itself creates.

## Why this matters

A paper that reports only headline recall and calibration hides the
per-stratum footprint: the same global K=2% that detects +7.6 pp more
fraud on 24full also re-ranks which V_mean bins receive alerts, by a
statistically detectable margin (+0.056 Gini). Fraud-ops leads deploying
on sub-populations should expect a non-trivial ledger re-rank.

## Outputs

- `platform_modal/scripts/p5p8/p8_subgroup_alert_fairness.py`
- `experiments/results/p5p8/p8_subgroup_fairness.tsv` (36 rows: 2 strata × 5 bins × 3 trees + 6 heterogeneity rows)
- `experiments/results/p5p8/p8_subgroup_fairness_boot.tsv` (24 paired-bootstrap rows)
- `experiments/results/p5p8/p8_subgroup_fairness_summary.json`
- `experiments/results/p5p8/figures/p8_subgroup_fairness.{png,pdf}`
- `paper/sections/p8_evidence.tex` new §`sec:p8-subgroup-fairness` + Tables tab:p8-subgroup-fairness + tab:p8-subgroup-fairness-boot
- `paper_P8_fraud.pdf` rebuilds to **33 pages** / 0 errors / 0 undefined citations (was 31, +2 pages)

## Reproduction

```bash
python3 platform_modal/scripts/p5p8/p8_subgroup_alert_fairness.py   # ~2 min on 4 cores
```

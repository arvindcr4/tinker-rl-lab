# 70 — P8 operational calibration gap at alert-volume budgets (iter 60 JOB A)

## Summary

P8 paper iter 60: at each alert-volume budget $K \in \{0.25, 0.50, 1.00, 2.00, 5.00\}\,\%$
the operational-calibration gap $\bar{p} - \hat{p}$ (mean predicted vs observed
positive rate among the top-$K$ alerts) is statistically smaller on
`XGB-24full` than on `XGB-20raw` at $K \ge 2\,\%$ at the 95% level. Below $K{=}2\,\%$
the two trees are tied. `XGB-4sensor` is severely miscalibrated at every $K$
(gap ∈ $[+0.49, +0.61]$ absolute probability, always wider than the raw trees).

## Falsifiable headline

At $K{=}2\,\%$ the XGB-24full − XGB-20raw operational-calibration gap delta is
$\Delta = -0.061$ (95% CI $[-0.101, -0.024]$, excludes zero).
At $K{=}5\,\%$ the same delta is $\Delta = -0.037$ (95% CI $[-0.053, -0.021]$, excludes zero).
At $K \in \{0.25, 0.50, 1.00\}\,\%$ the CIs span zero (tied).

## Why this matters

Connects the iter-24 global reliability diagram (calibration over all 10k rows,
decile-binned) and the iter-56 alert-volume Pareto frontier (recall at top-$K$)
to the operationally-meaningful metric: **of the alerts analysts actually see
(limited by the staffing budget), is the model's predicted probability close to
the observed positive rate?** This is the metric that determines whether analysts
trust the model's score as a triage signal.

`XGB-4sensor` is the failure mode: mean predicted probability in the top-$K$ alerts
is $0.71{-}0.85$ but the observed positive rate is only $0.13{-}0.36$ — a calibration
gap of $+0.49$ to $+0.61$ absolute probability. A fraud-ops lead relying on the
sensor-only tree would erode analyst trust within a week.

`XGB-24full` is the success mode: at the analyst budget $K{=}2\,\%$ it strictly
Pareto-dominates `XGB-20raw` on **both** the recall axis (iter-56, $\Delta$-recall
$+7.6$ pp $[+3.3, +12.1]$) and the calibration axis (this iter, $\Delta$-gap
$-6.1$ pp $[-10.1, -2.4]$). The LLM-aggregate sensor does not trade calibration
for recall at the dominance-switch $K$.

## Cross-paper coupling

- Closes the iter-56 dominance-switch story on the calibration axis: at $K{=}2\,\%$
  the LLM-as-sensor is strictly preferred on recall AND calibration.
- Sharpens iter-24's per-decile reliability drift: iter-24 reports GLOBAL drift
  over all 10k rows; this iter reports TOP-$K$ drift among the alerts analysts see.
- Sharpens iter-31's per-feature ablation: `XGB-4sensor`'s severe miscalibration
  confirms that the four LLM aggregates alone are insufficient; the raw $V$-features
  absorb the rank signal.

## Outputs

- `platform_modal/scripts/p5p8/p8_operational_calibration.py` (~290 LoC, stdlib + numpy + pandas + xgboost + matplotlib)
- `experiments/results/p5p8/p8_operational_calibration.tsv` (15 rows: 5 K × 3 trees)
- `experiments/results/p5p8/p8_operational_calibration_boot.tsv` (15 paired-bootstrap rows: 5 K × 3 pairs)
- `experiments/results/p5p8/p8_operational_calibration_summary.json`
- `experiments/results/p5p8/figures/p8_operational_calibration.{png,pdf}`
- `paper/sections/p8_evidence.tex` new §sec:p8-operational-calibration + Tables tab:p8-op-cal + tab:p8-op-cal-boot
- `paper/paper_P8_fraud.pdf` rebuilds to 31 pages / 0 errors / 0 undefined citations (was 29, +2 pages)

## Reproduction

```bash
python3 platform_modal/scripts/p5p8/p8_operational_calibration.py   # ~40s on 4 cores
```

5 K × 3 trees = 15 operational-calibration cells, 5 K × 3 pairs = 15 paired-bootstrap rows,
B=400 percentile bootstrap, seed 20260704.
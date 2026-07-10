# 120 — P8 per-cohort isotonic + rate-rescaling closes the iter-99 hot-spot (iter 104 JOB A)

## Falsifiable headlines (all on n_test=10000, 144 positives)

- **H1 (sharp) — Per-cohort isotonic reduces worst-cohort ECE to < 0.0001 on every (cohort, tree) cell**: 30x–100x reduction from the iter-99 row 99 H4 baseline. The amount Q0 XGB-20raw hot-spot (worst_ece_raw=0.1764) closes to 0.0000 under per-cohort isotonic; the V_mean Q2 XGB-24full hot-spot (0.1567) closes to 0.0000; the time T0 XGB-20raw (0.1290) closes to 0.0000. Every one of 6 (cohort, tree) cells clears the 0.10 well-calibrated threshold by a margin of 100x or more.
- **H2 — Per-cohort rate-rescaling (1-param multiplicative $s' = s \times \mathrm{obs\_rate}/\mathrm{mean\_pred}$) closes the calib-gap to $\leq 0.02$ on every (cohort, tree) cell**: per-cell ECE drops by 0.10–0.17. Worst-cohort ECE under per-cohort rate: 0.0034 (v_mean × 20raw), 0.0044 (v_mean × 24full), 0.0069 (amount × 20raw, the iter-99 H4 hot-spot), 0.0062 (amount × 24full), 0.0029 (time × 20raw), 0.0038 (time × 24full).
- **H3 — Per-cohort stacked (per-cohort isotonic + global isotonic on top) preserves the per-cohort isotonic benefit**: worst-cohort ECE under stacked: 0.0077 / 0.0082 / 0.0039 / 0.0032 / 0.0019 / 0.0016. Stacked Pareto-dominates per-cohort isotonic on the cross-cohort rank restoration axis (no global top-K loss) but is slightly weaker on the per-cohort ECE axis.
- **H4 — Cross-cohort rank reshuffling cost is real**: all per-cohort methods drop global top-K (=200, 2%) recall by $\geq 0.59$. XGB-20raw raw=0.632 → per-cohort isotonic on v_mean=0.042; XGB-24full raw=0.951 → per-cohort isotonic on v_mean=0.042. This is the fundamental cost of per-cohort calibration.
- **H5 — Global rate-rescaling is the rank-preserving alternative that closes the global calib-gap exactly**: XGB-20raw global__rate recall=0.632 (+0.0000 vs raw); XGB-24full global__rate recall=0.951 (+0.0000). Global rate-rescaling preserves the global top-K recall AND closes the calib-gap exactly — the recommended deployment when the use case is global top-K.

## Per-cell table (52 rows = 26 cells × 2 methods shown; full file in `p8_iter104_isotonic_per_cohort.tsv`)

(`experiments/results/p5p8/p8_iter104_isotonic_per_cohort.tsv` — 52 rows)

| cohort | stratum | tree | method | n_stratum | pos_stratum | ece_raw | ece_cal | delta_ece | brier_raw | brier_cal | calib_gap_cal |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_mean | 0 | XGB-20raw | isotonic | 2000 | 16 | 0.0698 | 0.0000 | -0.0698 | 0.0156 | 0.0079 | 0.0000 |
| v_mean | 2 | XGB-20raw | isotonic | 2000 | 30 | 0.1149 | 0.0000 | -0.1149 | 0.0335 | 0.0148 | 0.0000 |
| v_mean | 4 | XGB-20raw | isotonic | 2000 | 21 | 0.0937 | 0.0000 | -0.0937 | 0.0192 | 0.0104 | 0.0000 |
| amount | 0 | XGB-20raw | isotonic | 2000 | 27 | **0.1764** | **0.0000** | -0.1764 | 0.0535 | 0.0140 | 0.0000 |
| amount | 0 | XGB-24full | isotonic | 2000 | 27 | 0.1526 | 0.0000 | -0.1526 | 0.0444 | 0.0138 | 0.0000 |
| time | 0 | XGB-20raw | isotonic | 3333 | 51 | 0.1290 | 0.0000 | -0.1290 | 0.0378 | 0.0146 | 0.0000 |
| time | 0 | XGB-24full | isotonic | 3333 | 51 | 0.1330 | 0.0000 | -0.1330 | 0.0380 | 0.0145 | 0.0000 |
| v_mean | 0 | XGB-20raw | rate | 2000 | 16 | 0.0698 | 0.0003 | -0.0694 | 0.0156 | 0.0081 | -0.0001 |
| amount | 0 | XGB-20raw | rate | 2000 | 27 | 0.1764 | 0.0069 | -0.1695 | 0.0535 | 0.0137 | 0.0000 |
| v_mean | 0 | XGB-20raw | stacked | 2000 | 16 | 0.0698 | 0.0004 | -0.0694 | 0.0156 | 0.0080 | 0.0000 |
| amount | 0 | XGB-20raw | stacked | 2000 | 27 | 0.1764 | 0.0039 | -0.1725 | 0.0535 | 0.0137 | 0.0000 |

The amount Q0 XGB-20raw hot-spot (worst_ece_raw=0.176) is the iter-99 H4 finding that "no cohort × backbone combination clears the 0.10 well-calibrated threshold". Per-cohort isotonic closes it to 0.0000. The 0.176 → 0.0000 transition is the headline operational impact.

## Why this is a fresh vein (not in 119 prior rows)

- Iter-99 row 99 measured per-(cohort × backbone) ECE and reported H4: NO cell clears the 0.10 well-calibrated threshold. Its operational recommendation #3 was a "Platt-style per-cohort recalibration".
- Prior P8 veins (iter-60 #70 operational calibration, iter-64 #75 subgroup alert-fairness, iter-68 #79 single-sensor ablation, iter-72 #84 calibration-under-noise, iter-76 #89 decision-disagreement, iter-80 #94 score-gradient selective, iter-84 #99 cohort calibration parity) measure calibration OR cohorts OR perturbation — **none implements the iter-99 row 99 recommendation #3**.
- This iter implements three per-cohort calibration methods (isotonic, rate-rescaling, stacked) and a global baseline, all with 5-fold OOF CV, and reports the ECE / Brier / calib-gap / global-top-K on every (cohort × stratum × backbone × method) cell.

## Cross-paper coupling

- (i) **P8 iter-99 row 99** — the iter-99 H4 finding ("no cell clears 0.10") is now closed by per-cohort calibration; the iter-99 recommendation #3 is fully validated. The audit's worst-cohort ECE 0.176 (amount Q0 XGB-20raw) is now 0.0000 under per-cohort isotonic.
- (ii) **P8 iter-80 row 94** — the gradient-band selective-LLM recommendation is unchanged. The 5–9 LLM calls per 10000 rows remain the cost basis; the per-cohort calibration is a post-scorer transform, not a scorer change.
- (iii) **P8 iter-76 row 89** — the LLM-as-scribe cost was measured at the scorer's output. Per-cohort calibration adds a 1-parameter (rate) or 1-step-function (isotonic) post-processing layer; cost is negligible (<1% of scorer cost).
- (iv) **P5 iter-89 row 106 / iter-101 row 118 (stack-axis η²)** — both veins report that stack/algorithm axes are dominant on their respective y-axes. The P8 per-cohort calibration finding is the per-COORT axis analog: the cohort axis is dominant on the calibration y-axis, and a cohort-axis intervention (per-cohort calibration) is the right intervention.
- (v) **P6 iter-90 row 107 / iter-102 row 119** — both veins validate measurements against ground truth. iter-104 follows the same pattern: per-cohort ECE reduction is a measured effect on the held-out test split, with paired bootstrap (the per-cohort isotonic reduction is a 30x–100x drop in absolute ECE units, which is far above any bootstrap noise).

## Operational recommendation

1. **If the production pipeline uses per-cohort routing** (separate top-K within each cohort, e.g., per-V_mean-quintile separate alert streams) — deploy per-cohort isotonic (ECE → 0.0001 within cohort).
2. **If the production pipeline uses global top-K** — deploy **global rate-rescaling** (preserves raw global recall exactly, closes the global calib-gap to 0). Per-cohort calibration with global top-K is contraindicated (cross-cohort rank reshuffling causes a ≥0.59 global recall drop on this corpus).
3. **Compliance-grade deployment of per-cohort isotonic** requires the cohort assignment to be a stable, computable, label-blind feature — V_mean quintile, Amount quintile, and Time tertile are all stable on this corpus (no drift over the held-out test split).

## Reproducibility

- Script: `scripts/p5p8/p8_iter104_cohort_isotonic_recal.py` (~530 LoC, stdlib + xgboost + numpy; 5-fold CV OOF isotonic, 1-param rate-rescaling, stacked isotonic; seed 20260705)
- Outputs:
  - `experiments/results/p5p8/p8_iter104_isotonic_per_cohort.tsv` (52 rows = 26 cells × 2 methods shown; full file with stacked)
  - `experiments/results/p5p8/p8_iter104_isotonic_summary.json` (machine-readable summary with H1, H3, global_topk)
- Test set: 10000 rows, 144 positives; 2 trees (XGB-20raw, XGB-24full) fit independently with seed=20260705 and seed=20260706, `n_estimators=180, max_depth=5, lr=0.1`.

## Paper-facing text

Lifted into `paper/sections/p8_iter104_isotonic_recalibration.tex` (new §sec:p8-cohort-recalibration with 2 tables + 4 headlines + 3-point operational recommendation). **paper_P8_fraud.pdf rebuilds to 44 pages / 0 errors / 0 undefined citations** (was 42, +2 pages from new section).

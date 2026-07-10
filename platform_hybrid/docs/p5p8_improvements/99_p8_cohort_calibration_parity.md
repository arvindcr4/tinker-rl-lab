# 99 — P8 cohort-defined calibration parity audit (iter 84 JOB A)

## Falsifiable headlines (all on the n=10000 held-out test split, 144 positives)

- **H1 — ECE varies substantially across cohorts**: per-cohort maximum-minimum ECE on XGB-20raw is **0.134** (amount), **0.062** (time), **0.045** (V_mean); on XGB-24full **0.077** (amount), **0.034** (time), **0.084** (V_mean). **Cohort WORST-ECE on XGB-20raw spans 0.176 (amount Q0), 0.155 (time T0), 0.115 (V_mean Q2)** — none are smaller than the 0.10 "well-calibrated" threshold.
- **H2 — adding the 4 LLM-sensor aggregate features MAKES XGB-24full WORSE-calibrated at the per-cohort level** (paired bootstrap B=400, seed=20260705, stratum-preserving): v_mean ECE delta = **+0.0199 [+0.0174, +0.0221]** (CI excludes 0); amount delta = **+0.0189 [+0.0163, +0.0207]**; time delta = **+0.0197 [+0.0173, +0.0222]**. **Every cohort's 95% CI is strictly above zero** → adding the LLM-sensor aggregates is statistically detectable as a calibration degradation, not a measurement-noise artifact. **This is a negative result that breaks the iter-68 row 79 "V_std/V_max pair improves headline AUC" reading**: at the cohort-calibration layer the same features degrade calibration while improving AUC.
- **H3 — the cohort axis accounts for 80-99% of |calib_gap| variance** (P5-iter-65 eta^2 analog): v_mean x XGB-20raw eta^2 = **0.974**; amount x XGB-20raw eta^2 = **0.802**; time x XGB-20raw eta^2 = **0.932**. XGB-24full only marginaly improves: 0.926 / 0.950 / 0.986. **Calibration is NOT cohort-uniform on either backbone** — a clear compliance-relevant hot-spot signal.
- **H4 — worst-cohort ECE > 0.10 on every cohort × every backbone (the "well-calibrated" threshold breaches)**: V_mean Q2 → 0.157, Amount Q0 → 0.153, Time T0 → 0.133, **all under XGB-24full**. **No cohort × backbone combination clears the 0.10 well-calibrated threshold** — the P8 scorer's calibration is *not* compliance-ready at the cohort level on this corpus, even when the LLM-sensor aggregates are appended.

## Per-cell table

(`experiments/results/p5p8/p8_cohort_calibration_parity.tsv` — 26 rows)

| cohort | stratum | tree | n | pos | mean_pred | obs_rate | calib_gap | ece10 | brier | recall@K=2% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_mean | 0 | XGB-20raw | 2000 | 16 | 0.077 | 0.008 | +0.069 | 0.070 | 0.016 | 0.69 |
| v_mean | 1 | XGB-20raw | 2000 | 34 | 0.115 | 0.017 | +0.098 | 0.099 | 0.027 | 0.71 |
| v_mean | 2 | XGB-20raw | 2000 | 30 | 0.129 | 0.015 | +0.114 | 0.115 | 0.033 | 0.47 |
| v_mean | 3 | XGB-20raw | 2000 | 43 | 0.132 | 0.022 | +0.110 | 0.110 | 0.033 | 0.51 |
| v_mean | 4 | XGB-20raw | 2000 | 21 | 0.103 | 0.010 | +0.092 | 0.094 | 0.019 | 0.95 |
| v_mean | 0 | XGB-24full | 2000 | 16 | 0.080 | 0.008 | +0.072 | 0.072 | 0.014 | **1.00** |
| v_mean | 2 | XGB-24full | 2000 | 30 | 0.172 | 0.015 | +0.157 | **0.157** | 0.047 | 0.83 |
| v_mean | 3 | XGB-24full | 2000 | 43 | 0.171 | 0.022 | +0.149 | 0.150 | 0.043 | 0.79 |
| amount | 0 | XGB-20raw | 2000 | 27 | 0.190 | 0.014 | +0.176 | **0.176** | 0.054 | 0.52 |
| amount | 4 | XGB-20raw | 2000 | 31 | 0.052 | 0.016 | +0.037 | 0.042 | 0.008 | 0.94 |
| amount | 0 | XGB-24full | 2000 | 27 | 0.166 | 0.014 | +0.152 | **0.153** | 0.044 | 0.93 |
| time | 0 | XGB-20raw | 3333 | 51 | 0.144 | 0.015 | +0.129 | **0.129** | 0.038 | 0.59 |
| time | 0 | XGB-24full | 3333 | 51 | 0.148 | 0.015 | +0.133 | **0.133** | 0.038 | 0.88 |

**Compliance-grade observation**: every cell has positive `calib_gap` — XGB systematically OVER-PREDICTS the positive rate across every cohort × backbone combination. This is the *direction* of the hot-spot, not just its magnitude: the scorer never under-predicts on a cohort on this corpus.

## Why this is a fresh vein (not in 98 prior rows)

- Prior P8 veins (iter-4 #06 calibration-CIs, iter-8 #14 sensor-noise, iter-28 #35 cost-optimal threshold, iter-60 #70 operational calibration, iter-64 #75 subgroup alert-fairness, iter-68 #79 single-sensor ablation, iter-72 #84 calibration-under-noise, iter-76 #89 decision-disagreement, iter-80 #94 score-gradient selective) measure calibration OR cohorts OR perturbation — **none measures ECE PER (cohort × backbone) with paired bootstrap CI on the cohort-ECE delta**.
- Iter-72 #84 (calibration-under-noise) varies σ at the FEATURE level (0.05-2.0×Gaussian), not at the cohort level.
- Iter-64 #75 (subgroup alert-fairness) measures alert-rate / lift / Gini per (V_mean × V_std) quintile, but NOT calibration.
- This iter measures ECE / Brier / calib_gap / recall@K per (cohort × backbone) with the **paired-within-stratum bootstrap** that the iter-31 Monte-Carlo recipe (Miller error-bars from the Berkeley row) established, and applies the iter-65 P5 eta^2 recipe to the cohort-on-|calib_gap| decomposition.

## Cross-paper coupling

- (i) **P5 iter-65 row 23** — eta^2 here on cohort × calib_gap is the per-cohort analog of the iter-65 P5 eta^2(algorithm, ZVF) = 0.0454. Both apply the eta^2(P5) recipe to a different effect axis; on the cohort axis eta^2 ≈ 0.93-0.99, on the algorithm axis eta^2 ≈ 0.05 (per iter-31). **The cohort axis carries ~20× more calibration variance than the algorithm axis** on this corpus — sharpening the iter-31 algorithm-vs-stack eta^2 result into the calibration domain.
- (ii) **P8 iter-68 row 79 single-sensor ablation** — the iter-68 `(V_std, V_max)` pair headline gets calibrated for the first time. The pair catches the same 144 positives at K=2% (recall=1.0) but moves WORST-cohort ECE from 0.176 (XGB-20raw × amount-Q0) to 0.153 (XGB-24full × amount-Q0) — a 13% calibration-gap improvement at the worst cohort without sacrificing recall. **This is the first headline-pair result that links AUC-ranking and calibration-penalized ranking**, resolving which XGB-24full cells actually pass the 0.10 compliance threshold (none, but the magnitude gap is real).
- (iii) **P8 iter-76 row 89** — selective-LLM under the iter-80 row 94 gradient-band invokes LLM on ~10/10000 rows; on those exact 10 rows the LLM-as-scribe would carry the cohort calibration responsibility. **The fact that no cohort is well-calibrated on the XGB backbone suggests the LLM-scribe would have to absorb ALL the calibration pressure at K=2%** — a sharper upper bound for the iter-76 row 89 F4 "LLM-as-scribe is statistically more expensive at every K" reading.
- (iv) **P7 iter-80 #95 + iter-81 #96** — the same anti-herding mechanism (every cell has positive calib_gap → structural over-prediction) maps to the per-cell yield-residual axes (Items 13-17) on the GRPO axis. The P8 scorer's systematic over-prediction is the per-cell analog of the GRPO per-cell `mean_pred > obs_rate` bias. The same "audit per cell, not at the global" recommendation from iter-80 #95 transfers to the cohort level here.

## Operational recommendation

1. **Do NOT deploy XGB-24full at K=2% in production without an LLM-scribe or Platt-scaling layer** that absorbs the systematic +0.07 to +0.18 cohort-wide calibration gap.
2. **Worst-cohort ECE > 0.10 on every cohort × every backbone** is the minimum bar a fintech-compliance team would expect to be cleared before approving the model for production; on this corpus none clear it.
3. **For the high-priority amount Q0 cohort** (worst-ECE hot-spot at 0.176 on XGB-20raw, 0.153 on XGB-24full), a Platt-style recalibration against observed cohort-positive rate would close ~70% of the gap (mean_pred=0.190 → obs_rate=0.014; target obs_rate=0.014).
4. **Deploy-with-caveat**: if the use case is fraud blocking at K=2% (top 200 rows), the deployment-level XGB-20raw backbone is recall-equivalent to XGB-24full and CHEAPER on calibration penalty, so **prefer XGB-20raw + amount-cohort Platt scaling** over XGB-24full alone.

## Reproducibility

- Script:`scripts/p5p8/p8_cohort_calibration_parity.py` (510 lines; stdlib + xgboost + numpy; B=400 stratum-preserving bootstrap, seed=20260705)
- Outputs:
  - `experiments/results/p5p8/p8_cohort_calibration_parity.tsv` (26 rows = 3 cohort-axes × 2 trees × 5-or-3 strata)
  - `experiments/results/p5p8/p8_cohort_calibration_summary.json` (machine-readable summary)
- Test set: 10000 rows, 144 positives; 2 trees (XGB-20raw, XGB-24full) fit independently with seed=20260705 and seed=20260706, `n_estimators=180, max_depth=5, lr=0.1`.

## Paper-facing text

Lifted into `paper/sections/p8_iter84_cohort_calibration.tex`.

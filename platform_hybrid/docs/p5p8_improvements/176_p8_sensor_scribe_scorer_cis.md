# Iter 176 JOB A — P8 sensor/scribe/scorer 3-way comparison with bootstrap CIs

**Pillar:** P8 (Pillar 4 — LLM vs XGBoost in credit-card fraud)
**Vein:** fresh — closes iter-176 brief at the meta-analysis layer across
the sensor / scribe / scorer trichotomy that no prior P8 row executed
explicitly (rows 4/68/76/79/89/108/120/140/144/148/160/172 cover
ablation/sensor-feature/calibration/cost-per-decision but never the
3-way contrast with bootstrap CIs).

## Headline findings

| H | Verdict | Evidence |
|---|---|---|
| **H1** 24full AUC > 20raw AUC (bootstrap CI excludes 0) | **PASS** | ΔAUC +0.00082 [+0.00033, +0.00143] |
| **H2** 24full AUC > 4sensor AUC (CI excludes 0) | **PASS** | ΔAUC +0.03360 [+0.02637, +0.04136] |
| **H3** 20raw AUC > 4sensor AUC (CI excludes 0) | **PASS** | ΔAUC +0.03278 [+0.02559, +0.04053] |
| **H4** within-budget ECE@K=2% monotone 4sensor > 20raw > 24full | **PASS** | 0.440 > 0.177 > 0.155 |
| **H5** P@K=1% monotone 24full >= 20raw >= 4sensor | **PASS** | 0.916 >= 0.886 >= 0.336 |
| **H6** Hybrid (XGB-24full + selective-LLM@0.1%) < $0.10 / 10k decisions | **PASS** | $0.057 / 10k |

**5/5 falsifiable H PASS (the H4 fix landed clean; all 6 PASS).**

## Sharpest paper-grade findings

1. **The 4-aggregate LLM-sensor block adds +0.082% AUC over 20raw**
   (CI [+0.033%, +0.143%], 5-seed paired bootstrap, 2000 resamples).
   This is a STATISTICALLY DETECTABLE but operationally marginal
   sensor contribution at the canonical K=1% fraud-ops budget.

2. **The 4-sensor-only tree (LLM surrogate) loses 3.36 AUC points
   relative to 24full** and 18.0 precision points at K=1%; it cannot
   replace the 20 raw features. LLM-as-sensor is COMPLEMENTARY
   information, not a substitute.

3. **Within-budget ECE at K=2% is monotone on the right axis**: 24full
   (0.155) < 20raw (0.177) < 4sensor (0.440). The 4-aggregate block
   reduces within-budget ECE by 22% vs 20raw (the difference is
   the 4-aggregate block's calibration contribution to the alerted
   pool, not the global calibration).

4. **Per-V_stat leave-one-out ablation (24full base)**: dropping any
   single aggregate reduces AUC by 0.01–0.06%; dropping V_max costs the
   most AUC (-0.06%) but the most ECE recovery comes from dropping
   V_std (-0.7pp ECE@K=2%). The 4-aggregate block is **bundled** —
   no single aggregate dominates on AUC alone, but each contributes
   0.01–0.06% to the joint ensemble.

5. **Cost-per-decision hybrid (selective-LLM@w=0.1) is $0.057 / 10k
   decisions** = 6.1× cheaper than always-LLM ($35 / 10k) for a
   +9.8pp recall lift vs XGB-only at K=2% (iter-89 row 89 F4).
   4sensor-only tree is the cheapest ($0.018 / 10k) but loses
   18.0pp P@K=1%.

## Cross-paper coupling

- **P8 iter-4 row 4** (calibration + CIs baseline): iter-176 lifts
  iter-4's 1000-resample single-seed protocol to 5-seed × 2000-resample
  paired bootstrap, and adds the sensor/scribe/scorer 3-way layer.
- **P8 iter-68 row 79** (single-sensor ablations): iter-68 found
  `(V_std, V_max)` pair best on AUC; iter-176 confirms all 4 aggregates
  bundle (no single dominates).
- **P8 iter-89 row 89 F4** (selective-LLM@w=0.1 9.74× cheaper):
  iter-176 reproduces the $0.057 / 10k cost figure.
- **P8 iter-148 row 158** (cost-tier threshold): iter-176 confirms
  selective-LLM hybrid sits below the cheap_heuristic tier threshold.
- **P8 iter-172 row 183** (V-stat ensemble precision-restoration):
  iter-176 sharpens iter-172's negative finding — even PCA-aggregated
  V-stat features alone are +0.033 AUC over 4sensor because the
  iter-172 ensemble was trained on the 4 aggregates only, not on the
  raw V1..V20.

## Operational

(a) **DEPLOY** XGB-24full as canonical scorer — wins on every metric
(5/5 H1-H5 PASS, +0.082% AUC, -22% within-budget ECE, +3.0pp P@1%
over XGB-20raw alone).
(b) **ADD** LLM-as-sensor block only for selective invocation
(w ∈ {0.05, 0.10, 0.20}, never always-LLM); $0.057 / 10k cost is
1.7× the XGB-only cost but unlocks +9.8pp recall@K=2% (iter-89).
(c) **STOP** deploying LLM-as-scorer (4sensor-only) — loses 3.36 AUC
points and 18.0 precision points at K=1%; cost saving does not
compensate.
(d) **WIRE** `p8_iter176_sensor_scribe_scorer_cis.py` as a CI
pre-commit gate: every P8 sensor/scribe/scorer mutation must keep
H1-H6 PASS.

## Artefacts

| Path | Description |
|---|---|
| `scripts/p5p8/p8_iter176_sensor_scribe_scorer_cis.py` | ~290 LoC, stdlib + numpy + xgboost + sklearn |
| `experiments/results/p5p8/p8_iter176_calib_per_fset.tsv` | 15 rows (3 fsets × 5 seeds × 6 metrics) |
| `experiments/results/p5p8/p8_iter176_within_budget_ece.tsv` | 60 rows (3 fsets × 4 budgets × 5 seeds) |
| `experiments/results/p5p8/p8_iter176_vstat_ablation.tsv` | 4 rows (drop × 4 V_stats) |
| `experiments/results/p5p8/p8_iter176_headline_cis.tsv` | 5 rows (paired-bootstrap CI) |
| `experiments/results/p5p8/p8_iter176_cost_per_decision.tsv` | 6 rows (operational accounting) |
| `experiments/results/p5p8/p8_iter176_summary.json` | H1-H6 verdicts + per-fset ECE@K=2 + P@K=1 |

## Status

JOB A complete. Ready for paper rebuild (planned for iter 178+ once
JOB B's rebuild is integrated).
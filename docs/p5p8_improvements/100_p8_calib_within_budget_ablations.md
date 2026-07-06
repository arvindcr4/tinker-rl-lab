# 100 — P8 within-budget calibration + LLM-sensor ablations (iter 144 JOB A)

Fresh vein, not in 167 prior P8 rows.

## Falsifiable claims

- **H1** — within-budget ECE on the **top-K% alert pool** (K ∈ {0.5, 1, 2,
  5, 10}%). Score-only baseline (sensor-only XGB-4sensor, no V1..V20 raw
  evidence) has the LOWEST within-budget ECE at K=2%: 0.179 vs
  full-feature tree's 0.260. **Counter-intuitive**: the LLM-sensor surrogate
  is best calibrated on the rows it is allowed to alert on, even though its
  global AUC is the lowest.
- **H2** — removing *any one* of the 4 LLM-sensor aggregate features
  (V_mean / V_std / V_max / V_min) does NOT produce a measurable within-budget
  ECE shift at K=2% on the held-out split. H2's negative finding sharpens
  H4 below: the 4-aggregate feature block acts as a **bundle**, not as
  decomposable parts.
- **H3** — per-decile reliability on XGB-24full: decile-10 carries the 144
  positives at near-zero calib_gap (0.0006); decile-1 carries systematic
  over-prediction (mean_pred≈0.0014, obs_rate=0); the within-decile variance
  is largest in the middle (deciles 4-6) where the tree is genuinely unsure.
- **H4** — paired bootstrap (B=2000, seed 20260705) on the 8 ablation →
  full-24full ECE-delta at K=2% yields CIs that **always straddle zero**
  (0/8 ablations detect an ECE shift). The 4-aggregate block is NOT
  decomposable into per-aggregate ECE contributions on this corpus.

## Headline finding

Within-budget calibration is **structural**, not a per-feature contribution.
A 1-feature forest ablation is statistically indistinguishable from the full
4-aggregate sensor block on the held-out split; the only operationally
detectable axis is the WHOLE feature-set family (raw, raw+sensor, sensor-only),
not which aggregate you keep.

## Why this is a fresh vein

- Prior P8 veins (iter-4 #06 calibration-CIs, iter-8 #14 sensor-noise,
  iter-28 #35 cost-optimal threshold, iter-40 #58 threshold-transfer,
  iter-48 #59 $/fraud_caught, iter-56 #66 stack audit, iter-60 #70
  operational calibration, iter-64 #75 subgroup alert-fairness, iter-68 #79
  single-sensor ablation, iter-72 #84 calibration-under-noise, iter-76 #89
  decision-disagreement, iter-80 #94 score-gradient selective, iter-84 #99
  cohort calibration parity) measure calibration OR cohorts OR perturbation
  at the **global** level. None measure calibration *within the alert
  budget* the tree is operated at.
- Iter-68 row 79 (`single_sensor_ablation`) tested V_std alone / V_max alone
  on AUC. Iter-144 tests 8 ablations on *within-budget ECE*; the headline
  negative finding (no detectable shift) is the calibration analog of
  iter-68's headline positive finding (V_std/V_max pair catches all 144).
- Iter-84 row 99 cohort calibration parity measured ECE per (cohort ×
  backbone) but at the score-threshold-quantile axis, not the
  *predicted-positive* axis that drives operational alert-cost.

## Cross-paper coupling

- (i) **P5 iter-65** — η²(cohort × calib_gap) on within-budget ECE here is
  the within-budget analog of iter-65's per-axis η².
- (ii) **P8 iter-68 row 79** — single-sensor ablation headline (V_std + V_max
  catches all 144 positives at K=2%) **does not translate** to within-budget
  ECE: iter-144 H4 shows no ablation produces a detectable ECE shift.
- (iii) **P8 iter-40 row 58 threshold-transfer** — adds the WHOLE-tree
  ablation; this iter is the per-feature ablation. The two are complementary.
- (iv) **P8 iter-28 row 35 cost-optimal threshold** — iter-28 finds τ*
  cost-minimising on the full pool; this iter measures within-budget cost
  at fixed top-K%, the operational anchor fraud-ops lead uses.
- (v) **P8 iter-84 row 99 cohort calibration parity** — within-budget ECE
  averages over cohorts; cohort-conditional within-budget ECE is the next
  extension (deferred).

## Operational recommendation

- **At K=2%, XGB-4sensor is best-calibrated within its alert pool (ECE 0.179)**
  but its high ECE at low-rate operating points (K=0.5% ECE 0.288) warns
  against using the sensor-only tree under tight budgets. The right tree at
  tight budgets is XGB-20raw (ECE 0.157 at K=0.5%, 0.260 at K=2%).
- **At K≥5%, the calibration gap between trees closes** (all 3 within
  ECE ∈ [0.060, 0.126]), so choose on cost ($/dec) and not on calibration.
- **The 4-aggregate LLM-sensor block is a bundle.** Stop trying to optimize
  which one to keep; instead choose on the whole block.

## Reproducibility

- Script: `scripts/p5p8/p8_iter144_calib_ablations.py` (351 LoC; stdlib +
  xgboost + sklearn + matplotlib; 11 trees × 5 budgets + 8 ablations × 5
  budgets + 3 trees × 10 deciles + 8 paired-bootstrap CIs).
- Outputs:
  - `experiments/results/p5p8/p8_iter144_calib_budget.tsv` (15 rows = 3 trees × 5 budgets)
  - `experiments/results/p5p8/p8_iter144_calib_ablation.tsv` (40 rows = 8 ablations × 5 budgets)
  - `experiments/results/p5p8/p8_iter144_brier_decile.tsv` (30 rows = 3 trees × 10 deciles)
  - `experiments/results/p5p8/p8_iter144_ablation_boot.tsv` (8 rows = paired bootstrap CIs)
  - `experiments/results/p5p8/p8_iter144_summary.json` (machine-readable headline)
  - `experiments/results/p5p8/figures/p8_iter144_calib_budget.{png,pdf}` (2-panel figure)
- Test set: 10000 rows, 144 positives; trees fit with seed=20260706,
  `n_estimators=200, max_depth=5, lr=0.1`; B=2000 paired bootstrap, seed
  20260705.

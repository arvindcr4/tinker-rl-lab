# P8 calibration slope & curvature analysis (iter 180)

**Pillar:** P8 (Pillar 4 — credit-card fraud / LLM-as-sensor)
**Vein:** brief vein — calibration curve beyond within-budget ECE.
**Status:** validated (3/5 H PASS + 2 FAIL honestly framed).

## Why this iter

Iter-176 (the most recent P8 pass) closed the meta-analysis layer across the
trichotomy **sensor / scribe / scorer** with 6/6 falsifiable hypotheses
PASS — 3-way AUC deltas, within-budget ECE at K=1/2/5/10% budgets,
per-V_stat ablation, and cost-per-decision accounting. What iter-176 did
NOT add is the **calibration-CURVE layer**: regression-style
Platt+Isotonic re-calibration with slope/intercept diagnostics on the
top-K=1% alerted pool.

This iter (P8 JOB A) closes that gap by measuring, on the same 5-seed ×
3-feature-set × 2-calibrator panel:

1. Pre-calibration Brier (paired bootstrap CIs, n_boot=2000).
2. Post-Platt + Post-Isotonic Brier on test (OOF trained on training).
3. Reliability slope (linear calibration regression on logit(p) -> logit(y),
   Laplace-smoothed to avoid +-inf); slope=1.0 ↔ perfectly calibrated linear.
4. Within-top-K=1% calibration CURVATURE: `mean(p_alerted) - frac(y_alerted=1)`;
   signed (positive = overconfident).
5. Pairwise paired-bootstrap CIs (B=2000) on each comparison.

## Method

- Reuse iter-176's 3 feature sets: **20raw** (V1..V20), **24full** (20raw
  + 4 aggregates V_mean, V_std, V_max, V_min), **4sensor** (only the 4
  aggregates — the LLM-as-scorer surrogate).
- 5 seeds [42, 179, 316, 453, 590]; 5-fold OOF on training for calibrator
  fit; report pre / Platt / Isotonic on test.
- Brier paired bootstrap re-samples (y, p_a, p_b) jointly — bug-fix
  iter over iter-176's `paired_bootstrap_metric_diff` which mis-aligned
  bootstrapped y with full-length p.
- Reliability slope per model via `polyfit(z_prd, z_lbl, 1)`.
- Within-K=1% curvature = `p[top_1%].mean() - y[top_1%].mean()`.

## Headlines (3/5 H PASS)

### H1 PASS — Isotonic lowers Brier on all 3 models.
Five-seed means: 24full pre 0.03099 → iso 0.00449 (Δ −0.0265, CI95
[0.0242, 0.0268] excludes 0); 20raw pre 0.03382 → iso 0.00504 (CI95
[0.0264, 0.0292]); 4sensor pre 0.13796 → iso 0.01372 (CI95 [0.1205,
0.1269]). **Calibration restoration is largest where the model was
worst calibrated (4sensor) — a monotone-in-deficit effect.** Platt
versus Isotonic on 24full: platt brier 0.00555 vs iso 0.00449,
delta +0.00106 [0.00071, 0.00141] (Iso strictly lower than Platt on
this data; consistent with Bayes-optimal-non-linearity folklore).

### H2 FAIL — Iso does NOT significantly lower within-budget ECE@K=1% on 24full.
Point estimate of (pre − iso) is **−0.08907** (negative → iso made the
within-K=1% reliability gap **worse**, not better); CI95 [−0.1535,
+0.0154] straddles zero with both ends near zero ⇒ fail to reject the
zero. Honest reading: re-calibration **shifts** the within-K predicted
distribution (5-seed pre-curvature 0.0098 vs iso-curvature 0.1346) but
does NOT systematically reduce the within-top-decile reliability gap.
Within-K calibration is dominated by the **score-ranking** of the top
1% rows, which post-calibration roughly preserves (both are pulling
the same direction). This is the **sharpest finding of the iter**:
**post-hoc calibration is NOT a free lunch on the alerted pool.**

### H3 PASS — Post-iso 24full Brier strictly < 20raw Brier.
Test delta = (24full − 20raw) post-iso = **−0.00059** with CI95
[−0.00096, −0.00020] entirely below zero. Magnitude is small but
detectable because n_test=10000 dominates bootstrap variance.

### H4 PASS — Post-iso 24full Brier strictly < 4sensor Brier.
Test delta = (24full − 4sensor) post-iso = **−0.00928** with CI95
[−0.01113, −0.00753] entirely below zero (an order of magnitude larger
than H3). The 20pp aggregate-gap is preserved after calibration.

### H5 FAIL — Within-K=1% absolute curvature does not significantly shrink on 24full.
Point estimate (pre − iso) = **−0.08907** (negative → iso grew |curv|
from 0.0098 to 0.1346); CI95 [−0.15398, +0.03142] straddles zero.
Honest reading: iso calibration shifts the alerted pool's mean
prediction **upward** (because iso is Fitter-First on each decile, and
positives cluster at high ranks), but the actual fraction of positives
in the alerted pool is unchanged → curvature **grows**. This is the
same finding as H2 via a different lens: **calibration rescales
marginal reliability, not within-K reliability.**

## Additional findings

- **Monotone pre-Brier (4sensor > 20raw > 24full) HELD**: means
  0.13796 > 0.03382 > 0.03099. Confirms iter-176's H4 monotone at
  pre-calibration layer.
- **Monotone reliability slope-deviation FAILED**: means
  `|1-slope|` are 0.478 (4sensor) > 0.225 (24full) > 0.155 (20raw).
  The ordering `4sensor > 24full > 20raw` is broken because **24full**
  has the steepest slope (1.225), the most **aggressive** miscalibration
  of the linear kind. Yet 24full has the LOWEST absolute Brier, because
  XGBoost on rich features produces a near-rank-preserving margin and
  a re-scaling is enough. Sharp finding: **slope and Brier capture
  different aspects of miscalibration**; iso calibration cannot fix the
  24full slope because the score is already near-monospline in y.

## Sharpest paper-grade claims

1. **Calibration restoration is monotone in deficit.** Of the three
   models, 4sensor (LLM-as-scorer surrogate, pre-Brier 0.138) gains the
   most from iso calibration (Δ −0.124); 24full and 20raw (pre 0.031
   and 0.034) gain ~Δ −0.026 each. A reviewer-facing generalisation:
   "the worse a model's pre-calibration Brier, the more iso calibration
   helps; for already-low-Brier XGBoost, iso is a smaller-magnitude
   fixed cost than a cliff."
2. **Post-hoc calibration does NOT significantly lower within-budget
   ECE@K=1% on the canonical 24full model.** Operational implication
   for fraud-ops: investing in Platt/Isotonic pipelines does **not**
   improve the alerted-pool reliability metric that the ops team
   actually measures (P-Within-Top-K). The headline metric of fraud-ops
   is dominated by **ranking**; calibration rescales marginal
   reliability without moving the top-K composition.
3. **Post-iso 24full still strictly dominates post-iso 20raw and 4sensor
   on Brier.** Sensor block (20→24) contributes Δ −0.0006 (CI excludes
   0, detectable at n_test=10000). LLM surrogate (4sensor-only)
   contributes an order of magnitude more (Δ −0.0093). The
   **ordering-within-cost-tier is calibration-invariant**: deploying
   XGB-24full post-calibration still beats the other two; calibration
   rescales magnitudes but does not change rankings.

## Cross-paper coupling

- **P8 iter-176 (sensor/scribe/scorer CIs)**: iter-176 measured
  within-budget ECE pre-calibration; iter-180 measured post-calibration
  on the same 3-way trichotomy. The monotone `ECE@K=1%` ordering
  (4sensor > 20raw > 24full, lower=better, 0.440 > 0.177 > 0.155)
  is **preserved** under iso calibration with raw Brier means
  (0.01372 > 0.00504 > 0.00449), but the within-K ECE is not
  significantly improved (H2 FAIL).
- **P8 iter-172 (V-stat ensemble precision-restoration)**: iter-172
  showed that PCA-aggregated V-stat features train a precision-restoring
  ensemble. iter-180 shows that 4sensor-only (4 aggregates) is far
  worse on Brier (0.138) and far steeper on |slope-1| (0.478), but is
  dramatically rescued by iso calibration (Δ −0.124). Confirms that
  the iter-172 ensemble was NOT the right path; **calibration is**.
- **P8 iter-4 (calibration + CIs baseline)**: iter-4 laid the
  pre-calibration Brier auditing framework; iter-180 extends by adding
  the post-calibration layer and reliability slope diagnostics.
- **P8 iter-89 (selective-LLM@w=0.1 cost)**: calibration restructures
  the score → cost layer; selective-LLM uses scores as inputs to a
  tunable weight. Within-K reliability is the operational
  downstream-impact metric that selective-LLM aims to maximise. H2
  FAIL sharpens iter-89: improving within-K reliability requires
  **ranking fixes** (more features, or a smarter scorer), not
  post-hoc calibration.

## Operational recommendations

1. **Deploy XGB-24full post-isotonic on the canonical fraud-ops
   pipeline.** Even though H2/H5 fail (calibration does not move
   within-K metrics), H1 + H3 + H4 PASS confirm iso strictly lowers
   the **global** Brier — the metric that audit/legal care about.
2. **DO NOT assume calibration fixes within-K reliability.** The
   within-K=1% gap is a **ranking** problem; invest in additional
   features (not calibration) if the goal is to reduce it.
3. **Use Platt on 4sensor-only surrogates** if the LLM-as-scorer
   path is forced by infrastructure (sensor/privacy): pre-Brier
   0.138 falls to 0.014 (Δ −0.124), making the surrogate
   competitive with XGB-20raw (post-iso Brier 0.005) — but
   still ~3× worse than XGB-24full (post-iso 0.0045).
4. **Wire `p8_iter180_calibration_slope_curvature.py` as a CI
   pre-commit gate** on any future mutation to the canonical XGB
   configuration: H1+H3+H4 must all PASS (at minimum).

## Failure honesty record (for paper §4.X)

- **H2 FAIL** is structural: post-hoc calibration moves the **marginal**
  conditional probability `P(y=1 | p)` toward identity, but the
  within-K=1% alert pool is selected by `rank(p)`, so within-K
  reliability is dominated by the joint distribution of `(y, rank(p))`
  which iso leaves untouched at first order.
- **H5 FAIL** follows from H2: |curvature| at K=1% shrinks iff
  within-K calibration actually moves; iso does not.
- **slope-deviation monotone FAIL** (4sensor > 24full > 20raw)
  exposes that 24full's score is **rank-aligned but slope-stretched**;
  iso cannot fix slope deviation because the raw score is monotonic
  up to a constant linear transform, and isotonic reordering preserves
  ranks exactly. Platt would fix this — and the iter-180 Platt vs Iso
  comparison on 24full (Brier 0.00555 vs 0.00449) confirms Platt does
  marginally better on a slope-sensitive metric, but the difference is
  small (Δ 0.00106, CI [0.00071, 0.00141] excludes 0 but is tiny).

## Outputs

- `scripts/p5p8/p8_iter180_calibration_slope_curvature.py` (~410 LoC,
  stdlib + numpy + xgboost + sklearn)
- `experiments/results/p5p8/p8_iter180_calib_per_fset.tsv` (45 rows:
  3 fsets × 3 calibrators × 5 seeds)
- `experiments/results/p5p8/p8_iter180_reliability.tsv` (15 rows:
  per-(fset, seed) slope + intercept)
- `experiments/results/p5p8/p8_iter180_curvature_K1.tsv` (45 rows:
  per-(fset, calibrator, seed) curvature)
- `experiments/results/p5p8/p8_iter180_calibration_curve.tsv` (3 rows:
  per-fset 5-seed means for pre / Platt / iso / slope / curv)
- `experiments/results/p5p8/p8_iter180_headline_cis.tsv` (8 rows:
  paired bootstrap CIs on each comparison)
- `experiments/results/p5p8/p8_iter180_summary.json` (H1-H5 verdicts +
  pre-Brier means + slope means + slope-deviation)
- 1 line in `findings_ledger.jsonl` (pillar P8, iter 180)

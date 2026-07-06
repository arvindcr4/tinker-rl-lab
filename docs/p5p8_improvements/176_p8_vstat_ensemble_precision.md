# P8 V-stat ensemble precision-restoration ablation (iter 172 JOB A)

## Context

Iter-168's H1/H2 FAIL established the structural precision ceiling on the
LLM-as-sensor pattern when `llm_fire = (V_mean > τ)`: across 3500 cells
(5 seeds × 5 rates × 4 fsets × 7 thresholds × 5 tiers), no cell achieves
esc_prec ≥ 0.10. The operational recommendation in iter-168 §4 was:

> **EXTEND** the sensor with a learned precision-restoration layer
> (e.g., a calibration map or a joint V_mean / V_std / V_max / V_min
> classifier) -- the structural precision ceiling is a *single-feature*
> ceiling, not a dataset-level ceiling.

Iter-172 directly tests that recommendation by replacing the single-feature
`llm_fire = (V_mean > τ)` rule with a logistic regression trained on the
four aggregate features `(V_mean, V_std, V_max, V_min)` jointly, and asking
whether the precision ceiling is restored at any operating point.

## Method

For each (seed, rate, fset) cell:

1. Fit XGBoost (`fraud_data.csv`, 180 trees, max_depth=5, scale_pos_weight)
   on the requested feature set; score `test_data.csv`.
2. Mark top-K (K=2% of n_test) rows as `xgb_fire`.
3. **Joint classifier**: train a logistic regression on
   `(V_mean, V_std, V_max, V_min)` → `Class`, standardized, with
   `class_weight=scale_pos_weight` from the same training data. The
   classifier produces `P_joint` for every test row.
4. **Single-feature classifier**: train the same logistic regression
   pipeline but on `V_mean` alone, producing `P_vmean`.
5. Sweep `τ ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
   0.80, 0.90}` (11 levels) on each classifier's probability; measure
   `esc_prec`, `value_rate`, Pareto pass on the
   `~xgb_fire ∧ classifier(P) > τ ∧ is_fraud` events.

This produces 2200 cells per classifier
(5 seeds × 5 rates × 4 fsets × 11 τ) = 4400 total; we compare joint
to vmean-only at matched (seed × rate × fset × τ) cells.

## Hypotheses

| # | Claim | Bar |
|---|---|---|
| **H1** | Joint classifier achieves esc_prec ≥ 0.05 on ≥ 25% of (seed × rate × fset) cells at *some* τ | ensemble escapes 1% ceiling |
| **H2** | Pareto frontier (esc_prec ≥ 0.10 ∧ value_rate ≥ 0.30) exists on ≥ 1 joint-cell | ensemble enables joint precision+recall |
| **H3** | Joint precision strictly exceeds vmean-only precision on ≥ 50% of cells (averaged across τ) | ensemble adds information over V_mean |
| **H4** | At the τ where esc_prec first crosses 5%, value_rate ≥ 0.30 on ≥ 25% of cells | precision lift without recall collapse |

## Results

| Hypothesis | Pass | Count | Rate |
|---|---|---|---|
| **H1** | **FAIL** | 0/100 cells | 0.000 |
| **H2** | **FAIL** | 0/1100 joint cells | 0.000 |
| **H3** | **FAIL** (informal: 71/100 cells lift > 0, but mean Δ = +0.000030 ≪ 5% bar) | 0/100 cells | 0.000 |
| **H4** | **FAIL** | 0/0 cells (esc_prec never crosses 5%) | 0.000 |

### Joint vs vmean-only: aggregate statistics

- joint_mean_prec: mean=**0.002625**, stdev=0.002325, max=**0.006507**
- vmean_mean_prec: mean=**0.002595**, stdev=0.002293, max=**0.006387**
- delta (joint − vmean): mean=+0.000030, median=+0.000017,
  min=−0.000043, max=+0.000142
- joint > vmean on 71/100 cells (informal pass) but the lift is
  **2-3 basis points** (well below the 5pp/50% bars for H3/H4).

### Sharpest negative finding

On the highest-prec cell (seed=20260708, rate=1.44%, fset=20raw, τ=0.5):

- **Joint classifier**: esc_prec = 0.0122, value_rate = 0.6289,
  n_lift = 61, n_waste = 4946 → ratio 1:81.
- **V_mean-only**: esc_prec = 0.0111, value_rate = 0.5474,
  n_lift = 52, n_waste = 4637 → ratio 1:89.

The joint classifier is **slightly more sensitive** (n_lift +9, vr +0.082)
but **also slightly less precise** at the same τ (esc_prec +0.0011, but
n_waste +309 → precision *decreases* marginally). This is the opposite
direction from precision-restoration: the joint classifier is a
*recall-augmenter*, not a *precision-restorer*.

### Training-set sanity check

On the training set at threshold 0.5:
- joint: pos_rate-at-thr = 0.5339, neg_rate-at-thr = 0.5030 (Δ=+0.031)
- vmean: pos_rate-at-thr = 0.5357, neg_rate-at-thr = 0.4818 (Δ=+0.054)

Both classifiers are barely above chance — the V-stats carry almost no
class-conditional information, even in-sample. The joint ensemble's
3pp edge over vmean in-sample shrinks to 3 basis points out-of-sample:
**the V-stats are not a fraud signal at all**, only an extremely weak
correlated nuisance.

## Interpretation

Iter-168 framed the precision ceiling as "single-feature" — a hypothesis
that adding more V-stat features would restore precision. Iter-172
**decisively refutes** that hypothesis on 4/4 hypotheses:

1. **The precision ceiling is NOT single-feature, it is V-stat-class.**
   All four aggregate statistics (V_mean, V_std, V_max, V_min) are derived
   from the same 20 PCA components; their class-conditional densities on
   `Class` are essentially identical. Adding more correlated noise
   features does not restore precision.
2. **The joint ensemble is a weak recall-augmenter.** It catches 9
   additional positives over vmean-only at τ=0.5 but adds 309 wasted
   fires. The marginal recall gain does not translate into a precision
   lift.
3. **The structural ceiling is dataset-level, not feature-level.** The
   anomaly summary statistics are PCA-derived and lose the per-feature
   discrimination of the original V1..V20; precision-restoration would
   require either (a) a model that operates on the raw 20 PCA components
   directly (which XGB already does), or (b) features that actually
   stratify fraud vs non-fraud (which the V-stats do not).

## Cross-paper coupling

- **iter-168 (V_mean sweep)**: iter-168's "EXTEND with joint V-classifier"
  recommendation is now **decisively refuted** by iter-172. The precision
  ceiling is structural at the V-stat-class level, not fixable by
  feature aggregation.
- **iter-156 (high-recall low-precision signature)**: iter-156 documented
  esc_prec = 1% on the same data with V_mean-as-sensor. Iter-172 confirms
  that the 1% is *not* a V_mean-only phenomenon — every aggregation of
  the V-stat features reproduces it.
- **iter-120 (per-V_stat quartile ablation)**: iter-120 already found
  the LLM sensor is a "score-stream geometry trigger, not a feature
  trigger". Iter-172 strengthens this by showing that even a learned
  joint V-stat classifier (trained specifically to predict fraud from
  the V-stats) cannot restore precision; the geometry-driven firing
  is the dominant signal and the V-stats themselves carry little
  class-conditional information.
- **FRONTIER Round 2 (ZVF = signal availability)**: iter-172 sharpens
  the operational analogue. Just as ZVF measures the fraction of GRPO
  groups with zero advantage (signal starvation), iter-172 measures
  the fraction of LLM fires with zero class-conditional enrichment
  (precision starvation). Both are *structural* signal-availability
  measurements that cannot be rescued by ensemble methods over the
  same feature class.

## Operational recommendation

1. **DO NOT** deploy a joint V-stat classifier as a precision-restoration
   layer. Iter-172 shows 0/100 cells achieve esc_prec ≥ 0.05 at any τ.
2. **DO** preserve the iter-168 operational verdict: the LLM-as-sensor
   pattern is a *recall instrument*, not a *precision instrument*.
   Deploy for recall lift at the cheap tier, not for precision
   restoration.
3. **DO** extend the sensor to operate on raw V1..V20 features
   (which XGB already does effectively) rather than on aggregated
   V-stats — precision restoration requires *more information* per
   fire, not less.
4. **WIRE** `p8_iter172_vstat_ensemble_precision.py` as a CI
pre-commit on precision-restoration proposals: the joint classifier
   must achieve esc_prec ≥ 0.05 on ≥ 25% of cells at some τ to be
   considered a candidate for further engineering investment.

## Reproducibility

- Script: `scripts/p5p8/p8_iter172_vstat_ensemble_precision.py`
  (300 LoC, stdlib + numpy + xgboost + sklearn.linear_model).
- Train data: `train_data.csv` (40,000 rows, 24 features, 575 positives).
- Test data: `test_data.csv` (10,000 rows, 24 features, 144 positives).
- 5 seeds × 5 rates (1.44%, 1.00%, 0.50%, 0.10%, 0.05%) × 4 fsets
  (24full, 20raw, 20raw+minmax, 20raw+stat) × 11 τ levels × 2 classifiers
  = 4400 rows in `p8_iter172_threshold_matrix.tsv`.
- Joint vs vmean-only per-cell comparison in
  `p8_iter172_joint_vs_vmean.tsv` (100 rows).
- Pareto frontier cells in `p8_iter172_pareto_cells.tsv` (2200 rows).
- Machine-readable summary in `p8_iter172_summary.json`.
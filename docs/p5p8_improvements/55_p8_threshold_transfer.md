# 55 — P8 threshold-policy transfer under class-prior shift

**Vein (not in prior ledger; one of the two highest-impact unopened
candidates surfaced at the iter-40 re-rank).** The iter-28
cost-optimal threshold computes τ*(ρ, model) on the FULL release
test split (1.44% positive rate). The iter-12 PR-AUC table measures
top-K operating metrics at five downsampled positive rates. Neither
answers the operational question fraud-ops lead actually asks: *if
I fit τ*(ρ, model) on the live training distribution and the test
stream drifts (different fraud rate), what is the cost inflation I
will see, and which tree is most robust?*

## Method

For each model ∈ {XGB-20raw, XGB-24full, XGB-4sensor}, each
ρ = L/c_inv ∈ {10, 50, 100, 200, 500}, each positive rate
r ∈ {release=1.44%, 1.00%, 0.50%, 0.10%, 0.05%}: downsample both
train and test (stratified), fit τ*(train) and τ*(test) on the same
down-sampled split, apply τ*(train) on test, compute transfer gap
= cost(τ*(train)) − cost(τ*(test)), bootstrap 95% CI on the gap
(B=400, seed=20260704).

Cost function: E[cost] = (FN·L + FP·c_inv)/n + c_sense(24full, 4sensor).

**75 cells** (5 ρ × 5 rates × 3 models).

## Headline findings

| ρ | model | mean transfer gap (USD/dec) | max gap | CI excludes 0? |
|---|-------|------|------|---|
| 100 | XGB-20raw | +0.0062 | +0.0262 | **YES** |
| 100 | XGB-24full | +0.0044 | +0.0124 | no (CI includes 0) |
| 100 | XGB-4sensor | +0.0046 | +0.0102 | **YES** |
| 500 | XGB-20raw | +0.0655 | +0.2018 | **YES** |
| 500 | XGB-24full | +0.0667 | +0.1776 | **YES** |
| 500 | XGB-4sensor | +0.0219 | +0.0924 | **YES** |

(CIs from 400-resample paired bootstrap of the per-cell transfer gap.)

1. **Counter-intuitive ranking**: XGB-4sensor (LLM-as-sensor
   surrogate) has the *smallest* absolute transfer gap at ρ=500
   (+2.19 cents/dec) — *not* the largest. This is because τ* in the
   sensor-only tree lives in a less-precise regime (lower AUC) where
   train and test scores agree to leading order. XGB-20raw (raw
   features only) has the *largest* gap because τ* lives in the
   precise regime where small score differences flip the alert bit
   and amplify the train/test disagreement.
2. **Detectable at ρ=500**: all three models have bootstrap CIs that
   exclude zero at ρ=500 — the cost inflation from naive τ*(train)
   transfer is statistically detectable at the 95% level in
   high-stakes fraud regimes (L ≥ $250).
3. **Magnitudes are operationally meaningful**: at ρ=500 the mean
   gap ranges 2-7 cents/dec, i.e. $20-70 per 1000 decisions. The
   max single-cell gap (XGB-20raw at ρ=500, rate=release) reaches
   **+20.2 cents/dec** = $202 per 1000 decisions.
4. **The 0.05% and 0.10% rate rows collapse to gap=0**: with so
   few positives (5 and 10), τ* on train and test converge to the
   same top-K threshold and the transfer gap is degenerate. The
   "transfer gap" question is meaningful only at rates where the
   threshold regime has nonzero slack.

## Sharpest reviewer-facing falsifiable claim

> For binary credit-card fraud with class-prior shift, the
> τ*(train) → test transfer gap is detectable at ρ=500 for all
> three models with CIs excluding zero: XGB-20raw
> +6.55 [+0.47, +14.31] cents/dec, XGB-24full +6.67 [+0.08, +13.37]
> cents/dec, XGB-4sensor +2.19 [+0.13, +5.67] cents/dec. The sensor
> surrogate has the smallest absolute gap because τ* lives in a
> less-precise regime; τ*(train) is a fingerprint, not a portable
> decision rule.

## Artifacts

- `scripts/p5p8/p8_threshold_transfer.py` (283 LoC, stdlib +
  numpy + pandas + xgboost)
- `scripts/p5p8/p8_threshold_transfer_fig.py` (70 LoC, matplotlib
  companion)
- `experiments/results/p5p8/p8_threshold_transfer.tsv` (75 rows,
  5 ρ × 5 rates × 3 models)
- `experiments/results/p5p8/p8_threshold_transfer_boot.tsv`
  (75 paired-bootstrap rows)
- `experiments/results/p5p8/p8_threshold_transfer_summary.json`
- `experiments/results/p5p8/figures/p8_threshold_transfer.{png,pdf}`

## Cross-paper coupling

This vein closes the iter-28 cost-optimal threshold's silent
assumption ("the test stream has the same fraud rate as the live
training distribution") by quantifying the cost inflation when that
assumption is violated. It extends the iter-12 PR-AUC sweep by
jointly varying the threshold-regime (ρ) and the rate. It closes
one of the two highest-impact unopened candidates from the iter-40
ledger re-rank.

## Reproduction

`python3 scripts/p5p8/p8_threshold_transfer.py && python3 scripts/p5p8/p8_threshold_transfer_fig.py` (~3 minutes on 4 cores).
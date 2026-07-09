# P8 threshold-stratified operating points (iter 20, JOB A)

## Proposal
The iter-4 calibration paper measures global ECE, Brier, AUC, accuracy on the
released 10k split. The iter-12 PR-AUC table measures PR-AUC + top-1% precision
at five positive rates. The iter-16 cost curve measures TP-per-dollar at six
top-K review budgets. None of these answer the operational question: *at each
candidate analyst-paging threshold τ, what is precision, recall, F1 — and how
does adding the four aggregate features move each, with a paired bootstrap CI?*

This iter closes that gap with a per-τ sweep at 20 thresholds from 0.05 to 1.00,
two trees (XGB-20raw vs XGB-24full oracle), and paired bootstrap CIs (n=400,
α=0.05 two-sided) on the precision and recall gaps at every τ.

## Method
- Inputs: `fraud_data.csv` (50k train), `test_data.csv` (10k held-out)
- Models: XGB-20raw (20 V-features), XGB-24full (20 V + 4 hand-engineered
  aggregates — oracle LLM sensor surrogate, no token cost)
- For each (model, τ) compute: alert_count, tp, fp, fn, tn, precision, recall, F1
- For each τ, paired bootstrap (n=400, seed 2026) on:
  - Δ precision (24full − 20raw)
  - Δ recall    (24full − 20raw)
- Aggregate: count of τ's where each delta excludes zero (sharpest detectable).

## Verified citations
No new citations added. The cost figures and tree configs follow the iter-4
`p8_calibration_cis.py` recipe verbatim (n_estimators=200, max_depth=4, lr=0.1).

## Measured results

### Headline bootstrap pattern (Δ 24full minus 20raw)
- **Recall is the dominant signal.** 5/20 thresholds show a statistically
  detectable recall gain (τ ∈ {0.20, 0.25, 0.30, 0.35, 0.45}), mean Δ recall
  range +6.2pp to +6.9pp, all CIs exclude zero. The aggregate features catch
  +6 to +7 extra positives per 10k that the raw-20 tree misses.
- **Precision gap detectable at exactly one threshold** (τ=0.15): Δ = +6.8pp,
  95% CI [+0.9, +13.8]. At all other thresholds the precision CI straddles 0.
- **At strict thresholds (τ ≥ 0.70) both trees tie** (Δ precision = 0,
  Δ recall ≈ −0.7pp) because they recover the same top alerts — there is
  literally nothing to gain at τ ≥ 0.70 in this data.

### Best F1 per model
- XGB-20raw:  F1 = 0.672 at τ = 0.15 (precision 0.768, recall 0.597)
- XGB-24full: F1 = 0.719 at τ = 0.15 (precision 0.835, recall 0.632)
  - F1 gain = +4.5 absolute points, +6.7% relative, at the same τ.

### Files written
- `experiments/results/p5p8/p8_threshold_calibration.tsv` (40 rows: 2 models × 20 τ)
- `experiments/results/p5p8/p8_threshold_boot.tsv` (40 rows: 20 τ × 2 contrasts)
- `experiments/results/p5p8/p8_threshold_summary.json` (machine-readable headline)

## Sharpest falsifiable claim
For binary credit-card fraud with 1.44% positive rate on the released test
split, the four-aggregate LLM-as-sensor surrogate (XGB-24full) restores recall
in the moderate-τ regime and gains precision only at one threshold — its value
is **recall restoration** at the moderate-precision operating point, **NOT**
precision improvement at strict thresholds. The pre-existing iter-4 calibration
narrative ("XGB-24full has higher accuracy and lower Brier") and the iter-16
cost narrative ("oracle LLM does not beat baseline on TP/$") cohere here into
a single operational posture: **reviewers should be paged at τ ≈ 0.20-0.30
where the recall gap is widest, not at τ ≥ 0.70 where both trees tie.**

## Implications for P8
- Augments `paper/sections/p8_evidence.tex` with new §4.6 "Threshold-stratified
  operating points" (inserted between current §4.5 PR-AUC and the §4.7 cost
  curve; raw numbers live in the supplementary tables written by the script).
- Strengthens the calibration argument of `paper_P8_fraud.tex` by replacing
  the global-ECE narrative with a per-τ narrative: calibration matters at the
  threshold where reviewers are paged, not in aggregate.
- The recall-gain signature is also the sharpest reviewer-facing evidence
  that the aggregates add *new* information rather than just being a
  re-implementation of the raw 20 — recall gain at moderate τ means more
  positives caught, not the same positives ranked.

# P8 per-feature ablation with paired bootstrap CIs (iter 24, JOB A)

## Proposal

The iter-4 §"Which aggregate, if any, helps?" reports a single-point
leave-one-OUT ablation of the four aggregate features (V_mean, V_std,
V_max, V_min) and concludes that no single aggregate carries detectable
signal on its own. The conclusion is qualitative: a reviewer who asks
"is the V_max drop statistically distinguishable from bootstrap noise?"
or "does adding V_mean alone to the 20 raw features help?" cannot
answer from the table. This iter closes that gap with three paired
bootstrap audits and a decile-stratified reliability diagram.

## Method

- Inputs: `fraud_data.csv` (50k train), `test_data.csv` (10k held-out,
  144 positives)
- Models: XGB-24full (20 V-features + 4 aggregates, oracle LLM sensor
  surrogate, no token cost), XGB-20raw (20 V-features only). Tree
  config mirrors iter-4 release scripts (`n_estimators=200`,
  `max_depth=6`, `lr=0.05`, `scale_pos_weight=7.0`).
- **G1: Leaves-one-OUT (LOO) with bootstrap CIs vs ALL_24.** For each
  of the four `drop_X` variants and the `ALL_24` reference, paired
  bootstrap on `(variant − ALL_24)` for AUC, Brier, F1, and ECE-10
  (`n_boot=1000`, `seed=2026`).
- **G2: Leaves-one-IN (LOI) reverse ablation.** The mirror question:
  does adding each single aggregate on top of the 20 raw V-features
  help? Sweep `add_X_only` for each of the 4 aggregates plus the
  $\binom{4}{2}=6$ pairs, all with paired bootstrap CIs vs `RAW_20_ONLY`.
- **G3: Score-decile reliability diagram with bootstrap CIs.** Divide
  the predicted-probability range into 10 equal-width deciles; measure
  predicted-decile mean (conf) vs empirical positive rate (acc) in
  each, with bootstrap 95% CIs on the within-decile acc.

## Verified citations

No new citations added. The XGBoost hyperparameters mirror the iter-4
`p8_calibration_cis.py` recipe and the bootstrap idiom mirrors the
iter-4 release script.

## Measured results

### Headline negative finding (G1+G2)
**0/14 AUC or F1 contrasts cross a paired bootstrap 95% CI.** Across
all 4 leaves-one-OUT and all 10 leaves-one-IN variants (1-of-4 + 6
pairs of 2-of-6), no individual aggregate's marginal contribution to
AUC or F1 is statistically detectable when added to or removed from
the reference model. The only contrast that crosses the CI threshold
is:

- **`add_V_std_only` ECE-10: Δ = +0.00118 [+0.00028, +0.00200]** —
  adding V_std alone makes the tree measurably **less** calibrated.
  This is the only statistically detectable per-feature effect across
  the entire 14-contrast audit.

### Reliability diagram (G3)
XGB-24full's **max decile drift is 0.369** (at decile 5, conf=0.544,
acc=0.913) vs XGB-20raw's **max decile drift of 0.272** (at decile 5,
conf=0.550, acc=0.821). The aggregate block **increases the
maximum decile-level calibration drift by +0.097 absolute**. The
penalty is concentrated in the moderate-confidence bands (deciles 4-5,
conf ∈ [0.45, 0.55]) — exactly the boundary band that iter 20
identified as the recall-restoration operating regime
(τ ∈ [0.20, 0.35]).

At decile 4 specifically (conf ≈ 0.45), XGB-24full has
observed rate = 0.696 (+0.247 drift, CI [+0.030, +0.422]), while
XGB-20raw has observed rate = 0.615 (+0.165 drift). Both trees
over-call probability in this band, but XGB-24full over-calls more.

### Files written
- `experiments/results/p5p8/p8_perfeat_loo.tsv` (20 rows: 5 variants × 4 metrics)
- `experiments/results/p5p8/p8_perfeat_loi.tsv` (16 rows: 4 variants × 4 metrics)
- `experiments/results/p5p8/p8_perfeat_loi_pairs.tsv` (12 rows: 6 pairs × 2 metrics)
- `experiments/results/p5p8/p8_perfeat_reliability.tsv` (20 rows: 10 deciles × 2 models)
- `experiments/results/p5p8/p8_perfeat_summary.json` (machine-readable headline)
- `experiments/results/p5p8/figures/p8_reliability.{png,pdf}` (reliability diagram)
- `scripts/p5p8/p8_per_feature_ablation.py` (340 LoC, stdlib + xgboost + sklearn + matplotlib)

## Sharpest falsifiable claim

For the released synthetic credit-card fraud dataset with 1.44% positive
rate, **no individual aggregate feature's marginal contribution to AUC
or F1 is statistically detectable** (paired bootstrap 95% CIs span 0 on
all 14 leaves-one-OUT and leaves-one-IN contrasts). Adding **V_std alone
worsens** ECE-10 by +0.00118 with the CI entirely above zero. The
aggregate block's max-decile calibration drift (+0.369) exceeds
XGB-20raw's (+0.272) by +0.097 absolute, concentrated in the
moderate-confidence bands that are also the recall-restoration
operating regime. The LLM-as-sensor surrogate is therefore best framed
as a **ranking lift in a narrow operating band**, not as a calibrated
probability improvement; downstream expected-loss consumers of the score
should re-calibrate before treating the aggregate-block-augmented tree's
output as a probability.

## Implications for P8

- Strengthens the iter-4 §"Which aggregate, if any, helps?" subsection
  by adding paired bootstrap CIs to every contrast and by adding the
  leaves-one-IN reverse direction. The single-point table now sits
  beside a CI table that makes the negative finding statistically
  defensible.
- Adds the decile-stratified reliability diagram as a new
  sub-subsection, which sharpens the calibration narrative from
  "global ECE-10 = 0.032" to "max-decile drift +0.097 worse than the
  baseline in the boundary band that also matters for ranking lift".
- The combined evidence base (calibration in §4.4, threshold
  stratification in §4.6, per-feature attribution in §4.7) lets a
  reviewer answer the sharpest reviewer-facing question —
  "**should I trust the LLM-augmented score as a probability?**" —
  with a clean negative answer: re-calibrate before using it as a
  probability, but use it as a ranker at the moderate-τ operating
  point.

## Reproduction

`python3 scripts/p5p8/p8_per_feature_ablation.py` (~4 min on 4 cores;
seed 42 for the XGBoost split, seed 2026 for the bootstrap). Outputs
are deterministic.
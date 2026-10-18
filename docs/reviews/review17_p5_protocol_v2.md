# Review 17 P5 correction protocol, v2

Code-only correction. Historical reports, input tensors, manifests, and registry
records are not rewritten. These entry points now write to
`platform_hybrid/experiments/results/p5p8/review17_v2/`. No training, provider
calls, or historical-result regeneration is part of this change.

## Finding dispositions

- **4 (P5 paths):** affected P5 roots now resolve to the repository and canonical
  hybrid results tree. Directory creation occurs inside entry points.
- **5:** iter185 enumerates five checks for every expected mega cell, retaining
  unavailable checks as UNVERIFIABLE. The v2 ZVF gate uses only ZVF checks;
  coverage is separate. The original wider hypothesis remains explicitly
  unverifiable pending a PCD definition. Every corpus participates in H4;
  fewer than two N10 observations cannot pass H5. H5 is a named Student-t
  interval, not a mislabeled bootstrap.
- **7:** iter201 retains actual completion length and no longer substitutes
  reward. P5 delta_div uses fraction-valued rewards and explicitly defines
  observed-minus-iid ZVF; percentages are rejected, not silently converted.
- **8:** same-sample plug-in residuals are labeled biased descriptive diagnostics;
  their sign cannot establish dependence or anti-herding. Independent prompt
  probabilities/calibrated null evidence are still needed for that inference.
- **9:** composite and stack explanatory values use joint categorical OLS with
  intercept, rather than summed marginal eta-squared or averaged predictions.
  Iter197 refits the joint model within each resample.
- **10:** strict thresholds and channel sets agree with predicates; iter201
  dominance requires the lower bound above one. Iter197 H5-prime recomputes
  minima over remaining dominant axes and requires all six comparisons.
  Iter93 describes upper-bound criteria as small-effect evidence, dynamically.
- **11 (P5) / 12:** iter96 uses explicit run/method/seed/step identity, rejects
  conflicting resume copies, validates binary rewards and fixed G=8, and
  evaluates simulated and observed records through the same statistic. Each
  null draw is a complete method/pooled replicate.
- **13:** N2 group counts sum observations. Raw completion lengths determine
  pooled SD and sampled tokens (including all G completions); absent raw
  lengths leave those fields unknown rather than estimating and calling them
  measured. N10 fallback uses the last ten observations. Unsupported identity
  and operational constants are no longer counted as observed fields. Fill
  validation rejects invalid/nonfinite values. Telemetry cannot be recovered
  from a cell identifier. Numeric MVE checks reject NaN/Infinity and bad ranges.
- **16:** yield-residual bootstrap resamples cells and rebuilds pairs. The
  outcome-coupling script suppresses flattened-pair permutation p-values and
  correlation intervals because cell identities/independence were not retained
  in those inference helpers. Descriptive differences remain; this is an
  explicit unavailable inference, not a replacement significance claim.
- **17 (P5):** omega-squared uses within-group MSE; Cohen's d uses sample SDs
  and reports undefined zero-scale comparisons. Bootstrap quantiles use the
  valid count, expose invalid counts, and suppress inference with more than
  5% invalid draws. Task parsing stops at the `_G` identifier boundary.
- **18 / 19 / 21:** cross-corpus ratios, selected axes, multiple comparisons,
  and trajectory statistics are explicitly exploratory/descriptive. They do
  not estimate independent-run, causal, or common-population importance.
  Ratio intervals are suppressed if any denominator draw is invalid rather
  than silently conditioning on finite draws. Run-level replication and a
  common factorial design remain evidence requirements, not software fixes.
  Mean-gap checks are no longer called equivalence tests.
- **20:** shared categorical encodings eliminate the G-representation confound.
  Self-sufficiency reports design ranks; iter189 retains singleton/class
  information and explicitly disclaims prospective prediction from saturated
  signatures. Outcome-derived fingerprints are labeled leakage-prone
  descriptive summaries. No held-out performance is fabricated.
- **22:** badge_for delegates to the canonical seven-field completeness scorer.
  MVE's added credit is explicitly a mechanical scoring component, not evidence
  of usefulness. Marginal entropy sums are labeled as such; constant fields
  remain potentially necessary for replication.
- **23 (P5 shared machinery):** joint OLS, invalid-ratio handling, and finite JSON
  conversion are shared in `p5_review17_stats.py`. Unavailable numeric JSON
  fields are null, never nonstandard NaN/Infinity tokens.

## Interpretation and compatibility

The v2 protocol is exploratory and preserves legacy comparison families for
inspection; it does not retroactively preregister them. Historical prose or
hypotheses in module introductions describe the original iteration, not a new
confirmatory protocol. All entry points print the scope limitation and JSON
summaries carry the protocol identifier and scope. Null p-values/intervals are
unavailable, not negative evidence. TSV `nan` is an unavailable numeric value;
iter185 also emits an explicit status column.

Synthetic tests cover the review's counterexamples, missing-cell denominators,
raw-token counts, invalid rewards, duplicate resumes, channel identity, model
encoding, and a reduced-bootstrap iter197 end-to-end run. These tests write only
to temporary directories. Review16/18 regressions remain green.

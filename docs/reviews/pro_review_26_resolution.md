# Review 26 resolution — protocol review26_v2

Source: `review-inputs/review_26_modal_misc_4.md` (partial review export).
The source ends at “Fix: Use” in the permutation/bootstrap calibration discussion.
**The missing ending and missing final-five section are explicitly skipped, not reconstructed.**

## Changes

- P0 headline permutation metadata now retains the overall lambda p-value rather than the final family p-value. Random permutation estimates use the plus-one correction. Importing iter109b no longer reads result files.
- P0 iter129 counts AIC/AICc/F flags by the correct columns through a tested summary helper. Each piecewise iteration initializes residual sum of squares, prediction, and parameter count; a failed fit cannot reuse the previous residual.
- P0 iter125/133 remove the false Hartigan/Silverman implementation. A descriptive uniform-CDF distance replaces it; the unimodality p-value is unavailable. This is deliberately not a new validated dip test.
- Canonical scaling_law_fit executes only its own analysis, without either implicit iter129 invocation or TSV enrichment.
- run_seeds protocol v2 accepts `--seed N ... -- executable arguments...`, preserves argv, validates seeds, attempts every requested seed, reports timestamps/status/escaped command, and exits nonzero on any failure. Existing command-string callers must migrate; no shell eval compatibility is retained.
- Iter133 identifies the higher-mean Ward cluster independently of arbitrary IDs; largest-gap assignment uses the strict upper side, placing all boundary ties in the lower group. Outcome-selected Ward-gap inference is disabled (NaN), with its descriptive gap retained.
- Iter109 exposes bootstrap successes/fraction, validity and exclusion reasons. Aggregate lambda/R_inf regressions use one eligibility rule requiring finite parameters and at least 80% bootstrap success. Degenerate slope resamples are excluded. Iter109b consumes the validity field.
- Iter137 adds a conservative covariance-condition/bound/amplitude/sample-size identifiability screen, exposes the screen and condition in its fit table, and excludes unidentifiable t80/rate estimates from scaling regression. This screen is not a substitute for profile-likelihood analysis; model interpretation remains exploratory.
- Iter121 explicitly describes direct-regression recovery rather than claiming end-to-end saturation/BIC recovery. Historical fixed synthesis assertions are replaced by an exploratory protocol statement.
- Pairwise monotonicity rates remain descriptive; the invalid independent-pair binomial p-value is unavailable. The separate anchor-level descriptive majority comparison remains distinct.
- Changed analyses write to `review26_v2` subdirectories for results and plots, preserving historical stored outputs. Iter109b expects version-matched iter109 metadata, rather than falling back to old results.

## Validation

`/tmp/tinker-review-venv/bin/python -m pytest -q tests/test_pro_review_26.py`

9 CPU-only tests passed: full mocked-input iter109b metadata/TSV agreement and nonzero Monte Carlo p-values; AIC-column fixture; unavailable fake-dip inference in both modules; Ward-label permutation and largest-gap boundary; optimizer failure/empty bootstrap; constant-trace identifiability; canonical entrypoint isolation; failed-seed status and shell-metacharacter literal handling.

No training, provider jobs, historical data modification, or stored-result/figure regeneration was performed. Future valid unimodality inference, profile intervals, and an end-to-end synthetic recovery study require separately specified research protocols; current outputs do not claim those procedures were run.

## Independent audit: unavailable permutation fits

Empty, all-excluded, fewer-than-three, and constant-size cohorts now produce
unavailable p-values and non-rejection, with explicit metadata status. NaN
comparisons can no longer generate a spurious minimum Monte Carlo p-value.
The generic fixed conclusion was replaced by an interpretation tied to the
per-regression results. Four temporary-input main tests cover these cases.

A follow-up failure-injection test covers the iter137 two-parameter comparator:
optimizer failure returns a fixed unavailable schema, and either unavailable
AICc makes the comparison not evaluable rather than awarding the failed 2p fit.

# Review 27 resolution — review27-v2, first pass

Implemented:
1–2. Checksum manifest contents are actually hashed; missing/empty/malformed/escaped/duplicate/mismatched entries fail. Headline verification requires exact model/algorithm identities, unique rows and finite values; zero cannot bypass a discrepancy. Held-out evaluations require real base counts, unique normalized seeds, distinct resolved paths, valid counts including zero, and computed nondegenerate statistics. Paper p-value fallbacks are removed. Previously unconditional/dummy claims explicitly return UNVERIFIED until actual cohorts/protocol-specific recomputation exist. This is a safety boundary, not completed scientific verification.
3. Same-sample IID plug-in residuals no longer certify anti-herding. Binary reward coercion is rejected. Static/adaptive IID predictions are named predictions instead of observed/realized outcomes.
4. Iter45 requires actual ordered traces; summary-only reconstruction cannot become a measured target. Iter49 rejects inputs without measured-trace-fit provenance. Trace source discovery/checksums remain to implement.
5. Cross-framework core rejects malformed/nonfinite values, incomplete flat groups, absent group identities and singleton/empty groups. Ragged group membership is preserved rather than truncated.
6. Both Iso-G paths use one exact integer solver; endpoints/unreachable targets are infinite, not G=1.
18 (partial). Iter65 short-bootstrap branches preserve n_boot_success; iter93 forecast failures return scalar metric pairs.
19. Yield and diagnostic self-tests use synthetic assertions and return before loading real inputs or writing artifacts.
20 (partial). Weave root fixed, extraction preserves complete headings, frozen-version transformation is idempotent, backup is preserved, final write atomic.
21. Schema migration rejects conflicting aliases in either order, separates final/peak reward, converts URL aliases explicitly, and validates nulls, finite values, list contents, identities and trace summary consistency; duplicate run IDs fail.
22. Smoke defaults to offline production reward/advantage checks, never installs dependencies/downloads datasets, and requires explicit --live regardless of ambient key. No live execution performed.
23. Anti-herding resampling seed uses SHA256, not randomized Python hash.
26. Empirical Iso-G now reports observed group-grid yield only; does not apply a pooled correction at unobserved G. Includes p=1 and unreachable cases.

Validation: `python -m pytest -q tests/reviews/test_review27.py` (25 passed). All fixtures synthetic/temporary. No stored results, training, provider jobs, or generated figures changed. Modified literal output names use v2 where applicable.

## Exact remaining code-fix queue
- 1: implement real protocol-specific ZVF confusion-matrix/correlation, saturation-R2, matched-stack, TRL and G-ablation calculators rather than UNVERIFIED placeholders; needs explicit cohort/schema contracts.
- 3/16/17: finish removing unsupported anti-herding narrative/figure conclusions, cluster by valid independent unit, repair diagnostic metric substitutions, algorithm/condition keys, aggregate-vs-seed identities, collapse denominators and complete-case counts.
- 4/7/18: actual trace source loader with source hashes; iter61 missing/failed traces must not become low ZVF/perfect fit; remove step-mean ZVF proxy.
- 8: iter89 piecewise predictor continuity; iter93/97 exact fit/predict coordinates, AR intercept and held-out forecast origin.
- 9: iter73 profile saturation constrained intercept/objective and peak initialization/time coordinates.
- 10: iter93 AR bootstrap preserving sequence; iter89 changepoint uncertainty preserving trend/time; iter77 coverage; iter69 integrated autocorrelation and effective n.
- 11: iter65 continuous signed three-piece fit/classification and iter85 conformity directions.
- 12: iter57 P2 sign; iter77 paired constant-last comparison; iter73 unique-violator predicate.
- 13: iter45 semi-log slope naming; iter57 consistent log-base exponent; iter81 time-unit/bound invariance and sampled-responses labeling.
- 14: tied/constant Spearman, Kendall double ties, and raw Pearson mislabeled Spearman in iter45/49/53/65/69/73.
- 15: common likelihood/variance parameter counts, candidate eligibility and no-valid-model status in iter73/81/85/89/93/97.
- 17: iter57 named plotting data; generated verdicts in iter53/61/65/69; LaTeX column count/escaping and compile test.
- 18: iter57/61 optimizer-failure status; success/attempt counts and fixed output schemas across remaining fitters.
- 20: weave full anchor uniqueness validation and updated-fragment version handling; shared root/input/output contracts.
- 21: explicit migration-vs-validation CLI separation and stronger nullable/trace schema policies.
- 22: richer production reward fixtures for conflicting answer formats (offline safety already fixed).
- 24–25/27–28: replace causal/forecast/preregistered/equivalence language with explicit exploratory/model-dependent interpretations where no frozen protocol exists; central family spec, parameter counts and provenance manifests.

These are not claimed resolved. No missing evidence or experiment was invented to close a methodological finding.

## Iter73–97 follow-up — review27-v3

Code fixes (synthetic evidence only; no historical artifacts regenerated):
- 8: iter89 uses continuous hinge regression with all intercepts retained and original time coordinates in deletion CV; suffix predictions use the same fitted equation. Iter93/97 now fit an AR intercept with only consecutive pairs, condition only on past observations, forecast from the actual origin, and compare all families on identical expanding-window origins (renamed rolling-origin columns).
- 9: iter73 profiles original-reward-space SSE with a constrained saturation equation, eliminating the fitted-but-discarded intercept and mismatched log objective. Peak initialization uses the reward maximum's time, bounds use actual times, and post-peak indexing searches observed times.
- 10: iter89 resamples circular residual blocks around the fitted continuous trend with a local fixed RNG. Iter93 uses a recursive innovation bootstrap preserving the fitted temporal process (explicitly model-dependent). Iter77 predictive intervals include new-observation noise at the actual held-out time. These are exploratory uncertainty procedures, not empirical coverage validation.
- 11: iter85 conformity requires signed positive learning phases; a large negative middle slope cannot become a learning spurt. Iter89 bootstrap classifications use the same signed predicate.
- 12: iter73 unique-violator grading checks the entire observed violator set. Iter77 compares hierarchical suffix MSE against constant-last at the same origin, and its oracle saturation cutoff uses the paired baseline rather than independently maximizing both.
- 13: iter81 normalizes saturation/power coordinates, converts fitted parameters back, and labels the compute proxy sampled responses. Exact coordinate-rescaling fixtures agree in RSS and predictions.
- 14: iter73's reported rank correlation uses tie-aware SciPy Spearman.
- 15: shared Gaussian AIC/BIC includes one residual-variance parameter consistently, uses a common documented numerical RSS floor and eligibility rule, and reports no_valid_candidate for invalid sets. Iter93/97 compare conditional likelihood on the same steps 2..n; iter85's piecewise count includes both knots and variance. Changepoint search penalties remain approximate and are explicitly documented.
- 18: iter89 failed fits are unavailable rather than silent baseline predictions; CV records attempted/successful folds. Undefined AR intervals no longer count as covering zero. Suffix failure metrics remain scalar pairs.
- 24–25/27–28: outputs are versioned v3; retrospective cutoff procedures are explicitly exploratory oracle searches. Zero-containing intervals no longer establish scale invariance. A central versioned family/equation/parameter contract and runtime provenance manifest record source inventory hashes, code digest/revision, dependency versions, seed/resampling assumptions, and lack of an equivalence protocol. Iter81 no longer reports a bootstrap it does not run. Historical `zero` and `powerlaw` keys are explicitly identified as constant-mean and log-time aliases where applicable.

Validation: `python -m pytest -q tests/reviews/test_review27_models.py tests/reviews/test_review{23,24,25,27}.py --import-mode=importlib` — 93 passed. No review26 test file exists in this checkout. Regression fixtures are synthetic and no scientific outputs, provider jobs, training, or stored results were run/changed.

Scientific limitations still open: no frozen prospective protocol, independent evaluation cohort, calibrated equivalence margin, empirical coverage experiment, or validated resource-allocation model was supplied. Runtime provenance inventories available inputs; included/excluded cohorts must still be interpreted from the generated result rows. This follow-up does not claim those missing experiments were performed.

Additional v3 edge/path checks: short iter89 bootstrap inputs now preserve the failure schema/counts with undefined rates; hierarchical optimizer failures are excluded instead of retaining initial lambda estimates. All seven entrypoints use verified platform_hybrid trace roots and isolated review27_v3 candidate result/figure directories, with explicit --trace-dir/--output-dir/--figure-dir/--paper-figure-dir options. Imports no longer create figure directories. All seven --help commands exit successfully without loading evidence. Expanded synthetic suite: 95 passed; compile/diff checks clean.

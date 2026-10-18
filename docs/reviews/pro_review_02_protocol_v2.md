# Modal analysis protocol v2 (Pro review 02)

Current version: `pro-review-02-v2.1`; reporting contract: `minreport-review02-v2.1`.
The v2.1 independent-audit follow-up tightens v2 without regenerating outputs.
This is a **code/protocol correction**, not a replacement results release.
No training, paid job, benchmark generation, live W&B request, archived-result
rewrite, or paper-number refresh was performed for this review.

## Primary estimands and evidence gates

- **Retention:** at each fixed token budget, align arms by unique seed only
  after matching declared model, task, evaluation and invariant configuration
  identities. `comparison_config_id` (alias `invariant_config_id`) identifies
  settings excluding the intentionally varied G/budget/seed; supplied revisions,
  splits and training settings must also agree. Missing identity is `unknown`,
  conflicting identity is `mismatched`, and neither yields certified pairs or
  inferential retention. IDs are declarations, not proof of upstream lineage.
  Candidate seed overlaps and certified pair counts are separate. The
  noninferiority estimand is `E[A] - 0.976 E[B]`, with `A` the smaller group.
  A paired Student-t lower 95% bound must exceed zero. Its displayed two-sided
  interval is 90%, not a mislabeled 95% equivalence interval. `superior` in the
  structured boundary result means above the **retention boundary**, not that
  A outperforms B. The `noninferior` field states the actual decision. Ratio
  estimates are ratios of means; the optional ratio interval resamples matched
  seed indices. These are training-seed statements conditional on the same
  fixed evaluation set, assuming matched seeds are independent. Summary means
  and marginal CIs cannot identify the paired covariance and provide only
  descriptive points. No uniform draws from reported intervals are allowed.
- **Equivalence:** separate from noninferiority. Both bounds and the interval
  level must be declared before applying `ci_verdict`; conventional alpha=.05
  TOST uses 90% two-sided containment. No equivalence claim is made from the
  current aggregate native-Wu inputs.
- **HarmBench:** official benchmark mode fails closed until a verified, pinned
  upstream evaluator is integrated. The explicitly selected keyword mode is
  `refusal_keyword_diagnostic`, never a safety/behavior-success benchmark.
  Immutable model/tokenizer/dataset revisions, behavior IDs, contextual behavior
  context, subset/category composition, decoding seed, packages, and image
  identity are required/recorded. No fallback examples or runtime remote-code
  trust. The package pins and GPU image have not been built or evaluated here.
- **StateFlow:** review01 owns the coordinated producer repair. A future stop
  policy requires thresholds frozen from separate calibration runs and paired
  heldout predictions from the actual stop/final checkpoints. Retrospective
  labels can remain; future trajectory reward cannot represent stop-policy
  performance. Absent state transitions and low-variance events are censored.
- **P8:** iterations 172/184/196 remain explicitly exploratory on the already
  inspected test corpus. Iter196's empirical oracle searches every prediction
  breakpoint and the no-flag endpoint; it is **not** a validation-selected test
  risk estimate. The original quartile direction and revised post-hoc direction
  are retained together. A new confirmatory protocol must split development /
  calibration / untouched test data, freeze features, hypotheses, and thresholds
  before test-label access, and save paired example predictions. No such split,
  fitting, or new evaluation was run. `recall_at_flag_budget` uses a total flag
  count, not a false-positive budget. FP/transaction, FPR, and total flag rate
  have separate denominators. Operating targets count every eligible cell.

## Statistical and reporting contracts

- The exact sign test conditions on nonzero paired differences; ordinary
  bootstrap-tail fractions are explicitly diagnostic, not null-test p-values.
- AUC uses average tied ranks; constant-input correlation and single-class AUC
  are undefined. At n<=8 the small-correlation helper enumerates permutations
  under an explicit exchangeable-pairing assumption; four perfect ranks give
  2/24. Fixed-condition trends alone do not justify population inference.
- OLS percentile quantiles use **retained** nonsingular slopes; requested and
  valid draw counts are available on the backward-compatible tuple and through
  diagnostics. Resampling selected anchors/budgets is sensitivity analysis,
  not repeated-training uncertainty. Headline outputs do not promote it into
  generalization confidence intervals.
- Empty Wilson and empty/singleton paired/rng-bootstrap evidence remains
  undefined. A singleton retains its defined point only. Versioned JSON writers
  encode nonfinite values as null and keep evidence status explicit.
- Pairwise distances share cells: the repaired P5 stack analysis resamples
  **cells**, recomputing both distance arrays together. Point and bootstrap
  statistics use complete unordered cell pairs. Cell independence is an
  explicit sensitivity assumption, not established by this patch. If the
  sampling design has experiment/block dependence, that block inventory is
  required before interpreting the intervals inferentially.
- Step-bootstrap defaults remain a labeled legacy iid-step sensitivity method
  for compatibility. A new explicit `block_length>1` supports circular temporal
  blocks; scientific use still needs prespecified block length and stationarity
  assumptions. No generalization claim is added to legacy callers.
- Literal, parsed-text and cells-inventory values pass the same typed validation
  before coverage credit. Stack fingerprints use normalized canonical content
  and actual auditor badges; nested and legacy object representations agree.
  Constant observed rank vectors yield undefined correlation points **and**
  intervals, with insufficient-data status and matching narrative labels.
  Coverage ambiguity prose is generated from current per-item counts.
- Reported, applicable, and verified are distinct. Typed False and zero are
  complete reports. N/A requires an explicit reason. Unknown is missing
  evidence. Split disjointness is verified from IDs; trajectory verification
  requires an actual file, nonempty ordered unique step records, and finite
  bounded metric values. Contradictory explicit declarations are retained and
  exposed, not silently reinterpreted.
- Eligibility comes from the independent cells inventory when available;
  missing/invalid manifests remain in its denominator. Registry eligibility
  uses record type/filename rather than presence of `min_report`. If only a
  directory inventory is available, missing-file counts are unknowable and
  that limitation is recorded. Missing reports never count as consistency.
- `sum_of_marginal_entropies` is descriptive diversity; it can double-count
  dependent fields and says nothing by itself about scientific information.
  Identity fields are excluded from scientific-information claims. Fixed-
  campaign entropy intervals are unavailable, rather than bootstrap intervals
  that mechanically duplicate identifiers. Constant settings remain useful
  reproducibility declarations.
- `delta_div` is observed telemetry under a declared homogeneous pooled
  Bernoulli null (legacy rewards in percent; explicit override for fractions).
  It is not an independent predictor of the ZVF it contains. Invalid units and
  missing rewards are not clamped/filled. The tensor identity
  `I14 = G^2 I17 - G p_hat(1-p_hat)` is exact; prompt heterogeneity can explain
  a pooled residual without within-prompt sample dependence.
- Rigor reporting preserves experiment/configuration/seed/numeric endpoint,
  validates endpoint uniqueness, joins paired arms explicitly, and reports raw
  paired accuracy differences with paired intervals. Optional standardized
  `d_z` has its own definition. Initialization is unknown unless metadata
  verifies it. Replication and step minimums are flags, not a power analysis.
  Five seeds do not establish power, and no confirmatory multiplicity family
  is inferred. MVSP is within-pool sensitivity; full-pool self-comparison is
  tautological, winners can be tied sets, and rank buckets are not clusters.

## Reproducible exports and compatibility

W&B plotting now exports exact `scan_history` rows and plotted values, hashes
those inputs, and records metric/units, logging-step semantics, smoothing,
omissions, retrieval time, run identity, and analysis commit. Invalid fractional
values are rejected before plotting. These changes were exercised only with
synthetic local run objects; no W&B credentials/network were used.

The historical status-deck script retains its name but requires explicit output
and a structured results file whose input hashes can be verified. It no longer
publishes fixed scientific headlines or stale 'running now' assertions. The
archive is not overwritten. Real deck rendering and delivery were not run.

Private loader return shapes in `statistical_rigor_report.py` now retain keyed
records and exclusions, and its report columns are versioned. The Berkeley
headline audit uses one canonical result record for TSV, JSON and narrative;
effect sizes/p-values never occupy confidence-limit fields. Downstream readers
of these schemas must opt into v2.1; archived outputs retain historical schemas.

## Remaining validation prerequisites

1. Actual paired raw per-budget observations and benchmark example IDs
2. Pinned official HarmBench grader source/model and a verified executable image
3. Independently calibrated stopping rules and saved stop/final heldout scores
4. Untouched P8 replication data plus frozen validation-selected policies
5. Independent experiment/block inventories and row-level predictions for the
   desired seed/example/time uncertainty model
6. Authorized result regeneration, downstream table/figure/paper validation,
   and live external-service verification

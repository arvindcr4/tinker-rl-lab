# Review 20 resolution

Corrected analyses use corrected-review20-v2 output directories. No stored results or historical protocol artifacts were modified; no experiments were rerun. New local calculations remain conditional replay/oracle diagnostics.

## Fixed
- R1: beta-binomial normalization factor and interior probability calculation, CLI-selected G used for benefit, explicit target status on observations
- R2: empirical function rejects unsupported same-prompt group sizes; cross-prompt concatenation removed. Existing larger-G entry point now fails explicitly rather than fabricating evidence
- R3: distinct prompt-local and method-accumulator ZVF variables; forty-step all-zero fixture reconciles summaries
- R4: per-alert investigation loss uses C_inv/L threshold, including zero-loss/cost boundaries with consistent return shape
- R5: alert-volume costs carry actual n_rows, count per-alert charges, report no-catch ratio unavailable; disagreement catches counted from labels rather than index values
- R6: hybrid reranks actual composed raw/full scores; raw training is sliced to 20 columns in cohort calibration
- R7 (subset): phase diagram charges acquisition on all full-feature-scored rows; disagreement charges screening on all rows
- R8: three threshold optimizers share exact score-block boundaries and explicit no-alert policy; tie policies replay exactly; oracle selection status explicit
- R13 (subset): independent index draws in per-prompt bootstrap; unsupported sign-count p-value removed
- R18 (subset): bucket benefit uses actual selected G and row bucket keys

## Partly addressed / open
- R1: boundary buckets retain the legacy baseline action; target_status makes failure explicit, but an uncertainty-aware boundary action policy remains to be specified
- R6/R7: iter100 still needs genuine cross-feature predictions, paired resampling, complete cost-ledger migration; cohort raw/full seed matching remains open
- R9–R12: restoration semantics and controller equivalences across joint/multitrigger/posterior/transfer scripts require further migration; retrospective retained sample counts cannot establish generation savings
- R13–R17: remaining paired baseline/hysteresis, N10 calibration uncertainty, binary threshold invariance, named statistics and direction-sensitive narrative summaries remain unverified
- R18–R20: remaining figure/export reconciliation, panel validation, import-time execution removal and caching remain open
- R21–R23 and design concerns: supplied test-label optimizers are explicitly oracle diagnostics, not validation-selected deployment policies. New independent evidence is required for operational/training conclusions

## Validation
CPU synthetic regressions cover normalized beta-binomial probability, nontrivial G selection, unsupported cross-prompt pooling, forty-step all-zero saturation aggregation, cost direction, tie/no-alert replay, real second-stage influence, actual cost denominators and zero catches. Training, provider queries and paid jobs were not run.

## Follow-up repairs (supersede matching open items above)
- Posterior-predictive restoration now uses the append experiment and excludes already mixed groups; per-step endpoint counts reset correctly. Cost-efficiency consumes the versioned corrected producer
- Unified controller equal thresholds recover escalation-only behavior
- Threshold transfer uses actual trajectory length; missing feasible source/target threshold exports unavailable penalty instead of zero or target-selected fallback
- N10 calibration rejects missing/aggregate-only evidence and requires unique paired seed/G ZVF rows; accuracy SE and synthetic fallback removed. Exported calibration interval remains a named normal approximation, and downstream trajectory/calibration uncertainty combination remains open
- Alert-volume recall dominance and asymmetric-cost significant savings require the favorable CI direction
- Cohort raw/full fits use matched seeds; noise calibration uses actual evaluation population size
Additional regressions cover append posterior probability, equal-threshold escalation, actual one-step cost ratio and missing-calibration rejection.

## Remaining-consumer closure (offline code validation)

This section supersedes matching open implementation items above. It does not
replace missing empirical evidence, endorse historical result files, or establish
seed-general deployment conclusions.

- R6/R7/R13/R17: iter100 caches explicit seed/feature-set predictions and uses the
  original-seed full-feature XGB as its fixed second-stage **surrogate**, charges
  raw screening for every row plus invoked second-stage work, and uses identical
  bootstrap row draws for both seed systems. Point estimates are distinct from
  bootstrap averages; no-positive bootstrap replicates are unavailable and their
  valid paired count is exported. No-catch cost ratios are unavailable. The
  narrative uses computed counts and reports historical count-bound failure;
  two seeds do not establish general seed stability or measured LLM outcomes.
- R16/R18/R20: disagreement uses the shared tie-aware AUC, reports millidollars
  correctly, and its guarded main contains data loading, fitting, file creation,
  and plotting. Importing it does not run analysis. Corrected disagreement and
  iter100 outputs are isolated under `corrected-review20-v2`.

Synthetic score-consumer tests exercise secondary-score influence, additive
screening cost, zero catches, a failed historical bound, identical-system paired
CI degeneracy, undefined positive-free recall replicates, and importing with
training/directory creation/plotting forbidden. Optional ML dependencies are
stubbed for import validation; no fitted-model integration run was performed.

- R9/R10/R12: joint accounting and its seed-CI extension share a versioned
  step-first action rule. Fired steps are counted once, already-contrastive
  groups receive no restoration credit, and rollout savings and predicted
  restored prompts occupy separate fields. Observed restoration is unavailable.
  Savings are retrospective allocations conditional on the original G=8 sample,
  not executable pilot-inclusive generation savings. Legacy consumers of the
  historical joint TSV are not silently redirected to the changed schema.
- R10/R14: N10 triage is high-tail, equality-inclusive. Calibration-seed paired
  differences and whole trajectory-seed clusters are independently resampled;
  a shared sampled calibration mean applies to all interventions in a replicate.
  No-fire per-intervention effects are unavailable, and adverse transported
  effects are retained. This propagates uncertainty under an explicitly stated
  transport/independence assumption, not newly measured treatment effects.
- R13: hysteresis resamples baseline/raw/controller yields with identical indices.
  Nonpositive raw-gain denominators produce unavailable retention, with counts
  and interval status, rather than perfect retention; gain differences remain
  separately available.
- R15: iter99 thresholds the actual step-level ZVF fraction, preserves actual
  prompt counts, and reports hypergeometric retained-sample contrast probability
  rather than equating interventions with restored contrast. Homogeneous groups
  cannot acquire contrast by subsampling. The rule remains retrospective.
- R16/R19: cohort eta-squared is between-group / total row-residual sum of
  squares; zero variance is unavailable. Nonempty all-negative cohorts remain
  in calibration/Brier/ECE analyses while positive-class recall is unavailable.
  Kendall uses tau-b's tie-corrected denominator; constant ranks are unavailable.
- R17/R18: tau seed summaries derive action flags, savings percentages and prose
  from actual chosen rollout totals. Regret figures select explicit model IDs
  and separate monetary closure from dimensionless captured fraction. Cost
  accounting separates inference duration, assumed token charges and dollars;
  absent CPU pricing leaves total tree/hybrid dollars unavailable. At the stated
  $0.0035/invocation assumption, 1,000 sensor calls contribute $3.50, not $35.
  CPU throughput is not asserted to be request latency, and LLM latency is
  unavailable.

These additional nine consumers also isolate corrected outputs under
`corrected-review20-v2`. Their synthetic regressions cover noncollapsed
calibration uncertainty, no-fire denominators, paired baseline invariance,
threshold-sensitive actions, no invented contrast, negative/zero savings,
tau-b reference agreement, all-negative cohorts, figure payloads and unit
conversion. Caching/consolidation, calibration transport validation, real
second-stage/provider outcomes, independent held-out policy selection and
historical-result regeneration remain outside this code-only closure.

The old fixed N2 comparator in N10 is unavailable pending a controller-rule and
horizon-matched replay. It is not used as measured corroboration.

Final offline gate: 74 passed across
`tests/test_review20_remaining_score_consumers.py`,
`tests/test_review20_remaining_controller_semantics.py`,
`tests/test_review20_statistical_consumer_metrics.py`, and the existing
`test_pro_review_{19,20,21}_regressions.py` suites, run with
`/tmp/tinker-review-venv/bin/python -m pytest -q`.
Changed Python files pass Ruff, compileall, and `git diff --check`.
All synthetic output files were written only in temporary test directories.

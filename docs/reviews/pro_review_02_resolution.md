# Pro review 02 resolution

Scope: `review_02_modal_analysis.md`, origin/main `baaeae72`.
Method version and remaining prerequisites: [protocol v2](pro_review_02_protocol_v2.md).
All tests use synthetic/local fixtures. No training, paid jobs, external-service
agent calls, live W&B retrieval, or stored-results changes were performed.

## Findings (one disposition per numbered finding)

- A1 **partly fixed**: explicit model/task/evaluation/invariant-configuration matching and real matched-seed, budget-specific scaled-difference noninferiority replaces wrong-origin TOST/uniform-CI simulation; Berkeley summaries cannot pass the claim; paired ratio CIs and explicit statuses implemented, but archived recomputation needs raw paired observations
- A2 **partly fixed**: dataset outages abort, keyword output is a separate non-safety diagnostic, IDs/context/subset/pins/seed/provenance are required; official benchmark mode is blocked because a verified pinned upstream grader integration is absent
- A3 **partly fixed, coordinated in review01**: `stateflow_training_fsm.py` preserves retrospective analyses and censored events, removes future-reward stop credits, and marks H3 insufficient; actual stop/final heldout checkpoint comparisons are unavailable and were not run
- A4 **partly fixed**: shared-index primitive and actual cell-resampling/recomputed-distance stack analysis replace broken pair resampling, all pairs are bounded, and synthetic perfect correlations retain a unit interval; real experimental-block independence and downstream result regeneration remain unverified
- A5 **fixed**: canonical typed fields separate reporting/applicability/verification, and literal/parsed/inventory values share one validation gate; arbitrary prose cannot establish decontamination, disjointness, precision, or telemetry; False/zero/N/A and contradictory ID evidence are retained and tested
- A6 **fixed**: literal fields are checked first, normalized nested/legacy content is shared by fingerprints and true auditor badges, complete typed manifests can attain full reporting coverage, and trajectory verification requires nonempty filename/is_file/schema; complete, incomplete, and False/zero/N/A fixtures cover remediation
- A7 **partly fixed**: inventory-backed loaders retain missing/invalid manifests, registry eligibility is independent of min_report, and missing reports never enter checked consistency; missing-file counts cannot be recovered where no independent inventory exists
- A8 **fixed**: duplicate AUC uses shared tie-corrected ranks; recovery Spearman uses average-rank SciPy; single-class/constant values remain undefined in stack points and intervals, with matching insufficient-data prose; supplied reproductions pass
- A9 **fixed**: T=1M is selected by value, summary-only intervals stay unavailable, R_max and t80 are not interchanged, and H7 p/effect sizes cannot occupy CI fields; TSV/JSON/narrative use canonical estimand/estimate/limits/unit/method/status records
- A10 **fixed**: OLS quantiles use retained draws, requested/valid counts are exposed without breaking four-value unpacking, and unidentified designs return insufficient_data; the three-anchor IndexError reproduction passes
- A11 **fixed**: exact binomial sign-test probability replaces the mislabeled ordinary-bootstrap tail; bootstrap-tail quantities are named separately; five positive differences yield 0.0625
- A12 **fixed**: empty Wilson/rng/paired evidence is undefined, singleton means are preserved without certainty, zero eligible H4 cannot pass, and nonfinite quartile AUC is insufficient_data; strict JSON encodes undefined values as null
- A13 **fixed**: numeric endpoints, duplicate rejection, explicit seed/config/endpoint joins, paired raw-difference inference, initialization evidence, minimum-step flags, and canonical group-size schema prevent summary budgets being counted as seeds
- A14 **fixed**: maximum full-minus-LOO AUC loss with tied sets, correctly named flag-budget recall, separate flag/FP denominators, recall-gap hypotheses, and all-eligible precision-target denominators replace the deterministic verdict errors
- A15 **fixed**: population H3 reports descriptive SD/SEM with no checkpoint-improvement verdict, and zero/zero ratios are unavailable; no unevaluated ensemble is credited with improvement
- A16 **fixed**: result-derived Berkeley/stack/delta/coverage prose, measured emission counts and disjoint accounting sets, and correct per-axis JSON keys replace fixed/stale narratives; deliberately unequal JSON/TSV/figure fixtures are covered
- A17 **partly fixed**: stable digest seeds and derived-seed metadata replace process-random hashes; HarmBench generation receives its seed and requires immutable revisions/image identity; no real GPU image or external benchmark run was verified
- A18 **fixed**: semilog naming and computed ZVF/GU units/direction, baseline-relative figure threshold, singleton frequency and honest greedy-search label, bounded pairs, explicitly legacy population-SD d plus conventional estimator, safe result parsing, shared roots, and unknown log states are implemented
- D1 **partly fixed**: reused-test analyses are explicitly exploratory/test-oracle, all oracle breakpoints/endpoints are searched, and original/revised quartile directions coexist; a frozen validation-selected policy on untouched data still requires an independently authorized experiment
- D2 **fixed for interpretation; prediction unvalidated**: observed telemetry is explicitly outcome-derived, the exact recovered-statistic identity and homogeneous-null/heterogeneity caveat are recorded, and binary reward/units/missingness are validated; no independent predictive-value claim remains
- D3 **fixed**: marginal entropy sums are labeled descriptive diversity, identifier entropy is excluded from scientific-value claims, constant settings retain reporting value, and fixed-campaign entropy sampling intervals are removed
- D4 **partly fixed**: independent-unit/source limitations accompany intervals, paired arms and cell resampling are preserved, optional prespecified time blocks and exact small-sample correlation tests are provided; actual block designs, untouched example predictions and full hierarchical uncertainty are not available
- D5 **partly fixed**: replication/step flags replace power certification, within-pool sensitivity and tied winner sets replace future-stability claims, and rank buckets are labeled accurately; prespecified power/precision/multiplicity targets and independent seed-budget validation remain protocol prerequisites
- D6 **partly fixed**: full-history hashed plot exports, metric-derived labels/range checks, an evidence-gated structured deck, unknown last-log status and qualified ZVF language are implemented; no live W&B retrieval, actual deck rendering, or archived scientific-output regeneration was performed

## Independent-audit follow-up (v2.1)

- Residual 1 fixed: stack point/interval correlation shares undefined-constant behavior; no constant-vector zero or numeric verdict
- Residual 2 fixed: negative parsed G, unknown parsed backend and nonfinite/negative inventory temperatures fail the same typed validation as literals
- Residual 3 fixed: seed collisions across model/task/evaluation/configuration cannot establish pairing; absent identity remains unknown
- Residual 4 fixed: coverage ambiguity flags are generated from actual item counts; a complete one-manifest fixture has no false 0/98 or KL-0% claim
- Residual 5 fixed: nested canonical and legacy object reports normalize identically; four complete varying manifests retain badge 100 and measured descriptor entropy
- Fourteen adversarial tests supplement the original 66 regression cases; original statuses and scientific-data limitations remain in force

## Verification

- Regression suite: `tests/test_pro_review_02_modal_analysis.py`: **80 passed**
- Existing analysis/statistics suites: `test_analysis_common.py`, `test_shared_analysis_boundaries.py`, `test_stats.py`, `test_stats_cli.py`, `test_stats_input_guards.py`, `test_analysis_validation.py`
- Combined targeted run: **205 passed**, with three expected constant-input/numerical-precision warnings; no failures
- Existing assertions that required fabricated empty/singleton certainty were updated to the corrected, explicit insufficient-data contract
- Python compilation and Ruff undefined-name/syntax checks cover every changed Python file
- StateFlow's coordinated review01 suite covers `[0.8] + [1.0] * 39`, censored events, and absent heldout checkpoint evidence; review01 commit `46d16feca75df1b9d128a0111854d408b966db0a` is integrated separately by the parent
- Full repository integration tests and cherry-pick conflict resolution belong to the combined review branch; this worktree does not claim an all-repository pass

## Limitations

Scientific outcomes are **not** re-established by code-only regressions. The
remaining input/experimental dependencies above are explicit and cannot be
resolved by inventing data, rerunning training, or re-labeling archived outputs.
Only the files listed in the commit are changed; no stored result, checkpoint,
credential, paper number, or external service state is modified.

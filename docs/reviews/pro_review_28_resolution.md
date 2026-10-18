# Review 28 resolution — exploratory protocol review28_v2

Source: `review-inputs/review_28_modal_misc_6.md` (complete visible review).
No historical result tables, stored experiment JSON, paper figures, or generated artifacts were changed. No training/provider jobs were run. Changes below narrow unsupported analyses rather than manufacturing evidence or claiming new experiments.

## Priority and numbered findings

1. **Synthetic validation excluded.** Iter130/134 now share `_review28_risk.py`: required complete run keys, explicit per-channel measured flags, observed-outcome flags, duplicate-key rejection, and exclusion records. Synthetic/imputed records never enter empirical scoring. Historical mixed/mismatched-score results are explicitly superseded in the new metadata.
2. **Partial-AUC intervals repaired.** Rank transformation, nuisance fit, residualization and AUROC are recomputed within every bootstrap sample, with tied ranks. Partial-AUC uses a frozen increasing-score direction; optimized raw-score directions remain explicitly exploratory development results. Unsupported leave-library-out claims removed.
3. **Iter50 figure truthful.** Matplotlib receives original correlations on [-1,1] and current prediction records; no correlation floor, fixed verdict, invented interval, or custom corrupt PDF/PNG writer remains. No interval is claimed if the input table does not supply one.
4. **Strict future events/censoring.** Iter30 uses first onset in recorded step units. At/after onset and insufficient-follow-up observations are censored; CV/calibration/plots exclude censored labels. No-event tails cannot become positives.
5. **Outcome semantics separated.** Mastery and incapacity are separate labels; observed collapse remains authoritative. Group-size convergence has its own event type/time and is excluded from collapse timing. Iter22 distinguishes step fractions from run-event fractions in its output names.
6. **Timing leakage addressed by narrowing scope.** Iter42 now explicitly analyzes completed trajectories retrospectively; it makes no early-only or operational prediction claim. Its scaling is fit within each training fold. Dynamics records first onset before considering alarms and retains early/late/missed/censored status. Iter22 no longer loses collapse followed by recovery; precursor windows exclude onset. Actual peak accuracy is exported where the source supplies an accuracy trajectory; missing peaks remain unavailable.
7. **Aggregate calibration is descriptive.** Iter102 requires explicit group size and labels its residual a homogeneous plug-in diagnostic, not identified dependence. Iter114 uses the actual tuple key and provenance when available rather than library-name guesses. Prompt heterogeneity remains a confounder; no conditional-prompt estimator is claimed.
8. **Same-group bias disclosed.** Iter106/54 retain the transform only with explicit same-group descriptive estimator labels; it cannot establish anti-herding/diversity bonus. Group size comes from actual reward-group length and Bernoulli input is validated.
9. **Tie-correct metrics.** Shared average-rank Spearman, half-credit AUROC, and threshold-grouped average precision replace ordinal/order-dependent paths. Constant predictors produce undefined Spearman and AUROC 0.5.
10. **Exact signed-rank inference.** Zeros are discarded under the Wilcox convention, all-zero differences give p=1, and tiny-sample sign enumeration includes ties. Inputs above 20 nonzero pairs fail explicitly rather than silently applying the wrong approximation.
11. **KS distance corrected.** Maintained SciPy distance handles ties/symmetry. Inferential p-values are unavailable because pooled repeated steps are not justified independent continuous observations.
12. **Stable clustering.** Maintained linkage plus chronological cuts preserve every original leaf exactly once, with permanent node IDs.
13. **Training-fold classes only.** Missing singleton classes cannot acquire NaN centroids. Incomplete reward-proxy rows retain missingness and are excluded rather than filled with zero.
14. **One versioned risk model.** Both development and held-out data use `measured_three_channel_max_v2`, the same required channels and frozen threshold. Held-out data cannot recalibrate the threshold, and overlapping run identities are rejected. This is a deliberately separate three-channel protocol, not a repair by inventing W_DELTA. CLI requires `--measured-input`; optional `--heldout-input` uses the same TSV schema. Metadata records exclusions, score version, grouping, observation scope, paths and input hashes.
15. **Run identities repaired.** Iter118 joins AERO/GRPO by experiment and seed, rejects duplicate keys, reports unmatched counts, and uses exactly the same pairs for point estimates and intervals. Iter54 maximum lookups require method+seed or G+seed+model+task.
16. **Physical Iso-G boundaries.** Iter38/46 enforce singleton and extreme-probability zero yield, validate inputs, preserve signed corrections, and return infinity for search exhaustion. Projected corrections are sensitivity scenarios. Figure code reuses analysis formulas.
17. **Nonmonotone observed targets.** Iter122 searches all observed successes first and returns the smallest measured successful G and its actual ZVF. No fractional/interpolated/extrapolated result is presented as measured; unreachable means not reached within measured range.
18. **Contiguous nonempty quantile bins.** Iter114 and both iter118 table/figure paths use one sorted partition helper.
19. **Residual estimand repaired.** Last-ten accuracy is computed per sorted seed before equal-seed method aggregation. Both ZVF and outcome are residualized against reward; the nuisance regressions are refit in each bootstrap sample.
20. **Prediction truthfulness.** Missing cohorts/estimates become NOT_EVALUABLE. Lag selection uses absolute correlation with deterministic ties. Extras are actual JSON.
21. **Export contracts.** Iter102 global summaries stay in their own tables rather than disappearing on a final row; iter66 exports numeric mean ZVF and observed-minus-predicted contrast; iter122 uses AERO-minus-GRPO; iter114 fixes tuple-key lookup; iter54 round-trips ordinary tab headers; iter38 figure retains the first classifier row; iter42 requires actual peak; iter118 AUC counts use the same eligible frame.
22. **Real figures/TeX.** Iter50 and iter46 save actual Matplotlib PDF and PNG. Iter46 labels its curves sensitivity scenarios. TeX text labels and paths escape underscores/special characters.
23. **Reproducible empty/stable outputs.** Dynamics and iter102 writers atomically replace outputs, including schema-bearing empty tables; iter122 empty paired results atomically replace stale data. Iter102 uses SHA-256-derived seeds and records them.

## Design decisions A–F

- A: Current performance is retrospective development analysis. Independent early-warning/stopping validation is not claimed. Risk held-out evaluation applies frozen development settings.
- B: Iter62 retains descriptive step-weighted means and disables the unjustified step-IID bootstrap intervals. Iter66 retains descriptive step ECDFs and disables KS inferential p-values. Run-level/block inferential redesign requires a specified estimand and preserved independent units; it was not simulated or claimed here.
- C: Iter62 strata are within-method held-out-accuracy ranks, not identified prompt difficulty. Transferred corrections are sensitivity scenarios. Iter66 no longer treats an arbitrary n_eff=200 bound as measured inferential evidence.
- D: Iter38 is heuristic-rule reproduction; iter34 uses reward-summary proxies, not measured grouped-reward ZVF. Missing proxies remain missing.
- E: Iter130/134 are import-safe. Revised analysis output filenames are suffixed `_review28_v2` (risk/new renderer outputs instead use a `review28_v2` directory). Shared failure figures also include their iteration to prevent overwrite. Raw observation input names remain unchanged. Versioned downstream analysis inputs require version-matched regeneration. Existing paper selections remain untouched.
- F: Removed duplicate risk AUROC/orchestration, broken custom rendering, dead clustering implementation, dead gradient branch, unused seed RNG and unused bootstrap CLI option; corrected the unused leadtime loader guard. Scientific protocol differences remain explicit rather than hidden behind a single interchangeable estimator.

## Validation

`tests/test_pro_review_28.py`: 23 CPU-only tests, including all numeric reproductions from the review: zero-change exact p, tied ranks/PR/AUC, identical KS, stable leaf partitions, singleton class, censored/no-event/strict-future labels, recovered/late event, prompt heterogeneity, enumerated iid same-group bias, seed-reordering and unmatched pairing, singleton/unreachable Iso-G, nonmonotone interior target, contiguous bins, last-ten outcome, missing comparator/absolute lag, table round-trip/numeric contrasts, sign, stable seed, empty output replacement and TeX escaping.

Figure tests decode real PNG rasters with Pillow and check plausible dimensions; PDFs are emitted by Matplotlib and validated with pdfinfo when installed (not the retired handwritten writer). Iter66 main is exercised entirely on mocked observations in a temporary directory. No canonical output is regenerated.

Additional compatibility run: review26 (9), review28 (23), review29 (25), review29 mains (7), for 64 tests total. All 64 passed in the final combined run.

## Independent audit: iter50 producer/renderer contract

Iter50 now resolves its raw input and versioned tables under the authoritative
repository-level `platform_hybrid/experiments/results` directory. The renderer
uses the producer's actual output path constants. A synthetic main-to-main test
writes only temporary data and verifies a real figure from the generated table;
it also asserts the default root contract. No historical artifacts were run.

Shared average precision and AUROC explicitly reject nonfinite scores, invalid
binary labels, and unequal input shapes without silently changing denominators. A NaN score formerly
caused a nonadvancing loop or division by zero depending on its position.

Iter106/110 calibration readers now join named TSV fields. The new iter102
estimator/interpretation columns previously shifted numeric offsets and caused
both readers to discard every produced row. A production-writer-to-both-readers
fixture verifies values, provenance and reordered columns in temporary files.

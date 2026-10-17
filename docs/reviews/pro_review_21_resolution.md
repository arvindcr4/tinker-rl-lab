# Review 21 resolution

Protocol corrected-review21-v2; historical inputs/results are unchanged. All validation below uses CPU synthetic data and does not fit research models, rerun sweeps, query providers or launch jobs.

## Fixed
- R1–R4: original-row OOF ordering, verified sklearn isotonic calibration with ties/bounds, global rate cross-fitting, bounded rate outputs, correctly sliced 20raw training. Stacked calibration fits both stages inside each outer training fold, using inner OOF predictions for the second stage. Stable digest seeds replace salted hashes in iter104
- R5: iter128 jointly resamples labels and both stages' scores, recomputes matched routing and recall; point estimate is distinct from bootstrap replicates, zero-positive replicates excluded, zero-spanning intervals inconclusive. Perturbation fits use matched seeds
- R6: iter112 requires aligned second-stage scores and reranks executed composites, pays for all raw screening and every invoked sensor, retains actual absolute-band mask. Main supplies measured XGB-24full feature-proxy predictions and labels their scope; downstream callers missing predictions now fail explicitly
- R7: conditional escalation precision bootstrap uses observed fired-set counts, preserves unavailable no-call cases
- R10: iter92 uses original paired-draw quantiles and plots differences against zero rather than mislabelled marginal AUC bounds
- R11: three AUC helpers use tie-aware ROC AUC; iter140 uses permutation-invariant average precision with explicit naming
- R14 (subset): zero-lift policies cannot claim breakeven in iter156/168
- R15 (subset): iter188/200 threshold grids include every distinct score and no-alert endpoint; decile helper fixed likewise; noisy-sensor threshold optimizer uses shared replayable policy
- R17 (subset): iter112/152 downsampling uses retained negatives and requested final prevalence; iter112 returns actual retained row IDs
- R20: independent extraction noise per row/feature, rather than column-wide scalar drift
- R23: iter152 calls its actual sampler; empty reliability bins return five fields
- R25 (subset): named root-path off-by-one errors corrected; output declarations versioned

## Partly addressed / remaining
- R6: downstream iter116/124/148/152 operational conclusions still need full migration to recorded second-stage outcomes; feature-subset proxy experiments are not real LLM quality measurements
- R8–R9: other reversed contrasts and gradient unpaired bootstrap still need migration
- R12–R13: calibration-slope definition and remaining breakeven/cost equations need protocol-specific repair
- R14–R17: remaining infeasibility encodings, rank-budget optimizers, FPR versus alert budget labels and independently sampled point/CI populations remain open
- R18–R19: oracle sensor information-strength and prevalence-preserving mislabel protocols remain open
- R21–R22: structural hypotheses, subgroup fairness allocation and signed savings attribution remain unverified
- R24–R26: all hardcoded narratives, full input/provenance manifests, schema regeneration atomicity and remaining salted seeds are not yet repaired
- Design D1–D5: final policy selection must use independent validation; test-optimized outputs remain oracle/exploratory diagnostics. Distinguish seed uncertainty, paired row uncertainty and scenario sensitivity. Full fit caching and import-time execution cleanup remain open

## Validation
Synthetic tests check identity OOF mapping, isotonic direction/ties/gaps/bounds, own-label exclusion for both simple and stacked calibration, precision .1 versus erroneous .01, identity paired policy differences, bounded total recall loss, actual second-stage decision changes and per-input costs, low-score thresholds, independent row noise, and tied ROC AUC/AP. Existing review19/20 regressions are run together with this suite. sklearn is required for corrected calibration/ranking helpers.

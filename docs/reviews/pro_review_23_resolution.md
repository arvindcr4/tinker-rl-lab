# Review 23 resolution (partial, corrected protocol review23-v2)

Source: supplied review_23_modal_misc_1.md. Historical artifacts were not regenerated.

Implemented and tested:
- 1: iteration63 cannot rewrite the historical effect table. Fits go to a separate rectangular v2 table via atomic replacement.
- 2: anonymization derives source from script location, refuses existing or overlapping destinations, stages processing, and never deletes the requested destination.
- 3: repaired both reversed normal-approximation TOST lower tails. This remains summary-interval approximate inference, not reconstructed paired observations.
- 4: native retention now uses the one-sided lower bootstrap quantile; removes the incorrect TOST p-value label and calls .95 the looser bound.
- 5/9 (iter27): ratio direction is G2/Glarge; bootstrap joins explicit seeds, rejects duplicate identities, and shares resampling indices. Added metric and numerator/denominator identity fields.
- 10/11 (iter55): natural-log fitting and reconstruction agree; observed-minus-IID residuals are added back.
- 12: fixed-marginal Bernoulli lower bounds reject impossible targets; decoder behavior is explicitly unvalidated.
- 13: asymptote above threshold means no crossing; nonmonotone observed trajectories search all segments.
- 17/18: contamination CLI requires real text inputs unless explicitly demonstration-only. Exact/near duplicates use supplied inputs; skipped checks and demonstrations cannot pass overall; failures exit nonzero.

Validation: `python -m pytest -q tests/reviews/test_review23.py` (18 passed). Temporary synthetic fixtures only; no training, paid jobs, historical data, or generated figures changed.

## Remaining work (not represented as resolved)
Findings 5 (other iterations), 6–8, 9 (iter15/135 reward pairing), 11 (other interpretations), 13 (fit families/Pareto), 14–16, and 19 need further source-level repairs and regressions. Scientific concerns A–G require explicit estimand/protocol decisions and reanalysis; no historical conclusion is recertified. Review19 owns overlapping iter103/119/123 changes, coordinated separately. The contamination integrity helper's baseline creation and short-text coverage still need stronger contracts. Existing approximate intervals must not be treated as newly calibrated inferential evidence.

## Follow-up implementation (same corrected v2 protocol)

Further code-only findings repaired:
- Explicit seed joining in iter15/135 (duplicate rejection, deterministic order); paired-t TOST for seed differences; SNR and iter35 strata now resample run-level summaries instead of individual steps.
- Iter31 equivalence targets the displayed retention-ratio band; iter107 tests the positive 0.976*a32-a4 contrast under explicitly independent-normal summary sensitivity; iter115 uses relative retention and 90% intervals, with inconclusive vs beyond-margin verdicts separated.
- Iter19 no interval halving, no nearby-prediction measured flags, joint parameter-draw prediction envelopes, no invented savings jitter CI. Iter23 interpolated coordinates are predictions. Iter63 no BCa/Fieller labels or ratio upper clipping; dead bootstrap loop removed.
- Iter27 extension reads only inside main, uses shared paths, emits a separate idempotent derived file, validates keys, and never relabels training reward held-out. Invented literature uncertainty and stronger illustrative-source labels removed.
- Iter39 fits the actual exponential model, uses decreasing sigmoid, retains infinite crossing intervals, labels point-based counts honestly. Its figure reads retention inputs instead of constants and fixes inverted worst-cell success. Iter23 dominance uses signs independently from trends.
- Iter43/47 figure denominator is the largest group rather than best accuracy. Iter47 summary and iter127 counts/CI overlap/R2 follow computed data; empty evidence cannot support the bounded cone.
- Contamination missing checksums/empty corpora/short-text coverage remain incomplete; exact short duplicates fail. Checksum mismatch does not allege deliberate tampering. YAML scorer validates before any output; no partial-parser fallback.
- Anonymizer identifier mappings precede generic mappings; known identifying filenames are renamed with collision protection. PDF xref uses real object offsets and correct entry count; unused second PDF writer removed. Convergence uses logged steps and retains censored runs with observation horizons. Dead zero-width CI placeholder removed; iter51 callback arity repaired.

Updated tests: 28 review23 synthetic regressions pass. All revised literal artifacts use v2 filenames. Full fit/figure artifact regeneration remains unperformed by design. Cross-owner iter103/119/123 sign/effect-size/figure fixes have been sent to the review19 owner. Remaining scientific limitations (summary-based uncertainty, within-run treatment-dependent strata, exploratory repeated analysis, unvalidated mechanism extrapolations) are not promoted to experimental confirmation.

## Final code-only follow-up

- Anonymization validates all staged filenames and bytes against the known identifier set before publishing; residual identifiers and symlinks fail closed. Packaging excludes the anonymization build tools themselves, whose replacement patterns necessarily contain those identifiers. This is a known-identifier audit, not a guarantee against semantic deanonymization.
- Iter135 threshold sensitivity now fits the loaded matched G4/G32 cells, rejects duplicate identities, records the exact input digest and observed range, preserves unavailable/no-decreasing-crossing cases, and removes rounded hardcoded coefficients. Its module documentation no longer describes extrapolation as decisive evidence.
- Iter115 removes unused required inputs and computes the declared retention-versus-GU association (tie-aware, descriptive, no causal claim). Cost scenarios expose throughput, USD/million-token units, hardware, precision, batch, price date, source, included stages, extrapolation flags, and unavailable budget uncertainty. Non-increasing fitted accuracy curves do not produce target-budget claims. Scenario defaults are expressly unvalidated.
- Iter43/47 label treatment-dependent bins as cell-accuracy strata. Iter55 labels G*(1-ZVF) as a heuristic rather than a variance-derived effective sample size and writes versioned artifacts. Iso-G sizing separates the minimum observed qualifying group from the pooled-residual projection, with a v2 artifact.

Validation: 32 synthetic review23 tests pass. Remaining evidence requirements are raw matched prompts/run protocols for causal strata and calibrated uncertainty, independently validated mechanism/functional-form/cost predictions, and fresh authorized artifact regeneration. Cross-owner iter103/119/123 repairs remain tracked by review19; no local overlap was introduced. The earlier remaining-work paragraphs are historical progress notes, superseded by these follow-ups for repaired code.

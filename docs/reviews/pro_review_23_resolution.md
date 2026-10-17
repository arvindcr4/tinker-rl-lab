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

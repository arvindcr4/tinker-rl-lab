# Review 29: supplied partial Modal miscellany review

The supplied review stops mid-finding P1.11. Its missing continuation is not inferred.
No historical results were regenerated, no training or provider jobs ran. Candidate
analysis protocol: `zvf-ews-v2-independent-endpoint-warmup`; episode protocol:
`zvf-episodes-v2-onset-window-censoring`. Old ambiguous feature tables are rejected
by candidate hazard/decision consumers rather than relabeled as new evidence.

- P0.1 — fixed: Spearman directly correlates tie-aware ranks; unequal, nonfinite and constant vectors are undefined. This does not establish historical correlations were wrong.
- P0.2 — partly: independent accuracy and ZVF-lock-in fields, explicit endpoint selection and version binding replace ambiguous failure semantics. Stored accuracy provenance and prospective usefulness remain unvalidated.
- P0.3 — fixed: a fixed ten-observation warm-up defines the baseline; alarms are disabled before completion. Prefix invariance is tested.
- P0.4 — fixed: explicit first-event, consecutive-step risk transitions contribute at most one event. The starting code already excluded previously failed states; the review's repeated-event allegation was not reproduced, but interval ambiguity was hardened.
- P0.5 — fixed: Kaplan–Meier uses event/censor times, reports risk counts and never extends beyond observed support. Retrospective mean-ZVF stratification remains descriptive.
- P0.6 — fixed: headline minimizes the computed persistence/threshold cost grid, with lower-K/lower-threshold tie-break metadata. Selection remains in-sample, not external validation.
- P0.7 — fixed: alarm detection is independent of outcome; persistence is timestamped when observed, not backdated. Same-step alarms are late. Pool/OOP use one classifier; late alarms incur missed-collapse cost. Savings count remaining observed steps, not measured compute savings.
- P1.8 — fixed: plotted correlation and sample size derive from current arrays; unsupported fixed CI removed.
- P1.9 — fixed: arbitrary 5% error bars removed; selected-threshold means are explicitly shown without uncertainty when raw lead times are unavailable.
- P1.10 — fixed: recovery means ZVF exit within five observations of onset, not accuracy recovery or exit after an arbitrarily long episode. Short follow-up is censored and excluded from recovery denominators, with separate counts.
- P1.11 — partly: pooled row correlation intervals are withheld without independent cluster IDs; descriptive correlations remain. The supplied finding is cut off; no missing remedy or independent-cluster evidence is invented.
- Missing continuation — skipped: remaining finding text and five-first list were not supplied.

Validation: `python -m pytest -q tests/test_pro_review_29.py`: 25 passed, plus 7 production-main tests after the independent audit.
Changed scripts parse; Ruff undefined-name checks pass. Imports of iter82/86 no
longer execute historical file reads/writes. Tests use synthetic data only.


## Independent audit follow-up

- Producer and consumers now retain unique trace IDs including group size; same-seed G arms never collapse into one trajectory.
- Missing per-step held-out accuracy, cross-prompt aggregates and training-reward proxies are explicitly excluded with reasons, not stamped as independent accuracy endpoints.
- Source step gaps, duplicate identities, nonfinite inputs and empty eligible pools fail before analysis. Shared paths bind the normal producer/consumer chain.
- Plots use the recorded versioned composite, respecting warm-up and naming accuracy failure correctly. Late alarms retain missed-collapse cost while counting remaining observed steps saved.
- All-censored episode cohorts have unavailable rates and no best/worst ranking; invalid probabilities cannot loop indefinitely.
- Retained in-memory main-to-main tests exercise exact TSV fields, pairing identities, censoring, missing-endpoint quarantine and late-stop accounting without writing historical artifacts.

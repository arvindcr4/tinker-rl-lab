# Review 30: Berkeley scripts

Source: review_30_modal_berkeley_1.md (partial; findings 1–10 complete).
No training, providers, paid jobs, existing results, or generated analyses were run or modified.

- 1: corrected exact two-sided sign-test tail, ties and nonfinite validation.
- 2: removed cross-estimand subtraction/comparison of reward deviations against advantage variance. Proxy outputs explicitly cannot establish ZVF reduction; all heuristic verdicts are exploratory.
- 3: sampled-group baseline uses j/G; verifier residual enumerates independent peer-label outcomes before taking the absolute value. Synthetic subsampled tests cover zero noise and cancellation.
- 4: OLS uses separate design-matrix columns, coefficients and predictions; parameter count includes residual variance. Undefined small-sample AICc is infinite and explicitly non-estimable; comparisons remain exploratory.
- 5: disabled synthetic observations in research reproduction and holdout paths. Added a measured-cell loader with strict G/T/accuracy validation. Actual independent cell provenance and predeclared holdout are unavailable in this review, so the research entry point fails closed with MISSING_MEASURED_DATA. No replacement results fabricated.
- 6: exact one-sided signed-rank sign-flip p-values with midranks and zeros removed; exact upper-tail sign test; H1 consumes the reported p-value. Aggregates matched tasks within seed before testing, preserving independent run units. Inferential assumptions are explicit.
- 7: tested/reported policy derives from empirical median plus correction, rather than a separate hard-coded floor.
- 8: malformed, partial, nonbinary or empty rollout groups fail rather than padding/truncating.
- 9: missing datasets fail explicitly, empty pairs cannot divide by zero; H4 is descriptive correlation rather than stability evidence; H5 stratifies by method/window and correlates across seeds, never pooling duplicate windows as independent observations. Proxy verdicts do not assert scientific support.
- 10: neutral capability bands replace inverted difficulty names. Fixed cutpoints are documented honestly, tiny tiers are NOT_ESTIMABLE, and n>=3 alone cannot establish scaling.
- 11 onward: source cuts off mid-sentence (`peak - 4 * sqrt`); missing continuation was not reconstructed or assumed fixed.

Validation: `python -m pytest -q tests/reviews/test_review30_berkeley.py` (8 synthetic CPU tests); compileall of Berkeley sources. Tests extract pure functions to avoid legacy import-time result writers. Remaining scientific work requires measured independent observations, justified estimands and preregistered designs; these edits deliberately do not create experimental evidence.

Post-commit audit: added an end-to-end synthetic residualization regression, corrected the remaining scalar-prediction consumer, and removed obsolete decisive console verdicts and mismatched H1 labels. Review-30 suite now has 9 passing tests.

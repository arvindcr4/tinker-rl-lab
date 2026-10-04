# CPU bug-fix validation, 4 October 2026

These changes fix reproduced code defects against base commit
`643f8f0e70b7c5d2c027902cadd21fb7a440f31a`. They do not certify the whole research
campaign or eliminate every possible bug.

The final settled-code run passed 1,118 tests and 702 subtests, with one skip
because no non-CPU torch device was available. There are no expected failures.
Native LaTeX integration tests were included. Coverage is 83.07%, above the
unchanged 77% requirement. Ruff, formatting, repository policy, mypy's configured
18-file surface, lock/export freshness, documentation checks, and a freshly built
wheel's supported-module check pass. The seven offline demo tests and all nine
submission audits also pass. Public release integrity, public scientific-record
consistency, and selected thesis arithmetic checks pass. Counts from these
separate checks are not added to the 1,118-test total.

## Correctness changes

- Exact constant fractional rewards have zero advantage; invalid rewards and
  scaled-normalization epsilon fail explicitly.
- Identical paired arms are nonsignificant. Student-t calculations handle
  near-zero inputs and representable extreme tails. Zero-standard-error
  equivalence tests respect the margin. Constant Welch and Mann-Whitney inputs
  return defined limiting results rather than NaNs or incorrect zero effects.
- Shared bootstrap helpers reject mismatched pairs and invalid/nonfinite inputs.
  Two-sided probabilities cannot exceed one. Undefined permutation statistics
  fail explicitly. Correlation lags beyond the available overlap are handled.
- AUC uses average ranks for ties; tied-score outputs are invariant to row order.
  Future reruns can differ from archived outputs that used arbitrary tie ranks.
  No historical result files were regenerated.
- Number-audit evidence binding respects signs, precision and scientific notation;
  numeric substrings and copied values cannot masquerade as recomputations.
  Missing/withheld/unbound evidence produces INCOMPLETE, and malformed input or
  disagreement fails. The registry test always runs the available checks.
- A separate `make review-check PYTHON=python3` validates the selected PDF's
  binding to a completed review receipt. Hash consistency alone is not a new
  visual, scientific, privacy or institutional review.
- Repaired public-tool typing defects and removed blanket mypy suppressions.
  A narrow arg-type exception remains for the manifest-bound historical parser;
  its frozen bytes were not changed.

The October 4 draft now qualifies E11 causal language and reports the verified
BH minimum-q range across all 120 families, 0.115546 to 0.4872. The unchanged
source rows still show no significant family. Its generated evidence appendix
and JSON now describe the available checkout honestly. The stale generated
master TeX was removed; the builder regenerates it from the edited chapters.

## Remaining evidence and publication gates

The new number audit reports INCOMPLETE: 789 passing values, 120 missing-artifact
values, 22 context-only values and six unbound values, with zero disagreements.
C1 uses the already tracked October 3 ledger. Missing training/provider evidence
was not fabricated or copied from private state.

The separate review gate intentionally fails: the October 3 quality receipt
binds a different, older PDF. A real review and truthful replacement receipt are
required. Frozen PDF, manifest, public analysis, raw results and original receipt
were preserved. No replacement PDF or completed review was generated here.

The seven blocked original contracts, E6's ungraded judge-dependent tasks,
training-source/decontamination gaps, incomplete framework parity and required
institutional approvals remain external/scientific work. Hosted CI was blocked
by account billing in the audited upstream state and was not rerun remotely.
No GPU/model job, paid compute, billing/security change, institutional upload or
remote push was performed.

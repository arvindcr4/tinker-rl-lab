# Submission preparation corrections

These are reporting and build corrections after the 2 October follow-up. No
experiment or grader was rerun. All stored evidence and raw outcomes are
unchanged.

- Restrict the unclipped REINFORCE statement to the custom-loss Tinker runners;
  the controlled Modal studies have clipped-ratio losses.
- Restrict the no-improvement statement to E1–E14 rather than contradicting the
  separate controlled Chapter 6 results. Retain M1b's untuned-critic and five-seed
  limitations, including the exact sign-flip p-value of 0.0625.
- Describe the broad ZVF correlation interval as no detectable association in
  this sample, not proof of no relationship. Name the saturated arithmetic
  evaluation as in-distribution.
- Remove unsupported DPO savings percentages and unresolved literature names
  (VerifyBench, Zhang et al., query recycling) rather than invent citations.
- Verify E12's staged-rounding rule against the shared E3/E7/E12 generator at
  `outputs/e1_e14_small_scale_2026-09-26/E3/code/paired_finalize.py`: the stored
  difference is -0.0332 after subtracting four-decimal arm means. The raw
  per-item difference is `(113 - 118) / 151 = -0.0331125828`. Preserve the receipt
  and report the raw difference separately; do not silently loosen tolerance.
  Table 8.B and the inferential conclusion do not change.
- Correct E13's parenthetical re-computation from 1.876 to 1.8752 percentage
  points. Direct per-item, equal-environment aggregation gives
  1.8752228163993; the stored 1.875 difference is correct at its stated precision.
- Point current readers to SUBMISSION.md. Preserve earlier submissions as
  historical records. Exclude the stale Markdown/DOCX assembler, alternate
  chapter, and review slides from the current package.
- Wrap the overview table's source paths and give Appendix A's repeated
  same-stack heading a unique PDF destination. These are layout/link fixes.
- Harden thesis/figure builds and the older scientific audit so failed checks
  cannot delete or silently replace published PDFs. Add focused regression tests.
- Restore the experiment smoke tests' real source-directory path and reject
  empty input sets. Remove unused import-time CodeCarbon tracking so importing
  the TRL module neither starts telemetry nor creates emissions artifacts.
- Repair 18 pre-existing formatter failures using the locked Ruff formatter;
  no research behavior is intentionally changed by those formatting edits.
- Update the draft AI-tool declaration to include this submission preparation
  and the independent Gemini 3.1 Pro code reviews. Author approval is still needed.

The prepared certificate and AI-tool declaration still require the author's and
institution's approval. No portal upload or guide signature was performed.

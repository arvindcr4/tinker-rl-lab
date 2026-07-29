# Autoresearch lineage: persuasive NeurIPS 36320 rebuttal

## Dispatch

- Mode: orchestrator, single-pass `reason` dispatch.
- Goal archetype: decide-design.
- Constraint: maximize reviewer persuasion without publishing, posting, or expanding the evidence beyond what the repository supports.
- Incumbent: `zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`.

## Round 1: adversarial reviewer attack

Three independent personas represented the two movable score-3 reviewers, the Strong Reject, and the Area Chair. All returned `REVISE`:

- the acceptance case was buried under withdrawals;
- the use case lacked explicit inputs, output, and changed decision;
- 4G4H's concrete per-run reward/ZVF/GU/held-out request remained unanswered;
- the Strong Reject response made the application sound invented after review; and
- the AC comment lacked a copyable positive rationale.

The responses were rewritten to foreground reviewer-recognized value, anchor the practitioner aim in the submitted introduction, operationalize it as a bounded evidence check, provide the exact 40-cell telemetry table to 4G4H, and make E1 explicitly post-submission and nonessential to acceptance.

## Round 2: blind evidence audit

The evidence judge rejected the E1 `DISAPPEARS` claim. The frozen aggregate used a normal-approximation MDE of 0.867 pp; the finite-sample paired-t MDE is 1.012 pp, above the preregistered 1 pp gate. The code also does not apply the preregistered multiplicity step.

The rebuttal was corrected to treat all four E1 comparisons as inconclusive. It also changed direct Tinker-linkage language to plausible source-run alignment, disclosed that DAPO changes both the upper clipping bound and filtering/refill behavior, and stated that the private receipts are not independently reviewer-verifiable without an anonymized artifact.

## Round 3: held-out verification

- Evidence judge: `PASS` after re-read; all 40 telemetry cells match the frozen records.
- AC judge: `PASS` after two copy edits; all four questions are answered and the use case is a bounded operationalization rather than a submitted deployment claim.
- Score-change judge: `PASS` after surgical edits; retaining the 4G4H table was preferred because it directly answers that reviewer's request.

Predicted movement is strongest for PYUJ, plausible but uncertain for 4G4H, and limited for 9kjk. The AC comment now supplies a coherent borderline-acceptance rationale without asking the AC to credit private post-submission evidence.

## Winner

`persuasive-bounded-v11`, in `zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`.

No OpenReview note was edited or posted.


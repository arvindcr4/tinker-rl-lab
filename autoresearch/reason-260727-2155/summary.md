# Autoresearch summary: NeurIPS 36320 rebuttal

## Winner

The winning strategy is a credibility-first replacement of all three reviewer
responses and the confidential AC comment. It explicitly corrects the
partially backfilled five-seed claim, answers the AC's four questions, preserves
only two claims from the reviewed paper, and treats the completed E1 audit as
separate post-submission context rather than retrospective repair.

The character-counted text is in
`zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`.

## Convergence

Ten evidence/reason rounds were used. The initial scope-narrowing candidate required
provenance corrections. A full-response replacement then won, followed by a
three-judge claim audit and rebuttal-guide polish. W&B access materially reopened
the evidence question. The W&B-expanded candidate was revised because it risked
asking the AC to accept a different paper. The final bounded candidate received
three independent PASS verdicts for evidence/statistics, rebuttal strategy, and
skeptical-reviewer/AC robustness. Subsequent read-only Tinker and Hugging Face
audits refined provenance without changing the bounded strategy.
The final acceptance-framing pass preserved every evidentiary limit while
replacing unnecessarily surrendering use-inspired language with one supported
application: pre-adoption auditing of RLVR algorithm claims.

## Key findings

- The exact reviewed target is OpenReview forum `CXbcYe69BQ`, submission 36320.
- The five W&B summary pairs arithmetically reproduce .926 versus .921 and a
  recomputed paired `p=.3739`, but seed pairs 42--44 are zero-runtime W&B
  backfills. The live seed-45/46 records have plausible source-run alignment,
  but no metadata uniquely maps the backfills. The five-seed transfer claim is
  withdrawn.
- W&B confirms that Qwen PPO .350 and .225 are two distinct Modal runs, not two
  summaries of one trace. The row is quarantined.
- The submitted early rule remains 2 positives among 22 heterogeneous runs,
  both tool-use; early reward separates the same failures and continuous ZVF
  ranking is weak.
- The submitted runner lacks the PPO importance ratio/clip, frozen-reference
  KL, and completion-only mask. Only the zero centered reward-contrast term is
  claimed to transfer algebraically.
- The submitted held-out result is inconclusive: base 164/200; trained
  checkpoints 166, 165, 161, 168, and 173; seed-level `p=.256`; all five
  per-seed McNemar comparisons non-significant.
- E1 is a separate post-submission 40-unit same-stack audit using a clipped,
  completion-masked TRL GRPO baseline with `beta=0`. A later statistical audit
  found that the frozen DAPO verdict used a normal-approximation MDE and omitted
  the preregistered multiplicity step. The conservative paired-t MDE is 1.012
  pp, just above the 1 pp gate, so all four comparisons are treated as
  inconclusive. None collapsed. E1 does not resolve runner transfer,
  reference-KL dependence, early-rule validity, pre/post capability, or
  controller utility.
- A fresh authenticated Hub audit resolves all 40 frozen E1 repositories and
  pinned commits. Every unit has all six checkpoint trainer states, a final
  adapter, and a 500-row final manifest whose hash matches both the local
  manifest and frozen campaign receipt. Four GRPO units lack only the
  evaluation-resume sidecar. The private evidence is not independently visible
  to reviewers without an anonymized artifact.
- The use-inspired claim is now explicitly bounded to a pre-adoption audit:
  require a controlled stack, fixed held-out evaluation, and reconciled
  provenance before accepting an RLVR algorithm claim. No controller,
  deployment, or compute-saving claim is made.

## Final response sizes

- PYUJ: 4,854 characters of 10,000.
- 4G4H: 5,661 characters of 10,000.
- 9kjk: 4,733 characters of 10,000.
- Confidential AC comment: 4,627 characters of 5,000.

## Recommendation beyond rebuttal

The best near-term ACL/ARR base remains the E1 same-stack audit design, but its
statistical pipeline must be repaired and independently reviewed before any
equivalence verdict is used. The May heterogeneous study should not be
expanded. A use-inspired ZVF controller paper requires a separate prospective
intervention measuring charged rollout savings under a frozen held-out
non-inferiority margin.

No OpenReview note was edited or posted.

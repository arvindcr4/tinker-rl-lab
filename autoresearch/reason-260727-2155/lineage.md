# Rebuttal autoresearch lineage

Task: Frame the NeurIPS author response for maximum acceptance probability by inspecting the exact OpenReview submission, reviews, author responses, and repository experiments, without posting or publishing.

Domain: research hypothesis and content/writing  
Mode: convergent  
Maximum rounds: 8  
Convergence target: 3 consecutive wins

The installed `autoresearch` package lacked its documented `scripts/orchestrate.sh`; routing was therefore applied directly from `references/orchestrator-routing.md`. The goal maps to the single-pass `decide-design -> reason` route.

## Evidence boundary

- Exact conference forum: `CXbcYe69BQ`, submission 36320.
- Exact reviewed PDF downloaded through the authenticated forum: 17 pages,
  2,996,794 bytes, SHA-256
  `b15ac7e5f673473cf8edc07634f6acbd9fcd54b9f0d5d1f75b106565a174a62d`.
- The initially discovered forum `ObvFxM58Kb` is a different NeurIPS workshop
  proposal and was excluded.
- Three reviewer rebuttals and one confidential AC comment were read from the
  authenticated forum. All are writable as of 2026-07-27.
- Reviewer-response due date: 2026-07-28 17:29 IST, 10,000 characters each.
- AC-comment due date: 2026-08-04 16:29 IST, 5,000 characters.
- No OpenReview content was edited or posted.

## Round 1 — scope-narrowing draft

### Author candidate

The first author draft used the correct high-level strategy: concede the
two-positive rule, distinguish the runner from canonical GRPO, describe the
82.0% to 83.3% held-out result as inconclusive, remove heterogeneous means,
withdraw use-inspired framing, and restructure around a claim-to-evidence map.

### Critic

The critic found two stop-ship errors:

1. `0.225` and `0.350` were not two aggregations of one Qwen PPO trace. They
   came from different experiments that were misattributed to one row.
2. The 99.0%/99.2% arithmetic comparison was neither estimator-only nor validly
   paired: prompt exposure and evaluation draws differ between arms.

The critic also verified that the asserted Qwen3-8B 92.6%/92.1%, `p≈.37`
control has no supporting raw artifact.

### Judge

Verdict: revise. A candid scope-narrowing response was preferred over a
post-submission rescue, but the remaining provenance errors were acceptance-
damaging. The judge's hard caps rejected equivalence claims from nonsignificance,
generalized triage from two positives, canonical-GRPO labeling, and use-inspired
framing without an evaluated intervention.

## Round 2 — exact OpenReview replacement strategy

### New evidence

Authenticated OpenReview inspection established that the live record contains
three reviewer responses of 8,946, 9,196, and 9,702 characters plus a 4,993-
character AC comment. All four repeat the unsupported 92.6%/92.1% result.
Revision history makes silent deletion inappropriate; explicit correction is
required.

Repository checks added four material findings:

- the submitted tool-use rows have reward 0 and ZVF 1, so they are collapse,
  not high-reward saturation;
- the fused AUROC 0.929 used synthetic/imputed anchors, and the later real-only
  threshold table does not support the advertised stop bands;
- the paper defines GU as Gradient Utilization, not Group Uniformity;
- the `p=.256` GSM8K result is a seed-level one-sample test, distinct from the
  per-seed McNemar comparisons.

### Author candidate

The second author candidate produced four short replacements centered on two
surviving claims: homogeneous centered rewards zero the submitted runner's
reward-contrast term, and online reward, held-out capability, proxy reward, and
algorithm labels must be separated. It explicitly withdrew the unsupported
matched control, the pooled heterogeneous mean, predictive/stopping language,
canonical transfer, held-out improvement, and use-inspired application.

### Critic

The critic issued a stop-ship verdict for the live notes and supplied a
claim-level checklist. Retain only the reward-derived centering identity,
mixed-group idealization, descriptive ZVF/GU definition, exact inconclusive
held-out record, runner disclosure, and normative reporting proposal. Remove
the unsupported matched control, confounded arithmetic rescue, AUROC/threshold
claims, compute savings, headroom claims, and estimator-ordering claim.

### Judge

Verdict: `WIN-90`. Fully replace all four notes rather than patching them.
Score: evidence 38/40, directness 19/20, clarity 13/15, feasibility 10/10,
likely score change 10/15. The judge expects credibility recovery with the AC
and possible movement of a score-3 reviewer, but not a guaranteed flip of the
strong reject.

### Incumbent

`zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`

## Round 3 — three-judge audit

Three independent judges reviewed the claim-level evidence and live-response
strategy. They required five changes before accepting the candidate:

- remove the 12-run group-size table from the live rebuttal because it would
  repeat the fragmentation problem;
- qualify `p^G+(1-p)^G` as a conditional-i.i.d. Bernoulli model rather than a
  general identity for correlated rollouts;
- disclose that three exact-ID/model conflicts remain unresolved;
- remove threshold/AUROC/controller language based on synthetic or imputed
  anchors; and
- keep the future factorial subordinate to one primary operational question.

After these edits the candidate received a winning aggregate assessment and
remained within all OpenReview character limits.

## Round 4 — rebuttal-guide polish

The user-supplied Foerster/Rocktäschel rebuttal guide, Neel Nanda advice, and
the ML-review guide changed the presentation in four ways: lead with strengths
the reviewers themselves recognized, answer the decisive concern before
secondary issues, give the AC copyable bounded language, and ask explicitly
for score reconsideration. Three judges then reported no stop-ship issue.

## Round 5 — W&B evidence reopening

Read-only W&B inspection materially changed the provenance picture:

- project `neurips36320-matched-grpo` arithmetically reproduces the reported
  .926 versus .921 means and recomputed `p=.3739`;
- seed pairs 42--44 are zero-runtime backfills, while 45--46 are live records;
- the checked records do not contain source commits, captured training source,
  checkpoints, per-item evaluation traces, or upstream source-run IDs, and
  seed 42 uses a different batch size;
- the Qwen PPO .350 and .225 summaries are two distinct Modal runs, not two
  aggregations of one trace; and
- the separate E1 campaign is complete at 40/40 remotely reconciled units.

The first W&B-expanded draft treated E1 too much like a replacement primary
study. Strategy and skeptical-reviewer judges rejected that framing.

## Rounds 6--8 — bounded W&B final

The final candidate keeps only the two submitted-paper claims, withdraws the
five-seed comparison as transfer evidence, and labels E1 as separate
post-submission feasibility evidence. It discloses that the E1 baseline is
clipped and completion-masked with `beta=0`; therefore E1 does not answer
reference-KL dependence. The DAPO result is reported using its full
preregistered compound rule: paired-bootstrap 90% CI
`[-0.35,+0.575]` percentage points inside the +/-1-point margin and achieved
80%-power MDE `0.867` points. The 95% interval remains descriptive and is not
used alone to claim equivalence.

Three independent final judges returned PASS: one for evidence/statistics, one
for rebuttal strategy, and one from a skeptical Strong-Reject/AC perspective.
Final live-section character counts are 4,661 (PYUJ), 4,231 (4G4H), 3,347
(9kjk), and 4,303 (AC), below the 10,000/5,000 limits.

## Round 9 — Tinker and Hugging Face provenance refresh

Read-only Tinker history directly corroborated the four live seed-45/46 runs,
each with 31 sampler checkpoints, and exposed three plausible earlier complete
pairs. Because neither the W&B backfills nor the Tinker API record cross-system
identifiers or treatment metadata, this did not restore the five-seed transfer
claim.

A separate authenticated Hugging Face audit resolved all 40 frozen E1
repository/commit pairs. Every pinned commit contains checkpoint trainer states
at steps 5, 10, 15, 20, 25, and 30, a final adapter, and a final 500-row
manifest. All 40 remote manifest hashes match the local manifests and frozen
campaign receipts. Four GRPO units lack the separate evaluation-resume sidecar,
but retain the complete final trace. The private-Hub evidence strengthens
provenance only; it is not reviewer-verifiable without an anonymized artifact
and does not alter E1's evidentiary scope.

## Round 10 — acceptance-oriented application framing

The Strong-Reject response and AC comment previously conceded the entire
use-inspired category even though E1 executes one concrete operator decision:
whether an algorithm claim survives controlled reimplementation and held-out
evaluation before adoption. The final pass now names that narrow application
while retaining the hard boundaries: E1 is separate post-submission evidence,
not retrospective repair, and the paper makes no stopping-controller,
deployment, or compute-saving claim.

Final paste-ready section sizes are 4,661 (PYUJ), 4,231 (4G4H), 3,700 (9kjk),
and 4,587 (AC), below the 10,000/5,000 limits.

## Rounds 11--12 — persuasion and independent evidence audit

Three adversarial reviewer personas found that the evidence-safe draft still
buried its acceptance case. The responses were rewritten to lead with the
reviewer-recognized audit contribution, anchor the practitioner aim in the
submitted introduction, define the audit's inputs and decision use, give 4G4H
the requested exact per-seed reward/ZVF/GU/held-out table, and give the AC a
copyable positive rationale.

A fresh evidence judge then found a stop-ship statistical issue in E1: the
frozen aggregator used a normal-approximation 80%-power MDE of 0.867 pp and did
not apply the preregistered multiplicity step. The finite-sample paired-t MDE is
1.012 pp, just above the 1 pp equivalence margin. The rebuttal now treats DAPO
and all three other E1 comparisons as inconclusive, identifies E1 as private
post-submission feasibility evidence, and makes acceptance independent of it.
The judge subsequently returned PASS.

Final paste-ready section sizes are 4,854 (PYUJ), 5,661 (4G4H), 4,733 (9kjk),
and 4,627 (AC), below the 10,000/5,000 limits.

## Final artifacts

- `zvf-program/flagship/paper/NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`
- `zvf-program/flagship/paper/MAY_REVIEW_RESPONSE_AND_ACL_PLAN.md`
- `autoresearch/reason-260727-2155/WANDB_EVIDENCE_AUDIT.md`
- `autoresearch/reason-260727-2155/TINKER_HISTORY_AUDIT.md`
- `autoresearch/reason-260727-2155/HUGGINGFACE_EVIDENCE_AUDIT.md`
- `autoresearch/reason-260727-2155/openreview_submission_CXbcYe69BQ.pdf`

No OpenReview note was edited or posted.

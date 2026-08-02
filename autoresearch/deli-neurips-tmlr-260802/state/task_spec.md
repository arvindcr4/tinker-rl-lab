# Task specification

## Objective

Determine the truth of the scientific claims in the current Tinker RL manuscript
family, then produce the strongest evidence-bounded paper that is genuinely ready
for NeurIPS or TMLR review.

## Primary research question

What is the strongest original claim about reward contrast, zero-variance groups,
GRPO-family implementation semantics, or claim-to-run auditing that the checked-in
theory, code, experiments, and provenance can support without importing missing
evidence?

## Scope

- Exact NeurIPS 2026 main-track submission and OpenReview response record.
- July flagship methods/reproducibility manuscript and review bundle.
- E1 same-stack audit and its statistical re-audit.
- S1 objective-conformance work, r4-2 pilot, and next-submission experiments.
- Public literature index only where it supplies verified prior-art context.
- Live external sources for venue policy and current prior art.

## Non-negotiable evidence rules

1. Tests, preflights, receipts, and internal consistency are not learning effects.
2. A missing cell remains missing; failed or partial runs never enter aggregates.
3. Online reward, verifier/proxy reward, held-out capability, and cost stay separate.
4. A non-significant difference is not equivalence.
5. Heterogeneous models, tasks, runners, evaluators, or seed structures are not pooled
   unless the estimand explicitly permits it.
6. Every retained numerical statement must resolve to an immutable source or be
   recomputed from checked-in raw data.
7. Citation metadata and novelty claims must be checked against primary sources.
8. The final paper must state its non-claims as clearly as its claims.

## Milestones

1. Build a claim-to-evidence ledger for all candidate manuscripts.
2. Recompute every high-risk quantitative claim independently.
3. Falsify or quarantine claims that lack provenance, power, or matched controls.
4. Compare surviving contributions against current primary prior art.
5. Select one manuscript spine; do not merge incompatible paper stories.
6. Revise source, tables, abstract, limitations, and artifact appendix.
7. Compile and run all relevant validators from a clean extraction where practical.
8. Obtain an independent evidence-chain audit and venue-style review.
9. Produce a submission package plus an explicit ready/not-ready decision.

## Success criteria

- One canonical manuscript and one canonical PDF.
- Claim ledger with source paths, hashes, status, and allowed inference.
- No unresolved contradiction in any headline result.
- No causal or capability claim supported only by tests, receipts, or toy fixtures.
- Reproducible build with no undefined references or clipped content.
- Artifact verification commands pass from the documented environment.
- Independent reviewer can summarize the method, result, and boundary accurately.
- Venue policy, anonymity, overlap, and dual-submission checks are documented.

## Stop conditions

- If the evidence cannot support a NeurIPS/TMLR empirical claim, publishability is
  assessed for a narrower methods, reproducibility, or negative-results paper.
- If even that claim fails, the final output is a documented no-go with the minimum
  experiment needed to change the decision.


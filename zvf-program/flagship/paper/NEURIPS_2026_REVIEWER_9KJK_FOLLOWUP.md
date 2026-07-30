# Reviewer 9kjk follow-up: evidence-bounded final response

**Prepared:** 2026-07-30<br>
**Submission:** 36320, OpenReview forum `CXbcYe69BQ`<br>
**Decision:** if the discussion remains open, post only the acknowledgement below. Do not ask this reviewer for score reconsideration and do not promise a camera-ready rewrite as a remedy for missing experiments.

The follow-up changes the correct response strategy. The reviewer has explicitly rejected three arguments in the previous response: that a promised restructuring can be assessed now, that moving artifact paths solves absent numerical results, and that an internal ML adoption audit establishes a use-inspired contribution for users outside the NeurIPS community. Repeating those arguments would not answer the review.

## Postable reply

<!-- POSTABLE_REPLY_START -->
Thank you for the clarification. We agree that the concerns remain unresolved in the reviewed submission, and we understand why your score remains unchanged.

A promised structural rewrite cannot be evaluated in the current review. Artifact paths can document provenance, but filenames, appendix reorganization, and prose changes do not supply missing numerical results or experiments.

The corpus was assembled retrospectively from available completed runs, with cells differing in model, task, sampler, runner, and evaluator, rather than from a predeclared complete matrix. HumanEval and MATH have no numerical main-result evaluation comparable to GSM8K, and the synthetic tool-use evidence is only two boundary cases. The missing cells are missing evidence, not implicit replications. We withdraw all-corpus, cross-corpus, and task-comparative interpretations.

Some single-seed cells were retained after run failures as exploratory cases during paper construction. That explains the record but cannot justify comparative inference. The Qwen PPO/GRPO source values conflict and are quarantined; the Llama comparison is single-seed and backend-confounded. We withdraw the stack-sensitivity, PPO/GRPO, algorithm-ranking, and other single-seed comparative claims rather than treating optimization steps or selected checkpoints as replications.

The fixed GSM8K evaluation used one base checkpoint at 164/200 and five trained checkpoints at 166, 165, 161, 168, and 173/200, yielding a trained mean of 83.3% versus 82.0% for the base. The reported `p=.256` is a one-sample seed-level test of the five trained-checkpoint accuracies against the fixed base accuracy, not a paired-seed test; all five per-seed item-paired McNemar tests were nonsignificant. This result is inconclusive and does not establish capability improvement.

The retrospective early-triage set contained only 2/22 collapsed runs, both tool use, and those same two cases also had zero early reward. In both cells, online reward was 0, the zero-variance fraction (ZVF) was 1, and its reported complement, gradient utilization (GU), was 0. This is two-case descriptive concordance and all-wrong homogeneity, not evidence of diagnostic performance or a validated stopping policy. For the submitted centered reward-contrast term, homogeneous rewards analytically make that term zero; these two cases do not establish a learning or intervention outcome.

The submission evaluated neither a pre-existing external-user need nor an external-user decision and outcome, so we withdraw the use-inspired contribution type. What remains is a bounded methodology/reproducibility observation: the exact within-run consequence of homogeneous rewards and the need to keep reward contrast, online reward, held-out capability, and algorithm labels separate. This is not an empirically validated method or a demonstrated real-world use case.

Any future broader empirical claim would require predeclared estimands and in-scope cells, explicit missing-cell accounting, fixed held-out evaluations, and replication and evaluation sample sizes prospectively justified by power or precision targets. Renewing the use-inspired designation would additionally require a prospectively tested external-user decision and outcome. Those requirements do not alter the reviewed record. Thank you for identifying where our prior response exceeded the evidence.
<!-- POSTABLE_REPLY_END -->

## Why this is the defensible response

- It directly answers both previously unanswered “why” questions: the study was retrospective rather than factorial, and single-seed rows survived because of run failures and exploratory paper construction. It does not present those facts as scientific excuses.
- It gives actual numerical results in the response and does not substitute artifact paths for evidence.
- It withdraws the contribution-type claim the reviewer rejected instead of relabeling an ML-internal workflow as an external use case.
- It does not import the post-submission E1 matrix into the reviewed record. E1 is useful for a future same-stack audit paper, but it cannot make the current rewrite reviewable.
- It preserves only the exact signal-availability observation and the separation of online reward, proxy telemetry, held-out capability, and algorithm labels.

## Resource-derived design correction

The [RLHF Book policy-gradient chapter](https://rlhfbook.com/c/06-policy-gradients) makes the within-prompt sampler, group contrast, advantage construction, clipping, and loss aggregation part of the realized treatment. Its [over-optimization](https://rlhfbook.com/c/14-over-optimization) and [evaluation](https://rlhfbook.com/c/16-evaluation) chapters require proxy reward to be separated from held-out quality and evaluation choices. The book's practical variance guidance motivates independent training-seed replication rather than treating optimization steps or checkpoints as replicates.

The pinned [Harvard CS2824](https://harvard-cs2824-s26.github.io/) transfer adds the missing theoretical discipline: stationarity claims require coverage and distribution-mismatch assumptions, and optimization, estimation, approximation, and verifier errors must remain separate. Applied here, an unpopulated corpus-task cell is a coverage failure, not evidence that an observed relationship transfers.

Those sources improve the next design; they do not retroactively strengthen the reviewed experiments.

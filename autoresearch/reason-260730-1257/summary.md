# Reviewer 9kjk response verification

Status: `CONVERGED`

The three-judge panel converged in 3 rounds, below the 4-round ceiling. The current reply is evidence-bounded and directionally correct, but the refined Candidate AB safely improves it by naming the `p=.256` test, describing early triage as two-case descriptive concordance, defining ZVF/GU, replacing the universal five-seed prescription with prospective power/precision justification, and scoping external-user validation specifically to any renewed use-inspired claim.

Judge agreement was unanimous in rounds 1 and 2. In round 3, two judges retained the incumbent verbatim and one selected a cosmetic polish from the same response family. Exact-label agreement with the panel verdict was 8/9 votes (88.9%); substantive-family agreement was 9/9 (100%).

## Winning postable response

Thank you for the clarification. We agree that the concerns remain unresolved in the reviewed submission, and we understand why your score remains unchanged.

A promised structural rewrite cannot be evaluated in the current review. Artifact paths can document provenance, but filenames, appendix reorganization, and prose changes do not supply missing numerical results or experiments.

The corpus was assembled retrospectively from available completed runs, with cells differing in model, task, sampler, runner, and evaluator, rather than from a predeclared complete matrix. HumanEval and MATH have no numerical main-result evaluation comparable to GSM8K, and the synthetic tool-use evidence is only two boundary cases. The missing cells are missing evidence, not implicit replications. We withdraw all-corpus, cross-corpus, and task-comparative interpretations.

Some single-seed cells were retained after run failures as exploratory cases during paper construction. That explains the record but cannot justify comparative inference. The Qwen PPO/GRPO source values conflict and are quarantined; the Llama comparison is single-seed and backend-confounded. We withdraw the stack-sensitivity, PPO/GRPO, algorithm-ranking, and other single-seed comparative claims rather than treating optimization steps or selected checkpoints as replications.

The fixed GSM8K evaluation used one base checkpoint at 164/200 and five trained checkpoints at 166, 165, 161, 168, and 173/200, yielding a trained mean of 83.3% versus 82.0% for the base. The reported `p=.256` is a one-sample seed-level test of the five trained-checkpoint accuracies against the fixed base accuracy, not a paired-seed test; all five per-seed item-paired McNemar tests were nonsignificant. This result is inconclusive and does not establish capability improvement.

The retrospective early-triage set contained only 2/22 collapsed runs, both tool use, and those same two cases also had zero early reward. In both cells, online reward was 0, the zero-variance fraction (ZVF) was 1, and its reported complement, gradient utilization (GU), was 0. This is two-case descriptive concordance and all-wrong homogeneity, not evidence of diagnostic performance or a validated stopping policy. For the submitted centered reward-contrast term, homogeneous rewards analytically make that term zero; these two cases do not establish a learning or intervention outcome.

The submission evaluated neither a pre-existing external-user need nor an external-user decision and outcome, so we withdraw the use-inspired contribution type. What remains is a bounded methodology/reproducibility observation: the exact within-run consequence of homogeneous rewards and the need to keep reward contrast, online reward, held-out capability, and algorithm labels separate. This is not an empirically validated method or a demonstrated real-world use case.

Any future broader empirical claim would require predeclared estimands and in-scope cells, explicit missing-cell accounting, fixed held-out evaluations, and replication and evaluation sample sizes prospectively justified by power or precision targets. Renewing the use-inspired designation would additionally require a prospectively tested external-user decision and outcome. Those requirements do not alter the reviewed record. Thank you for identifying where our prior response exceeded the evidence.

## Actionable defects in the current reply

1. `p=.256` is called only “seed-level”; the reply should identify it as the one-sample test over five trained-checkpoint accuracies against the fixed base, distinct from the five item-paired McNemar tests.
2. “Early reward identified the same two failures” can sound like validated diagnostic performance. “The same two cases also had zero early reward” plus “two-case descriptive concordance” is safer.
3. “At least five independent training seeds” is an unsupported universal prescription in this reply. Seed and evaluation-sample counts should be prospectively tied to a stated effect and power or precision target.
4. ZVF and GU should be defined once for a standalone response.
5. Future external-user validation is necessary for a renewed use-inspired claim, not for every possible methodology or capability claim.

No post-submission E1 result was used, no score reconsideration was requested, and no primary manuscript or rebuttal file was edited.

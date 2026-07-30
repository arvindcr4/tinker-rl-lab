# Blind judge packet

Task: select the safest postable reply to Reviewer 9kjk's follow-up. The reviewer says that a promised structural rewrite is not reviewable, filenames are not results, the use-inspired framing lacks an external real-world user, the corpus-task analyses are incomplete, several evaluations are single-seed, and the score is unchanged.

Domain: research content/writing. Evaluate clarity, accuracy, engagement, completeness, and actionability. Rank all three labels in each round and select exactly one winner. Do not infer candidate provenance from wording.

Evidence boundary: do not use post-submission E1 as reviewed evidence. The reviewed surviving numbers are one fixed base GSM8K checkpoint at 164/200 and five trained checkpoints at 166, 165, 161, 168, and 173/200; `p=.256` is a one-sample seed-level test over those five trained-checkpoint accuracies against the fixed base accuracy, and all five per-seed item-paired McNemar tests are nonsignificant. Early triage contains 2/22 collapsed runs, both tool use; early reward selects the same two. Both tool-use cells have online reward 0, ZVF 1, and GU 0. The Qwen PPO/GRPO source values conflict and are quarantined. The Llama comparison is single-seed and backend-confounded. HumanEval and MATH lack comparable numerical main results. Cross-corpus and single-seed comparative claims and the use-inspired type must be withdrawn. Do not ask for score reconsideration.

## Round 1

### Label-X

Thank you for the clarification. We agree that the reviewed submission does not justify score reconsideration. A structural rewrite, appendix reorganization, or artifact filenames cannot substitute for reviewer-visible results.

We withdraw the following concrete claims: that training produced a GSM8K capability improvement; that early reward or reward homogeneity is a validated collapse diagnostic or stopping rule; that the findings generalize across tasks or corpora; that PPO, GRPO, models, or training stacks can be ranked from these results; that the work demonstrates a methodology or reproducibility contribution; and that it qualifies as use-inspired research.

For GSM8K, the deterministic base checkpoint scored 164/200, while five trained checkpoints scored 166, 165, 161, 168, and 173/200. The reported `p=.256` comes from a one-sample seed-level test treating those five trained-checkpoint accuracies as observations against the fixed base accuracy; it is not a paired-seed comparison. Every per-seed, item-paired McNemar test was also nonsignificant. These results are inconclusive and do not establish improvement.

The early-triage observation is strictly retrospective: 2 of 22 runs were collapsed, both were tool-use runs, and early reward marked exactly those same two runs. In those two cells every online reward was 0, so there was no within-group reward contrast; the table represented this as ZVF=1 and GU=0. With only two positive cases and no held-out validation, this coincidence supplies neither diagnostic-performance evidence nor cross-task support.

The corpus was assembled retrospectively from completed runs after API failures and other missing cells, across differing models, tasks, samplers, runners, and evaluators. That missingness cannot be repaired by filenames or treated as implicit replication. HumanEval and MATH lack comparable numerical main results, and the synthetic evidence is limited to the two tool-use cases above. The conflicting Qwen PPO/GRPO values remain quarantined; the Llama comparison is single-seed and backend-confounded. Accordingly, no all-corpus, task-comparative, or algorithm-comparative conclusion survives.

Because the submission evaluates neither a pre-existing external user need nor an external-user decision and outcome, we withdraw the use-inspired designation. What remains is only a set of descriptive observations and an account of evidentiary limitations, not a validated contribution claim. Any future study should predeclare its estimand and task-by-corpus matrix, report all cells and missingness, choose seed and evaluation-sample counts from a prospective power or precision analysis tied to a stated effect of interest, use fixed held-out evaluations, and prospectively test an external-user outcome before making broader claims.

### Label-Y

Thank you for the clarification. We agree that these concerns remain unresolved in the reviewed submission, and we understand why the score is unchanged.

A promised structural rewrite cannot be assessed during review. Artifact paths can document provenance, but neither filenames nor prose changes supply missing numerical results or experiments.

The corpus was assembled retrospectively from available completed runs rather than as a predeclared factorial study, and its cells differ in model, task, sampler, runner, and evaluator. HumanEval and MATH have no numerical main-result evaluation comparable to GSM8K, while the synthetic tool-use evidence consists of two boundary cases. Missing cells therefore remain missing evidence. We withdraw all-corpus, cross-corpus, and task-comparative interpretations.

The single-seed cells were retained after run failures as exploratory cases during paper construction. That history does not justify comparative inference. The Qwen PPO/GRPO values conflict across sources and are quarantined, while the Llama comparison is single-seed and backend-confounded. We therefore withdraw the stack-sensitivity, PPO/GRPO, algorithm-ranking, and other single-seed comparative claims rather than treating steps or checkpoints as replications.

The fixed GSM8K evaluation used one base checkpoint at 164/200 and five independently trained checkpoints at 166, 165, 161, 168, and 173/200, for a trained mean of 83.3% versus 82.0% for the base. The reported `p=.256` is a one-sample seed-level test over the five trained-checkpoint accuracies against the fixed base accuracy, not a paired-seed test; all five per-seed item-paired McNemar tests were nonsignificant. This is inconclusive and does not establish capability improvement.

In the retrospective early-triage set, only 2/22 runs were collapsed, both tool use, and those same two cases also had zero early reward. Both tool-use cells had online reward 0, ZVF=1, and GU=0. This is two-case descriptive concordance and observed all-wrong homogeneity, not diagnostic-performance evidence or validation of a stopping policy. For the submitted centered reward-contrast term, homogeneous rewards analytically make that term zero, but the two cases do not establish a learning or intervention outcome.

The submission also evaluated neither a pre-existing external user need nor an external-user decision and outcome. We therefore withdraw the use-inspired contribution type. The surviving scope is limited to the exact within-run analytical observation and a methodology/reproducibility reporting principle that reward contrast, online reward, held-out capability, and algorithm labels must not be treated as interchangeable; it is not an empirically validated method or demonstrated real-world use case.

A future study would need predeclared estimands and in-scope cells, explicit accounting for every missing cell, fixed held-out evaluations, replication and evaluation sample sizes prospectively justified by power or precision targets, and a tested external-user decision and outcome before making broader claims. These requirements do not repair the reviewed record. Thank you for identifying where our previous response exceeded the evidence.

### Label-Z

Thank you for the clarification. We agree that these concerns remain unresolved in the reviewed submission.

**Structure and results.** A promised rewrite cannot be assessed during review, and moving filenames to an appendix would not supply missing results. Our previous response conflated an editorial repair with an evidentiary repair. The latter would require a reviewer-visible revised paper and experiments, which are not part of the reviewed record.

**Why the corpus-task matrix is incomplete.** The corpus was assembled retrospectively from available completed runs with different models, tasks, samplers, runners, and evaluators; it was not designed as a full factorial study. Consequently, the omitted cells are missing evidence, not implicit replications or results recoverable from an artifact path. In particular, the reviewed paper contains no numerical main-result evaluation for HumanEval or MATH comparable to its GSM8K analyses, and the synthetic tool-use evidence consists only of two boundary cases. We therefore withdraw cross-corpus and all-corpora interpretations.

**Why some evaluations are single-seed.** Several managed-run cells were retained after API failures and were treated as exploratory case studies during paper construction. That practical history is not a scientific justification for comparative claims. The single-seed stack-sensitivity and PPO/GRPO rows cannot support a population comparison and should not have carried inferential weight. We withdraw those comparisons rather than treating step-level observations as replications.

**Numerical boundary of the reviewed evidence.** The fixed GSM8K check used one base checkpoint at 164/200 and five trained checkpoints at 166, 165, 161, 168, and 173/200; the trained mean was 83.3% versus 82.0% for the base, with a seed-level `p=.256` and no significant per-seed McNemar comparison. This is inconclusive. The early-triage set had only 2/22 collapsed runs, both tool-use, and early reward identified the same two failures. In those two tool-use runs, online reward was 0, ZVF=1, and GU=0; this is observed all-wrong homogeneity, not a validated stopping policy. The Qwen PPO/GRPO row has conflicting source values and is quarantined; the remaining Llama comparison is single-seed and backend-confounded. These results support no algorithm ranking, capability gain, or cross-corpus generalization.

**Contribution type.** We also accept that the submission did not evaluate a pre-existing need of users outside the NeurIPS community or a concrete real-world outcome. We therefore withdraw the use-inspired contribution type. The surviving scope is a methodology/reproducibility observation about reward contrast and evidence separation, not a demonstrated real-world use case.

Accordingly, we do not claim that prose changes resolve the reviewed paper. A future submission must use a predeclared in-scope matrix, report every cell or an explicit missing-cell rationale, include at least five independent training seeds per primary cell, use fixed held-out evaluations, and test an external-user decision and outcome before claiming a use-inspired contribution. Thank you for identifying where our previous response remained incomplete.

## Round 2

Round 1's winning approach is carried forward and refined.

### Label-X

Thank you for the clarification. We agree that the concerns remain unresolved in the reviewed submission, and we understand why your score is unchanged.

A structural rewrite cannot be evaluated during review, and filenames are not substitutes for reported results. Prose changes cannot repair missing experiments.

HumanEval and MATH have no numerical main results comparable to the GSM8K evaluation, and the synthetic tool-use evidence comprises only two boundary cases. Missing cells therefore remain missing evidence. We withdraw the cross-corpus and all-corpus claims. We also withdraw the PPO/GRPO and stack-sensitivity comparisons: the Qwen values conflict across sources and are quarantined, while the Llama comparison is single-seed and backend-confounded.

GSM8K accuracy was 164/200 for the base checkpoint and 166, 165, 161, 168, and 173/200 across five trained checkpoints (83.3% mean versus 82.0% base). The seed-level test gave `p=.256`, and every per-seed McNemar comparison was nonsignificant. This is inconclusive. In early triage, 2/22 runs collapsed, both on tool use, and the same two cases also had zero early reward. For those two cells, online reward was 0, ZVF was 1, and GU was 0. This is an observation of all-wrong homogeneity in two cases, not evidence for a validated stopping policy.

The submission does not establish an external real-world user need or outcome, so we withdraw the use-inspired contribution type. After these withdrawals, the reviewed record supports only limited descriptive methodology and reproducibility observations within the reported runs, not algorithm ranking, capability improvement, cross-corpus generalization, or real-world utility. Thank you for identifying where our earlier response exceeded the evidence.

### Label-Y

Thank you for the clarification. We agree that the concerns remain unresolved in the reviewed submission, and we understand why your score remains unchanged.

A promised structural rewrite cannot be evaluated in the current review. Artifact paths can document provenance, but filenames, appendix reorganization, and prose changes do not supply missing numerical results or experiments.

The corpus was assembled retrospectively from available completed runs, with cells differing in models, tasks, samplers, runners, and evaluators, rather than from a predeclared complete matrix. HumanEval and MATH have no numerical main-result evaluation comparable to GSM8K, and the synthetic tool-use evidence is only two boundary cases. The missing cells are missing evidence, not implicit replications. We withdraw all-corpus, cross-corpus, and task-comparative interpretations.

Some single-seed cells were retained after run failures as exploratory cases during paper construction. That explains the record but cannot justify comparative inference. The Qwen PPO/GRPO source values conflict and are quarantined; the Llama comparison is single-seed and backend-confounded. We withdraw the stack-sensitivity, PPO/GRPO, algorithm-ranking, and other single-seed comparative claims rather than treating optimization steps or selected checkpoints as replications.

The fixed GSM8K evaluation used one base checkpoint at 164/200 and five independently trained checkpoints at 166, 165, 161, 168, and 173/200, yielding a trained mean of 83.3% versus 82.0% for the base. The reported `p=.256` is a one-sample seed-level test of the five trained-checkpoint accuracies against the fixed base accuracy, not a paired-seed test; all five per-seed item-paired McNemar tests were nonsignificant. This result is inconclusive and does not establish capability improvement.

The retrospective early-triage set contained only 2/22 collapsed runs, both tool use, and those same two cases also had zero early reward. In both cells, online reward was 0, the zero-variance fraction (ZVF) was 1, and its reported complement, gradient utilization (GU), was 0. This is two-case descriptive concordance and all-wrong homogeneity, not evidence of diagnostic performance or a validated stopping policy. For the submitted centered reward-contrast term, homogeneous rewards analytically make that term zero; these two cases do not establish a learning or intervention outcome.

The submission evaluated neither a pre-existing external-user need nor an external-user decision and outcome, so we withdraw the use-inspired contribution type. What remains is a bounded methodology/reproducibility observation: the exact within-run consequence of homogeneous rewards and the need to keep reward contrast, online reward, held-out capability, and algorithm labels separate. This is not an empirically validated method or a demonstrated real-world use case.

A future study would need predeclared estimands and in-scope cells, explicit missing-cell accounting, fixed held-out evaluations, replication and evaluation sample sizes prospectively justified by power or precision targets, and a tested external-user decision and outcome before broader claims. Those requirements do not repair the reviewed record. Thank you for identifying where our prior response exceeded the evidence.

### Label-Z

Use Round 1 Label-Y verbatim.

## Round 3

Round 2's winning approach is carried forward and tested against a shorter challenger and one final polish.

### Label-X

Use Round 2 Label-Y verbatim.

### Label-Y

Use Round 2 Label-Y verbatim, but omit its final paragraph beginning "A future study would need". End after: "This is not an empirically validated method or a demonstrated real-world use case."

### Label-Z

Thank you for the clarification. We agree that the concerns remain unresolved in the reviewed submission, and we understand why your score remains unchanged.

A promised structural rewrite cannot be evaluated in the current review. Artifact paths can document provenance, but filenames, appendix reorganization, and prose changes do not supply missing numerical results or experiments.

The corpus was assembled retrospectively from available completed runs, with cells differing in model, task, sampler, runner, and evaluator, rather than from a predeclared complete matrix. HumanEval and MATH have no numerical main-result evaluation comparable to GSM8K, and the synthetic tool-use evidence is only two boundary cases. The missing cells are missing evidence, not implicit replications. We withdraw all-corpus, cross-corpus, and task-comparative interpretations.

Some single-seed cells were retained after run failures as exploratory cases during paper construction. That explains the record but cannot justify comparative inference. The Qwen PPO/GRPO source values conflict and are quarantined; the Llama comparison is single-seed and backend-confounded. We withdraw the stack-sensitivity, PPO/GRPO, algorithm-ranking, and other single-seed comparative claims rather than treating optimization steps or selected checkpoints as replications.

The fixed GSM8K evaluation used one base checkpoint at 164/200 and five independently trained checkpoints at 166, 165, 161, 168, and 173/200, yielding a trained mean of 83.3% versus 82.0% for the base. The reported `p=.256` is a one-sample seed-level test of the five trained-checkpoint accuracies against the fixed base accuracy, not a paired-seed test; all five per-seed item-paired McNemar tests were nonsignificant. This result is inconclusive and does not establish capability improvement.

The retrospective early-triage set contained only 2/22 collapsed runs, both tool use, and those same two cases also had zero early reward. In both cells, online reward was 0, the zero-variance fraction (ZVF) was 1, and its reported complement, gradient utilization (GU), was 0. This is two-case descriptive concordance and all-wrong homogeneity, not evidence of diagnostic performance or a validated stopping policy. For the submitted centered reward-contrast term, homogeneous rewards analytically make that term zero; these two cases do not establish a learning or intervention outcome.

The submission evaluated neither a pre-existing external-user need nor an external-user decision and outcome, so we withdraw the use-inspired contribution type. What remains is a bounded methodology/reproducibility observation: the exact within-run consequence of homogeneous rewards and the need to keep reward contrast, online reward, held-out capability, and algorithm labels separate. This is not an empirically validated method or a demonstrated real-world use case.

Any future claim would require predeclared estimands and in-scope cells, explicit missing-cell accounting, fixed held-out evaluations, replication and evaluation sample sizes prospectively justified by power or precision targets, and a prospectively tested external-user decision and outcome. Those requirements do not alter the reviewed record. Thank you for identifying where our prior response exceeded the evidence.

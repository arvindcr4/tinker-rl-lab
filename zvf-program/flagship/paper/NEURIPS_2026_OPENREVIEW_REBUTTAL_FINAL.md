# NeurIPS 2026 OpenReview rebuttal: evidence audit and replacement text

**Prepared:** 2026-07-28  
**Submission:** 36320, OpenReview forum `CXbcYe69BQ`  
**Title:** *Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs*  
**Exact reviewed PDF:** 17 pages, 2,996,794 bytes, SHA-256 `b15ac7e5f673473cf8edc07634f6acbd9fcd54b9f0d5d1f75b106565a174a62d`  
**Reviewer rebuttal deadline:** 2026-07-28 17:29 IST; 10,000 characters per response  
**Confidential AC-comment deadline:** 2026-08-04 16:29 IST; 5,000 characters  

## Recommendation

Replace all three current reviewer responses and the confidential AC comment. Do not patch them incrementally. All four current texts repeat an unsupported Qwen3-8B “matched control,” and OpenReview retains revision history. The correction therefore needs to be explicit, consistent, and easy to find.

This cannot guarantee acceptance. It is the best credibility-preserving response because it answers the AC directly, withdraws claims the artifacts cannot support, and preserves the contribution the score-3 reviewers recognized.

## Mandatory corrections before the deadline

| Current response claim | Audit result | Required action |
|---|---|---|
| Qwen3-8B canonical GRPO 92.6% vs submitted runner 92.1%, paired `p≈.37` | The five W&B summaries reproduce the means and recomputed `p=.374`. The four live seed-45/46 W&B records strongly timestamp-align with four Tinker histories through step 30, and Tinker shows plausible earlier pairs. However, the three zero-runtime backfills contain no upstream IDs and Tinker exposes no seed/arm/algorithm/W&B metadata with which to identify them; seed 42 also uses batch 4 while the others use batch 8. | Explicitly withdraw the five-seed result as transfer evidence in every response where it appeared. |
| Qwen2.5-0.5B arithmetic 99.0% vs 99.2%, paired `p=.374`, “only estimator differs” | The script uses 16 unique prompts repeated eight times for the group-relative arm versus 128 unique prompts for the value-head arm; evaluation draws also differ after divergent RNG histories. | Do not substitute this result for the unsupported Qwen3 claim. |
| Qwen PPO 0.225 vs 0.350 is a rolling-aggregation difference | The values came from different Modal experiments that were misattributed to one row; the statistical artifact is stale. | Quarantine the row and make no PPO-vs-GRPO inference. |
| Fused risk AUROC 0.929 and thresholds 0.55/0.30 validate actionable monitoring | The 0.929 analysis used synthetic/imputed anchors. In the later real-only audit there are four positives; at threshold 0.55 the table reports TP=4 and FP=41. | Remove the AUROC, thresholds, “safe,” “failure-positive,” compute-saving, and actionable-stopping language. |
| Tool-use at ZVF=1 is high-reward saturation | The submitted Table 6 reports both tool-use rows at reward 0 and ZVF=1. | Call this observed all-zero reward homogeneity; it does not by itself prove temporal collapse or high-reward saturation. |
| 82.0→83.3%, `p=.256`, is a paired per-prompt test | `p=.256` is a one-sample test over five trained-checkpoint accuracies against the deterministic base accuracy. The five per-seed McNemar tests are separately non-significant. | Describe the test correctly and avoid “paired p=.26.” |
| GU means Group Uniformity | The submitted paper defines GU as Gradient Utilization, `1−ZVF`. | Use the submitted definition. |
| Multi-seed evidence supports all comparative claims | Several central cells are single-seed, post-selected, cross-stack, or have unresolved identity/provenance. | Remove this sentence. |
| Reward contrast matters more than estimator choice | No valid estimator-only experiment establishes that ordering. | Withdraw it. |

## Surviving claim boundary

The rebuttal should preserve only two claims:

1. For the submitted group-standardized runner, homogeneous rewards make the centered reward-contrast term zero. This does not imply zero total gradient in implementations with reference KL or auxiliary losses, and it does not establish empirical transfer to canonical GRPO.
2. The audit documents concrete conflations among online reward, held-out capability, proxy/verifier reward, and nominal algorithm labels, and motivates reporting them separately.

Do not claim calibrated prediction, a stopping policy, compute savings, canonical-GRPO equivalence, held-out improvement, algorithm superiority, deployment benefit, or controller impact. The only application claim is a pre-adoption audit that tests whether an algorithm claim survives controlled implementation and held-out evaluation.

## Replacement response: Reviewer PYUJ

Thank you for recognizing that separating training reward, verifier/proxy reward, held-out capability, and algorithm labels has “significant methodological value.” Your review identifies the contribution we intend to preserve: a claim-to-evidence audit that prevents these quantities from being treated as interchangeable. We agree that the heterogeneous presentation obscured this contribution, and we will reorganize the paper around the specific inferences each diagnostic can and cannot support.

**Correction to our initial response.** The summaries reproduce 92.6% versus 92.1% (`p=.374`), but only seed-45/46 have live paired summaries with plausible source-run alignment; the three zero-runtime backfills lack unique upstream IDs. We therefore do not have an auditable five-seed matched comparison and withdraw it from all transfer claims. Neither the centered-reward identity nor the paper's claim-to-evidence taxonomy depends on it.

The paper's bounded contribution has two independently supported parts. First, it gives an exact condition for the submitted runner: homogeneous group rewards make its centered reward-contrast term zero. Second, it turns recurring evidence conflations into an auditable reporting protocol connecting every claim to its runner, reward, held-out evaluation, seed structure, and provenance. The first part diagnoses signal availability; the second prevents that diagnostic, online reward, or an algorithm label from being mistaken for capability evidence.

**Coherent evidence structure.** The submitted corpora support case-specific audit findings, not a pooled treatment effect or algorithm ranking. We will remove the heterogeneous mean and reorganize the evidence into one primary claim-to-run table plus explicitly labeled case studies. Every retained claim will state its model, task, runner, seed count, reward definition, evaluation, and provenance. This preserves the value of observing distinct failure modes without treating non-exchangeable runs as replications.

**Post-submission feasibility evidence.** Our frozen private ledger records execution of the protocol in a preregistered 40-unit same-stack audit. A conservative finite-sample re-audit treats all four held-out comparisons as inconclusive. We offer this only to show that the workflow is executable; our acceptance request does not depend on this post-submission evidence, which is not independently reviewer-verifiable without an anonymized artifact.

**Early triage.** The first-five-step result has only two positives, both tool-use, and early reward identifies the same failures. We therefore relabel it as an exploratory sample observation, remove the precision/recall headline and stopping-rule language, and make the negative result explicit: the submitted evidence does not show incremental predictive value over early reward. This calibration is part of the audit's purpose—distinguishing a useful measurement from an unsupported predictor.

**Runner scope.** The minimal runner was chosen to isolate reward-contrast availability, not as an ablation showing that clipping, reference KL, or completion masking are negligible. When the centered advantage is zero, PPO ratio/clipping or completion masking cannot restore that reward-contrast term; a reference-KL or auxiliary term may nevertheless produce a nonzero total gradient. Every empirical result remains specific to the submitted runner. We will call it the “group-standardized runner” throughout, reserve “GRPO” for explicitly specified implementations, and revise the title and abstract accordingly.

**Held-out evidence.** The same pre-RL checkpoint and five independently trained checkpoints were evaluated greedily on the same fixed 200-item GSM8K slice. Base is 164/200; trained scores are 166, 165, 161, 168, and 173, averaging 83.3%. The seed-level test gives `p=.256`, and all five item-paired per-seed McNemar comparisons are non-significant. We report this as a negative capability result: the online reward trace does not license a held-out improvement claim, which is precisely why the audit separates these evidence types.

**Presentation.** The revision has a concrete architecture: problem and use case first; runner semantics and diagnostics before results; one primary evidence table; descriptive cases separated from comparative evidence; no equation or thresholds in the abstract; and rebuilt tables without overflow or internal filenames.

The revised paper therefore offers a coherent methodology/reproducibility contribution: an exact signal-availability result plus a practical audit protocol for preventing reward, capability, and algorithm-label claims from being conflated. Given your Significance and Originality assessments of 4 and your stated willingness to reassess a refined version, we respectfully ask you to reconsider the overall score.

## Replacement response: Reviewer 4G4H

Thank you for the detailed auditability critique. Your review identifies the correct standard for this paper: a reader must be able to determine exactly what was measured, which implementation produced it, and what inference follows. Our re-audit lets us answer those questions directly and remove comparisons that do not meet that standard.

**Correction to our initial response.** The summaries reproduce 92.6% versus 92.1% (`p=.374`), but only seed-45/46 have live paired summaries with plausible source-run alignment; the three zero-runtime backfills lack unique upstream IDs. We therefore do not have an auditable five-seed matched comparison and withdraw it from all transfer claims. Neither the centered-reward identity nor the claim-to-evidence taxonomy depends on it.

**Runner scope.** The minimal runner was chosen to isolate reward-contrast availability, not as an ablation showing that clipping, reference KL, or completion masking are negligible. When the centered advantage is zero, PPO ratio/clipping or completion masking cannot restore that reward-contrast term; reference-KL or auxiliary terms may nevertheless produce a nonzero total gradient. Every empirical result remains specific to the submitted runner. We will call it the “group-standardized runner” throughout and reserve “GRPO” for explicitly specified implementations.

**Ledger audit and heterogeneous mean.** The direct answer is yes: the row-by-row audit found the Qwen PPO mismatch plus three additional exact-ID/model conflicts. Every affected row is quarantined, and no retained comparative claim depends on it. The Qwen values 0.225 and 0.350 came from distinct experiments, so we remove that comparison entirely. We also remove the ten-checkpoint heterogeneous mean because it estimates no coherent treatment effect. The revision replaces both with a claim-to-run table containing exact run identity, model, stack, seed, evaluator, provenance status, and the inference permitted from each row.

**Concrete reward/ZVF/GU/outcome values.** Separately after submission, our frozen private ledger records complete same-stack telemetry. Each entry is `last-10 reward / ZVF / GU / held-out accuracy`, in percent, with the same fixed 500-item evaluation:

| Seed | GRPO | DAPO | GSPO | Dr.GRPO | AERO |
|---:|---|---|---|---|---|
| 11 | 68.8/70.0/30.0/65.0 | 52.5/0/100/64.8 | 69.4/70.8/29.2/64.0 | 66.9/67.5/32.5/64.2 | 67.6/66.7/33.3/63.0 |
| 23 | 73.8/71.7/28.3/63.6 | 55.0/0/100/62.4 | 73.1/67.5/32.5/63.8 | 77.5/72.5/27.5/62.6 | 72.4/76.7/23.3/62.8 |
| 37 | 62.5/64.2/35.8/63.4 | 54.4/0/100/62.8 | 63.1/68.3/31.7/64.2 | 66.9/69.2/30.8/62.2 | 63.8/71.7/28.3/63.6 |
| 53 | 62.5/69.2/30.8/63.4 | 51.9/0/100/63.6 | 65.0/68.3/31.7/63.2 | 65.0/68.3/31.7/63.6 | 65.6/63.3/36.7/63.2 |
| 71 | 64.4/72.5/27.5/63.4 | 58.1/0/100/63.4 | 65.0/74.2/25.8/63.8 | 66.9/66.7/33.3/62.0 | 68.6/69.2/30.8/63.2 |
| 89 | 74.4/64.2/35.8/62.0 | 55.0/0/100/63.6 | 74.4/64.2/35.8/64.6 | 72.5/72.5/27.5/62.8 | 71.1/65.0/35.0/63.0 |
| 107 | 65.0/76.7/23.3/63.4 | 49.4/0/100/63.4 | 66.9/72.5/27.5/64.0 | 63.8/78.3/21.7/62.8 | 68.4/79.2/20.8/63.0 |
| 131 | 65.0/65.8/34.2/62.2 | 61.3/0/100/63.2 | 65.6/68.3/31.7/62.8 | 64.4/74.2/25.8/64.6 | 70.3/68.3/31.7/64.0 |

These values sharpen rather than broaden the interpretation: large differences in realized reward and ZVF/GU do not map mechanically to held-out differences in this controlled block. DAPO changes the upper clipping bound and prompt eligibility/filter-refill behavior and averages 1,734 rollouts versus 480 for GRPO, so the telemetry is descriptive and supports no compute-efficiency claim. Its paired held-out difference is +0.10 pp with 90% CI [-0.35,+0.575] pp, but a conservative paired-t 80%-power MDE is 1.012 pp, slightly above the preregistered 1-point margin; we therefore treat DAPO and the other three comparisons as inconclusive. This is post-submission feasibility evidence, not replacement primary evidence or a reference-KL result; our acceptance request does not depend on it, and independent verification requires an anonymized artifact.

**Corrected saturation interpretation.** Your request exposed that Section 4.4 makes conclusions its table cannot support. The two submitted tool-use cells are reward 0, ZVF 1, GU 0: an all-wrong homogeneous boundary, not high-reward saturation. We remove the cross-configuration reliability and saturation conclusions rather than infer missing per-run relationships. Under conditionally i.i.d. Bernoulli rollouts, `p^G+(1-p)^G` is only the probability of a homogeneous group; it does not establish temporal saturation, impaired learning, or wasted compute.

**Held-out result.** The same pre-RL checkpoint and five trained checkpoints use the same fixed 200-item slice and greedy evaluator: base 164/200; trained 166, 165, 161, 168, and 173. The seed-level test and every item-paired per-seed McNemar test are non-significant. We report a negative capability result, not improvement.

The revision will place Method before results, remove the heterogeneous mean and repeated capacity discussion, and give each retained primary run a readable row with model, task, stack, seed, reward, ZVF/GU, held-out result, status, and provenance. Filenames move to the artifact appendix.

These corrections directly answer the ledger discrepancy, heterogeneous aggregation, runner dependence, requested telemetry, saturation interpretation, and presentation concerns. The resulting contribution is an auditable claim-to-evidence protocol for preventing unsupported RLVR comparisons, not another algorithm leaderboard. We respectfully ask you to reconsider the overall score on this bounded claim set.

## Replacement response: Reviewer 9kjk

Thank you for identifying the decisive weakness in our presentation. Readers should not have to reconstruct either the method or its application from the abstract and results. We appreciate that you nevertheless found the diagnostics inexpensive, easy to apply, and potentially significant. The application is not a new controller proposed in rebuttal; it is the practitioner decision already motivating the submission but insufficiently operationalized there.

**Concrete use-inspired decision.** The submitted introduction describes a reporting framework that “lets practitioners decide whether a specific model–reward–sampler–stack combination is actually learning.” We operationalize that existing practitioner question as whether a declared comparison, fixed held-out evaluation, and reconciled provenance support an adoption claim, rule it out within a prespecified margin, or leave it unresolved. These are reporting outcomes, not a new controller or decision algorithm; an online-reward increase or algorithm label alone cannot trigger an adoption claim.

As post-submission feasibility evidence only, our frozen private ledger records a preregistered 40-unit same-stack execution of this checklist. A conservative finite-sample re-audit treats all four held-out comparisons as inconclusive. We do not ask you to treat this study, its operational labels, or its private receipts as reviewed evidence or as a new contribution.

**Method, made explicit.** The submitted protocol is distributed across the introduction, corpus section, and results. Its intended sequence is: (1) declare the full treatment—model, reward/verifier, sampler, runner, and evaluator—rather than treating an algorithm label as the intervention; (2) measure whether sampled groups contain reward contrast using ZVF, with GU=`1−ZVF`; (3) keep online reward, proxy telemetry, and fixed held-out capability non-interchangeable; and (4) permit a capability or adoption claim only when its evaluation and provenance support it. ZVF/GU are inexpensive optimization telemetry within this protocol, not capability scores or universal predictors. For the submitted group-standardized runner, homogeneous rewards make the centered reward-contrast term zero. Because it omits the PPO ratio/clip, frozen-reference KL, and completion-only masking, we do not infer empirical equivalence to standard GRPO. We agree that the reviewed version under-specified this workflow; the contribution is the submitted diagnostic and evidence separation, not the post-submission verdict labels.

**Evidence boundary.** The submitted evidence does not validate a general stopping rule. The deduplicated set contains 22 heterogeneous runs but only two collapsed cases, both tool-use, and early reward separates the same cases. We withdraw predictive and cross-corpus interpretations. Likewise, the fixed GSM8K comparison—base 164/200 versus trained 166, 165, 161, 168, and 173—is non-significant (`p=.256`, with every per-seed McNemar comparison also non-significant). Its role is not to claim improvement; it demonstrates why online reward must not be reported as held-out capability.

**Bounded structural revision.** The revision narrows claims and reorganizes the protocol already distributed across the submission; it does not introduce a new method. We will replace the detail-heavy abstract with the problem, protocol, and bounded findings; motivate the practitioner decision in the introduction; place a Method section containing the four-step protocol, runner objective, ZVF, and GU before results; separate primary evidence from descriptive cases; expand related work on RL reproducibility, group-relative objectives, and dynamic-sampling/zero-variance interventions; remove non-exchangeable aggregates and repeated discussion; and move filenames to an artifact appendix.

**Correction to our initial response.** The summaries reproduce 92.6% versus 92.1% (`p=.374`), but only seed-45/46 have live paired summaries with plausible source-run alignment; the three zero-runtime backfills lack unique upstream IDs. We therefore do not have an auditable five-seed matched comparison and withdraw it from all transfer claims. The method and application above do not depend on it.

The resulting paper is deliberately scoped: a low-cost reward-contrast diagnostic and evidence-accounting protocol applied to the submitted practitioner question of whether evidence supports changing trainers. This preserves the diagnostic value you recognized while removing unsupported predictive, transfer, and broad empirical interpretations. We respectfully ask you to reconsider the score on the submitted diagnostic and evidence-separation contribution, with its use case now made explicit.

## Replacement confidential comment to the Area Chair

**Why the scoped paper remains worth accepting.** Two reviewers explicitly recognized the methodological value of separating online reward, proxy/verifier reward, held-out capability, and nominal algorithm labels: PYUJ rated Significance/Originality 4/4, and 9kjk called the diagnostics cheap, easy to apply, and potentially significant. We agree that our breadth, organization, and runner labeling obscured that contribution. The bounded acceptance case is: (i) an audit protocol preventing those quantities from being treated as interchangeable, and (ii) an exact diagnostic for the submitted runner—homogeneous group rewards give zero centered reward-contrast advantage. Neither requires an algorithm ranking, calibrated predictor, or positive held-out gain.

**Correction.** The summaries reproduce 92.6% versus 92.1% (`p=.374`), but only seed-45/46 have live paired summaries with plausible source-run alignment; the three zero-runtime backfills lack unique upstream IDs. We therefore do not have an auditable five-seed matched comparison and withdraw it from all transfer claims. The Qwen PPO values 0.225 and 0.350 also came from two distinct Modal runs, not two aggregations of one trace; that comparison is quarantined. Nothing below relies on either result.

**1. Early rule.** The submitted set has 22 heterogeneous runs and only two collapsed positives, both tool-use; early reward separates the same two cases and continuous ZVF ranking is weak. We therefore demote this to an exploratory observation and withdraw calibrated-prediction, cross-domain, stopping, and compute-saving claims. The audit methodology does not depend on that rule.

**2. Standard GRPO.** The submitted implementation is a group-standardized advantage-weighted REINFORCE runner, not canonical GRPO: it omits the PPO ratio/clip, frozen-reference KL, and completion-only mask. Its empirical outcomes are runner-specific. The transferable statement is only about the reward-driven term: equal within-group rewards yield zero centered advantages; clipping cannot recover absent reward contrast, although KL or other auxiliary terms may still give a nonzero total gradient. We make no implementation-equivalence claim.

**3. Ledger and evidence.** Yes: beyond the Qwen PPO mismatch, the audit found three additional exact-ID/model conflicts; all affected rows will be visibly quarantined. The heterogeneous mean will be removed because the corpus supports case-specific audit findings, not a pooled effect or algorithm ranking. Revised tables will be stratified by model/task/stack and show per-run reward, ZVF/GU, held-out outcome, seeds, and provenance. The submitted GSM8K comparison is explicitly inconclusive (base 164/200; trained 166, 165, 161, 168, 173; seed-level and all per-seed McNemar tests non-significant).

Separately, our frozen private ledger records a preregistered post-submission execution of the workflow across 40 units. A conservative finite-sample re-audit treats all four comparisons as inconclusive. We offer this only as feasibility evidence; acceptance need not treat it as reviewed evidence, and independent verification requires an anonymized artifact.

**4. Structure and use case.** The submitted introduction frames the framework as helping practitioners decide whether a specific model–reward–sampler–stack combination is learning; it did not explicitly formulate a trainer-adoption gate. We narrow and operationalize that practitioner decision as a pre-adoption evidence check: require a controlled comparison, fixed held-out evaluation, and reconciled provenance before interpreting online reward as capability or changing trainers. That private post-submission study only illustrates this workflow; it is not evidence the AC must credit. We claim no controller validity, realized compute saving, or deployment benefit.

The revision will put Method before results; use a title centered on auditing rather than algorithm superiority; remove abstract equations/hyperparameters, the heterogeneous mean, repeated discussion, unsupported AUROC/thresholds, and the high-reward saturation claim; and move filenames to the artifact appendix. These edits do not manufacture evidence: they align each retained claim with its evidence and make the contribution auditable within the camera-ready.

We respectfully ask that the paper be evaluated on its submitted, now sharply bounded contribution: an exact reward-contrast diagnostic plus an auditable protocol for separating online reward, proxy reward, held-out capability, and algorithm labels, and that reviewers be invited to reconsider their scores on that basis.

## Rebuttal-writing resources applied

The response pack follows three user-supplied guides:

- [Foerster and Rocktäschel, *How to ML Rebuttal*](https://www.jakobfoerster.com/how-to-rebuttal-ml-paper): target the reviewer, other reviewers, and AC separately; lead with make-or-break concerns; contextualize any new result; re-emphasize reviewer-recognized strengths; use consistent nomenclature; ask explicitly for score reconsideration; stay scientifically honest.
- [Neel Nanda’s NeurIPS rebuttal advice](https://www.lesswrong.com/posts/vJNQZqgnKSxTBdFbS/neel-nanda-s-shortform?commentId=qqHPrBheFwQgrJbzN): treat the AC as the audience for the common response; give the AC a copyable positive case grounded in reviewers’ own assessments; prioritize proof of work over promises; answer even an unlikely-to-move reviewer carefully; maximize impact per unit time.
- [Foerster and Rocktäschel, *How to ML Review*](https://docs.google.com/document/d/1ApQjLIP29vHhs3uOWO3V6sn0GLJnPPfAPdeoz5_Kwys/edit?tab=t.0#heading=h.16t67gkeu9dx): a reviewer tests whether contributions, novelty, primary evidence, ablations, statistical practice, and reproducibility align. The corrected rebuttal therefore makes the surviving contribution and its evidence boundary explicit rather than treating presentation changes as a substitute for missing experiments.

Their advice changes the rhetoric, not the evidence boundary: the five-seed transfer result remains explicitly withdrawn, and the new E1 audit is labelled as separate post-submission evidence with its design, intervals, verdicts, and non-claims stated together.

## What post-May evidence does and does not clarify

| Evidence | Clarifies | Does not resolve |
|---|---|---|
| Preregistered E1 same-stack audit: 5 GRPO-family arms x 8 paired Qwen3-8B/GSM8K seeds in a clipped, completion-masked TRL stack (`beta=0`), 30 steps, fixed 500-item held-out evaluation, 40-unit private ledger | Directly addresses fragmentation, seed count, per-run provenance, and a concrete post-submission audit design. A live Hub refresh resolves all 40 pinned commits, all six checkpoint states per unit, final adapters, and byte-matching 500-row manifests. DAPO is +0.10 pp vs GRPO (90% CI [-0.35,+0.575] pp), but its conservative paired-t 80%-power MDE is 1.012 pp, just above the 1-point margin; all four comparisons are treated as inconclusive. | Transfer from the submitted runner, reference-KL dependence, pre/post capability improvement, early-rule validation (there were no collapses), a controller, or compute savings. The private Hub evidence is not reviewer-visible without an anonymized artifact. |
| W&B, Tinker, and Hub audit of the claimed 5-seed Qwen3-8B comparison | Reproduces the .926/.921 arithmetic and recomputed `p=.374`; finds strong timestamp alignment between four live seed-45/46 W&B records and four Tinker histories through step 30; finds plausible earlier pairs; confirms the PPO .225/.350 values belong to separate runs. | Unique source mapping for the three backfills, a five-seed matched experiment, an equivalence conclusion, or canonical-transfer evidence. The Hub inventory contains no recorded link for the backfills. |
| 12-run group-size sweep (`G∈{2,4,8,16}`, 3 seeds, 40 steps) | ZVF is measurable per run; in this arithmetic setting it changes strongly with group size while reward/held-out stay near ceiling. Across 480 autocorrelated steps, descriptive Spearman correlations are −.740 with gradient norm and −.543 with nonzero-advantage variance. | Early-rule generalization, stopping utility, canonical-GRPO transfer, or independent-sample statistical significance. |
| Three-seed Qwen2.5-1.5B GRPO/Dr.GRPO GSM8K panel | Demonstrates multi-seed pre/post logging and per-item correctness on another model/runner. | Replication of the submitted Qwen3-8B result; untreated causal effect; 600 independent observations. |
| Later provenance/claim-to-run audit | Finds exact-ID/model conflicts and supplies the correct reason to quarantine the PPO row. | A globally clean ledger; three exact-ID/model conflicts remain unresolved. |
| TRL/verl objective-conformance suite and six accepted balanced negative-control units | Shows how to bind realized objective semantics and gradients to evidence receipts. | A positive-control divergence, learning-effect comparison, or proof that implementation details are negligible. |
| Current 12-page flagship | Demonstrates cleaner objective disclosure, claim discipline, and layout. | A repaired version of the 17-page empirical submission; it is a different, narrower methods/reproducibility paper. |

## Literature identified after submission

Recent work helps the next paper but does not rescue this rebuttal:

- Coelho et al., [*Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training*](https://arxiv.org/abs/2606.10709), validates zero-variance groups as an operational target and evaluates a concrete intervention. This strengthens the mechanism’s relevance while raising the novelty bar: a future paper should evaluate a decision/intervention, not merely log ZVF.
- Zhang et al., [*Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective*](https://arxiv.org/abs/2605.12969), analyzes standard GRPO through clipped ratio-based positive/negative contrast. It reinforces the need to distinguish the submitted runner from canonical GRPO.
- Roth et al., [*Hack-Verifiable Environments*](https://arxiv.org/abs/2605.20744), and Helff et al., [*LLMs Gaming Verifiers*](https://arxiv.org/abs/2604.15149), strengthen the motivation for separating verifier reward from intended capability. They do not validate the submitted held-out result.

These papers belong in the next related-work section. They should not be presented as new evidence for the submitted experiments.

## ACL resubmission decision

Do not expand the omnibus paper. The strongest near-term ACL/ARR paper is now the completed E1 algorithm-claim audit, using `zvf-program/audit/reproducibility_audit.tex` as the manuscript base:

1. **Recommended paper: same-stack survival audit.** Research question: when GRPO-family changes are reimplemented as declared overrides in one frozen stack, which held-out gains survive? Primary evidence is the 40-unit E1 matrix, its preregistration, stack/treatment fingerprints, checkpoints, and 500-row traces. Under the conservative finite-sample re-audit, all four comparisons are inconclusive; the paper must report that limitation and repair the frozen statistical pipeline before making any disappearance claim. The present contribution is the audit protocol and evidence infrastructure, not ZVF prediction or an equivalence result.
2. **Optional later paper: prospective ZVF intervention.** Research question: can a frozen reward-homogeneity policy reduce RLVR rollout cost while staying inside a prespecified held-out-performance regret margin? This needs a new intervention study and should not be merged into the E1 paper unless complete before submission.

The E1 paper directly repairs fragmentation, single-seed evaluation, missing per-run provenance, and weak held-out comparisons. It should explicitly state that its clipped, completion-masked TRL baseline uses `beta=0`, so it does not answer reference-KL dependence. The submitted early-triage rule, heterogeneous model mean, PPO row, and use-inspired controller story should not appear as headline evidence.

For a comprehensive intervention paper, submission is a no-go until all of the following hold:

- one primary open canonical stack; a second stack only for external validation;
- component factorial for probability ratio/clip, completion masking, reference KL, advantage estimator, and update epochs;
- at least five independent training seeds per primary cell;
- at least 1,000 frozen held-out examples where available, identical prompts/decoder/parser across arms, and an untreated or compute-matched control;
- prospectively constructed all-wrong, mixed, and all-correct regimes;
- frozen thresholds selected only on development cells and evaluated on held-out model/task cells;
- early ZVF must improve over early reward alone;
- controller cost reduction must meet a preregistered held-out non-inferiority margin;
- every table generated from a reconciled claim/run ledger; unresolved rows quarantined automatically;
- abstract 150–180 words, Method before Results, one result per location, no internal filenames in the main narrative, and no overflow.

For a comprehensive ZVF intervention paper, the concrete application is not “a dashboard.” It is a prospective operator policy—continue, resample prompts, warm-start, change group size, or stop—evaluated on charged rollout cost and held-out regret. The E1 audit has a narrower application: an adoption gate for algorithm claims. Without the prospective intervention, do not claim controller validity, deployment benefit, or compute savings.

For the E1 audit paper, the final structure should be: problem and adoption risk; exact audit estimand and compound verdicts; frozen stack and treatment contract; preregistration and evidence gates; 40-unit paired results; per-arm/per-seed reward, ZVF/GU, held-out, rollout count, and status; limitations; artifact appendix. Add at most one preregistered external-validation block. Do not reopen the paper into a catalogue of models and tasks.

## Posting checklist

- [ ] Replace, rather than append to, each current reviewer rebuttal.
- [ ] Replace the confidential AC comment so the unsupported result is corrected there too.
- [ ] Confirm every response is below its character limit after Markdown is pasted.
- [ ] Preview equations and tables in OpenReview.
- [ ] Re-open each note after saving and verify the rendered text.
- [ ] Do not describe private W&B/Hub evidence as independently reviewer-verifiable unless an anonymized artifact is actually available under the venue rules.
- [ ] Do not upload a revised PDF during the response-only window.
- [ ] Do not claim acceptance is guaranteed; ask for re-evaluation only after the evidence boundary is clear.

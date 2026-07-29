# May review diagnosis, rebuttal, and ACL resubmission plan

> **Superseded for live OpenReview use.** The exact authenticated submission,
> response deadlines, current author responses, and corrected replacement text
> are in `NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md`. Do not paste the older
> recommended response in this document; its earlier ledger and same-stack
> explanations were invalidated by the repository audit.

**Date:** 2026-07-27  
**Scope:** the exact OpenReview submission *Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs* (forum `CXbcYe69BQ`, 17 pages, PDF SHA-256 `b15ac7e5...174a62d`), its current descendants, and repository evidence added after submission.

## Executive verdict

The reviewers found a real contribution and a real paper-design failure. The useful contribution was the separation of online training reward, reward proxies, held-out capability, and algorithm/stack labels. The failure was trying to establish that contribution through one heterogeneous omnibus paper whose strongest empirical rule had only two positive failures, whose only clean held-out comparison was underpowered, and whose central runner was not canonical GRPO.

The July flagship manuscript fixes the disclosure, mathematical specification, structure, claim discipline, and layout problems. It is publishable in its declared narrow scope: a methods/reproducibility preprint and registered-feasibility postmortem. It is **not** a repaired version of the May causal/diagnostic empirical paper. It deliberately makes no training-effect, controller-benefit, calibrated-triage, or real-world deployment claim.

Post-May evidence helps in four bounded ways:

1. it completes a preregistered 40-unit, five-arm by eight-seed same-stack audit using a clipped, completion-masked TRL GRPO baseline (`beta=0`) with a fixed 500-item held-out evaluation per unit;
2. it gives a 12-run group-size/ZVF sweep with per-run reward and held-out values;
3. it adds a three-seed canonical-style GRPO/Dr.GRPO GSM8K pre/post panel;
4. it adds objective-level TRL/verl conformance tests and six balanced negative-control training units.

The 40-unit audit supplies a concrete algorithm-claim audit design, but a later statistical re-audit invalidated the frozen aggregate's DAPO `DISAPPEARS` verdict: its conservative paired-t 80%-power MDE is 1.012 pp, just above the preregistered 1 pp margin, and the preregistered multiplicity step was not applied. All four comparisons must therefore be treated as inconclusive until the pipeline is repaired. E1 also does **not** solve the two-positive early-triage problem, prove equivalence between the May runner and standard GRPO, resolve reference-KL dependence, turn the original held-out result into a replicated pre/post finding, or validate a ZVF controller.

The best defensible response is therefore: **concede scope, correct the record, preserve the methodological contribution, and separate the current narrow paper from the future ACL empirical paper.**

## Artifact boundary

| Artifact | Located status | Consequence |
|---|---|---|
| Exact reviewed submission | Authenticated OpenReview forum `CXbcYe69BQ`; downloaded PDF SHA-256 `b15ac7e5...174a62d`, 17 pages | This is the authoritative review target. The 48–51 page repository PDFs are not byte-identical to it. |
| Submission README and claim verifier | Present in the frozen commit (`submission/contents/REVIEWER_README.md`, `REVIEWER_VERIFICATION.md`, `SOURCE_PRECEDENCE.md`) | Confirms the 82.0% to 83.3%, p=0.26 held-out result, two-positive triage rule, and known 0.225/0.350 ledger discrepancy. |
| Three submitted rebuttal texts and AC comment | Read from authenticated OpenReview on 2026-07-27 | All four are writable and overstate a partially backfilled 92.6/92.1 comparison; replace them rather than patching selectively. |
| Current omnibus descendant | `platform_hybrid/paper/main_eai.pdf` and `main_eai_body.tex` | It contains additional evidence and caveats but remains a 50-page omnibus draft, not an ACL-ready paper. |
| Current July flagship | `zvf-program/flagship/paper/main.pdf`, `main.tex`, `CLAIM_AUDIT.md`, and `review_bundle.zip` | Clean 12-page methods/reproducibility paper; explicitly not a causal-training result. |

## What W&B and the repository say about the claimed “5-seed rescue”

The five W&B summary pairs arithmetically reproduce “canonical GRPO 92.6% versus the REINFORCE runner 92.1%” and recomputed `p=.374`, but they do not substantiate a five-seed auditable matched experiment. Seeds 42--44 are zero-runtime W&B backfills; only seeds 45--46 are live W&B records. Those four live records strongly timestamp-align with four Tinker histories through step 30, and Tinker contains plausible earlier pairs, but the backfills have no upstream Tinker IDs and Tinker exposes no seed/arm/algorithm/W&B metadata with which to map them. The records also lack captured source and item-level evaluation evidence, and seed 42 uses batch 4 while seeds 43--46 use batch 8. The five-seed result must be withdrawn as transfer evidence.

The relevant file is `platform_hybrid/experiments/results/samestack_ppo_grpo.json`, generated by `platform_hybrid/experiments/modal/modal_samestack_ppo_grpo.py` on 2026-06-14. It reports:

| Arm | Seeds | Task/model | Held-out mean | Training last-10 mean |
|---|---:|---|---:|---:|
| Group-relative estimator, called `grpo` | 5 | generated two-digit arithmetic, Qwen2.5-0.5B | 99.0% | 97.89% |
| Learned value-head estimator, called `ppo` | 5 | same | 99.2% | 91.81% |

The artifact reports a held-out difference of GRPO minus PPO = **-0.2 percentage points**, t(4)=-1.0, p=0.374, with a t-based 95% CI of **[-0.76, +0.36] pp**. This statistic should not be presented as paired or estimator-only: the group-relative arm uses 16 unique prompts repeated eight times per step, the value-head arm uses 128 unique prompts per step, and the arms draw evaluation items after different RNG histories. Both use completion masking and no reference-KL term. The result is a confounded, ceiling-saturated exploratory comparison.

It is not:

- a comparison to the May REINFORCE-style runner;
- a full canonical-GRPO transfer test;
- evidence that clipping, KL, and masking are jointly second-order;
- an equivalence or non-inferiority result merely because p>0.05.

The arithmetic experiment below is not a substitute. It does not repair the partially backfilled Qwen3 claim and adds another design dispute. Retain it only in an internal evidence inventory.

## Post-May evidence inventory

| Date / evidence | What was measured | What it clarifies | What it cannot establish |
|---|---|---|---|
| 2026-07-14 to 2026-07-20, preregistered E1 audit | Five GRPO-family arms by eight paired Qwen3-8B/GSM8K seeds using a clipped, completion-masked TRL baseline (`beta=0`); 30 steps; fixed N=500 held-out evaluation per unit; 40-unit private ledger | One coherent algorithm-claim audit design. A live Hub refresh resolves all 40 pinned commits, six checkpoint states per unit, final adapters, and byte-matching 500-row manifests. DAPO is +0.10 pp vs GRPO (90% CI [-0.35,+0.575] pp), but the conservative paired-t 80%-power MDE is 1.012 pp, above the 1 pp margin; all variants are treated as inconclusive. | May-runner transfer; reference-KL dependence; pre/post capability improvement; early-rule validation (zero collapses); controller benefit or compute savings; independent reviewer verification without an anonymized artifact |
| 2026-07-24 to 2026-07-28, W&B/Tinker/Hub audit of claimed five-seed comparison | Five Qwen3-8B canonical/REINFORCE summary pairs | Reproduces .926/.921 and `p=.374`; identifies three W&B backfills; finds strong timestamp alignment between the four live seed-45/46 W&B records and four Tinker histories through step 30 | Unique mapping from the three backfills to candidate Tinker runs; a five-seed matched experiment; equivalence or transfer evidence. Hub inventory supplies no recorded link for the backfills. |
| 2026-06-14, `samestack_ppo_grpo.json` | Five seeds per arm, same codebase, Qwen2.5-0.5B arithmetic; group-relative versus learned value-head estimator | Shows that such an exploratory comparison can be run in one codebase | Estimator-only effect: prompt exposure and evaluation draws differ; original runner transfer; KL dependence; equivalence |
| 2026-06-15, `drgrpo_gsm8k_cot_full.json` | Three seeds per arm, Qwen2.5-1.5B-Instruct, 30-step clipped GRPO versus Dr.GRPO, 200-item pre/post GSM8K evaluation | Demonstrates multi-seed held-out measurement in a longer-output regime; GRPO mean 20.17% to 26.33%, Dr.GRPO 20.50% to 25.50% | Replication of the May Qwen3-8B result; untreated causal effect; independence of pooled per-item McNemar observations across seeds |
| 2026-06-29, `zvf_predictive_validation_results.json` | Pseudo-prospective first-five-step analysis, 22 deduplicated runs | Makes the rule, task breakdown, baselines, and weak rank association explicit | New failures: it is still 2/22, both tool-use; early reward alone has the same collapse AUC in this sample |
| 2026-07-11, claim-to-run and Tinker/W&B/HF provenance audit | Run IDs, model identity, seeds, steps, held-out availability, evidence tiers | Shows exactly which cross-stack and cross-model claims are descriptive, conflicted, or unsupported | A clean pooled causal estimate; three exact-ID/model conflicts remain unresolved |
| 2026-07-21 to 2026-07-27, S1 plus r4-2 | Frozen TRL/verl objective cases; six balanced equal-length negative-control training units; 600 stored gradient relations; common N=128 evaluation | Strong evidence that objective semantics can be tested and bound to receipts; in accepted balanced units, nonzero gradients are nearly collinear | A positive-control divergence, learning-effect estimate, complete screening matrix, or confirmatory result; the filtered positive control was infeasible |

The six r4-2 units are especially important for scope. They support “the intended and native implementations agree in these accepted balanced negative controls,” not “implementation differences do not matter.” The preregistered positive control was never constructed, four balanced seed-37 units remain quota-pending, and no confirmatory matrix ran.

## Criticism-by-criticism status

Legend: **fixed** means fixed in the July flagship's narrower scope; **partial** means clarified or supported but not resolved; **open** means the criticism still applies to an empirical resubmission.

| Reviewer concern | May submission | Post-May/current evidence | Verdict for a comprehensive ACL paper |
|---|---|---|---|
| Fragmented empirical basis | Tinker, Modal, TRL, toy RL, GSM8K, HumanEval, tool schema rewards, many models and protocols were combined. | E1 directly addresses this with one Qwen3-8B/GSM8K clipped, completion-masked TRL stack (`beta=0`), five arms, eight paired seeds, and a fixed 500-item evaluation. | **Substantially fixed for a future audit paper, not retrospectively for the submission.** Use E1 as the primary study in a resubmission; move May heterogeneity to descriptive appendices. |
| Early rule has only two failures | 2 collapsed tool-use runs among 22; precision/recall 1.0; early-ZVF rank correlation weak. | `zvf_predictive_validation.md` still contains the same 22 runs and the same two positives. Later ZVF tables are heterogeneous or use different collapse definitions and cannot be silently added. | **Open.** Do not headline precision/recall until a prospective, held-out, multi-corpus failure set exists. |
| Non-canonical runner | No clipping, reference KL, or completion-only mask; prompt tokens contributed materially to loss. | E1 uses one clipped, completion-masked TRL GRPO baseline with `beta=0`, but does not compare that stack to the May runner or test reference-KL dependence. | **Partial.** E1 supports same-stack variant auditing; May-runner outcomes remain runner-specific unless a transfer factorial is run. |
| Weak held-out evidence | Qwen3-8B, N=200, 82.0% to 83.3%, p=0.26. | E1 provides eight paired seeds per arm and a fixed N=500 evaluation for variant-vs-GRPO comparisons. It does not include an untreated pre-RL control. | **Fixed for variant-survival estimands; open for pre/post capability improvement.** Keep those questions separate. |
| Hasty layout / overflowing tables | Substantiated by reviews. | July flagship compiles to 12 pages with no overfull boxes. The current EAI omnibus is 50 pages; its PDF even clips the author-contact line. | **Fixed only in flagship.** Rebuild the ACL paper from the flagship's discipline, not by trimming the omnibus. |
| Abstract is long and method lives there | Equations, thresholds, and results appeared before a proper method. | Flagship abstract contains the question, method, bounded result, and non-claims; full objective appears before results. | **Fixed in flagship.** ACL abstract should be 150-180 words, no equation and no threshold hyperparameters. |
| Introduction assumes deep RLVR familiarity | The practical decision problem was not established. | Flagship now starts from terminal-signal ambiguity and software-semantic ambiguity. | **Mostly fixed.** Future empirical paper should open with the operational decision: continue, stop, resample, warm-start, or change G. |
| Thin related work | Reviews judged it insufficient. | The omnibus now has an overgrown survey; the flagship has targeted work on starvation, AVSPO, DAPO, verl, verifier errors, and conformance. | **Fixed in direction.** Use a compact comparison table, not a catalogue. |
| No real-world use for “use-inspired” | No deployed or validated decision was shown. | The submitted introduction frames a practitioner-facing learning question; E1 provides post-submission feasibility evidence for operationalizing it as a pre-adoption audit, but all four E1 comparisons are inconclusive under the conservative re-audit. | **Partly fixed.** Narrow the application to evidence auditing; do not claim controller validity, deployment benefit, compute savings, or a resolved adoption decision. |
| Omitted clipping/KL dependence not justified | The paper blurred mechanism and standard GRPO. | Literature and repository evidence show KL is not universal, but completion masking and clipping materially change the realized objective. S1 detects native differences; balanced training units show near-collinearity only in a negative-control regime. | **Open.** A citation that some papers use beta=0 cannot replace a matched ablation. |
| Mean across ten heterogeneous checkpoints | A post-selected mean was narratively easy to misread. | It is a selection-induced 92.08% mean across six distinct model identifiers, not a causal result. | **Remove from headline/body.** Put all rows in a ledger; never average incomparable checkpoints. |
| Repeated capacity-ceiling discussion | Same point appeared in multiple sections. | The omnibus still repeats it; the flagship does not make this claim. | **Open for omnibus; fixed in flagship.** One result, one location. |
| Ledger vs summary discrepancy | Qwen PPO last-10 was reported as 0.225 in one place and 0.350 in another. | W&B confirms these are two distinct 30-step Modal runs, not two aggregations of one trace. A later audit also records three unresolved exact-ID/model conflicts. | **The row is diagnosed, not repaired.** Quarantine it and make E1's reconciled ledger the primary evidence. |
| Section 4.4 lacked per-run ZVF/GU/outcome values | Interpretation substituted for a run table. | Per-run CSV/TSV files now exist, but their populations, labels, and protocols differ. | **Partial.** Publish a primary per-run table with run ID, seed, task, model, reward, ZVF/GU, held-out metric, status, and provenance. |
| High-reward/high-ZVF saturation asserted | Mechanism and consequence were not separated. | For i.i.d. binary rewards, homogeneous probability `p^G + (1-p)^G` explains why both low-p and high-p boundaries produce high ZVF. The flagship correctly treats this as sourced background. Empirical consequences for learning remain untested. | **Partial.** Call it “boundary homogeneity”; show that a controller action improves utility before calling saturation operationally harmful. |
| Single seeds / incomplete audit corpora | Many core cells were single-seed and short. | E1 is complete at 40/40 units with eight paired seeds per arm and remote evidence gates. | **Fixed for the E1 primary matrix.** External-validity and controller studies remain future work. |
| Filenames in prose | Reviewers saw internal artifact names. | The flagship still names artifacts where needed for reproducibility, but its narrative is readable without them. | **Fix editorially.** Use human-readable table/appendix references; put exact paths in the artifact appendix. |

## Answers to the Area Chair's four questions

### 1. Does the early-step rule generalize?

No broad claim is supportable yet. The submitted rule has two positives, both from tool-use. The same threshold has not been prospectively frozen and evaluated on enough independent collapsed runs across tasks/models/stacks. E1 adds 40 canonical-stack units but zero collapses, so it supplies no additional positive cases. The correct paper claim is **exploratory audit observation**, not predictor, validated triage heuristic, or stopping rule.

### 2. Do the conclusions transfer to standard GRPO?

The mathematical fact that homogeneous centered rewards give zero group-relative advantage transfers to the corresponding class of objectives. The May runner's empirical learning behavior does not automatically transfer. E1 demonstrates coherent auditing with a clipped, completion-masked TRL baseline and `beta=0`, but does not compare that stack to the May runner or resolve reference-KL dependence. Until a matched transfer experiment exists, May empirical conclusions must remain **runner-, model-, task-, and stack-specific**.

### 3. Does the ledger discrepancy extend elsewhere, and what can heterogeneous analyses support?

W&B confirms that the 0.225/0.350 values came from different experiments that were misattributed to one row. Later provenance work found additional model/run-link conflicts, including three still unresolved exact-ID conflicts. Heterogeneous pooled analyses can document provenance failure; they cannot estimate a population-average training effect or rank algorithms. E1 is a better design base for a future paper—a paired same-stack study with a private reconciled ledger—but it is not replacement reviewed evidence and its statistical pipeline must be repaired before any equivalence claim.

### 4. Can restructuring map claims to evidence?

Yes. The July flagship demonstrates the correct architecture: define the object, state evidence statuses, present methods before results, bind claims to receipts, and publish explicit non-claims. E1 now supplies the coherent primary audit study. A comprehensive ACL paper should use E1 as the empirical spine and add a separately preregistered intervention only if it is complete; it should not import the omnibus paper wholesale.

## Recommended rebuttal for the May submission

**Do not use the older text below.** It is retained only as lineage. The live, character-counted replacements are in `NEURIPS_2026_OPENREVIEW_REBUTTAL_FINAL.md` and explicitly correct the unsupported matched-control and ledger explanations.

> We thank the reviewers and AC for identifying a common failure in our presentation: the paper combined a narrow diagnostic contribution with a heterogeneous audit corpus in a way that made the scope difficult to recover. We agree that the defensible contribution is not a universal claim about GRPO or final capability. It is a measurement and reporting result: online reward, held-out capability, reward-proxy behavior, and nominal algorithm labels are different evidence types, and homogeneous group rewards identify absence of within-group contrast for the submitted runner.
>
> **Early triage.** The first-five-step rule has only two positive collapse cases, both in tool-use runs, and early reward alone separates the same two cases. We withdraw predictive and stopping-rule language. This is an exploratory audit observation, not a validated triage heuristic.
>
> **Runner scope.** The submitted runner is GRPO-style, not canonical GRPO: it omits the PPO ratio/clip, frozen-reference KL, and completion-only mask. We will move this fact to the abstract, method, and first limitations paragraph; give the exact scalar objective and token mask; and scope all empirical conclusions to this runner. The group-homogeneity mechanism applies to centered group-relative advantages, but learning outcomes need not transfer across implementations.
>
> **Held-out evidence.** The sole clean submitted capability comparison is Qwen3-8B on 200 held-out GSM8K items, 82.0% before versus 83.3% after training (p=0.26). We agree that this is inconclusive and will present it as a negative control against interpreting online reward as capability, not as evidence of improvement.
>
> **Heterogeneity and auditability.** We will remove the mean over ten post-selected heterogeneous checkpoints from the argument, separate primary evidence from descriptive case studies, and add a per-run table containing model, task, stack, seed, reward, ZVF/GU, held-out metric, and provenance. The Qwen PPO values 0.225 and 0.350 came from different experiments that were misattributed to one row; the comparison is quarantined and no directional PPO/GRPO claim is made. We will also remove internal filenames from the main narrative and repair all table overflow.
>
> **Application and contribution type.** The submitted introduction already frames a practitioner-facing question—whether a specific model–reward–sampler–stack combination is learning. We operationalize that aim as a pre-adoption audit: before changing RLVR algorithms or treating online reward as capability evidence, a practitioner requires a controlled stack, fixed held-out evaluation, and reconciled provenance. E1 provides post-submission feasibility evidence, but its conservative re-audit leaves all four comparisons inconclusive. We claim no stopping controller, deployment benefit, or compute saving.
>
> **Correction to our initial response.** The claimed Qwen3-8B matched control of 92.6% versus 92.1% has no supporting raw result and is withdrawn. We do not substitute the 99.0%/99.2% arithmetic experiment because unequal prompt exposure and different evaluation draws prevent an estimator-only interpretation.
>
> These changes narrow the paper substantially. We believe they preserve the methodological value recognized by the reviewers while making every claim auditable and preventing over-generalization.

### What not to say in rebuttal

- Do not say p=0.37 proves the runners are equivalent.
- Do not use 92.6%/92.1%; those values are not the checked-in five-seed control.
- Do not claim the early rule has been replicated; its published validation population remains 22 runs with two positives.
- Do not call the June GSM8K panel a replication of the May Qwen3-8B comparison; it uses Qwen2.5-1.5B and a different runner.
- Do not claim all ledger rows are audited cleanly; later provenance work found unresolved conflicts.
- Defend only the narrow algorithm-adoption-audit application; do not imply a validated controller, deployment benefit, or compute saving without a prospective intervention.

## ACL paper decision

The current ARR call welcomes negative results, reproduction studies, NLP engineering experiments, software, and data/model analysis. It does not require an application framing. A long paper currently has up to eight content pages, with limitations after the conclusion. See the official [ARR call for papers](https://aclrollingreview.org/cfp), [author guidelines](https://aclrollingreview.org/authors), and [review form](https://aclrollingreview.org/reviewform).

Two valid submission products are possible.

### Product A: focused Findings/short or reproducibility paper

Use the July flagship as the base. Contribution types: **NLP engineering experiment**, **reproduction study**, **publicly available software**, and possibly **theory**. The claim is that equation labels do not identify realized training semantics, and a fail-closed conformance chain prevents causal interpretation before objectives and gradients agree. The incomplete campaign is a registered-feasibility postmortem. This path is already internally publishable in its stated scope and needs venue adaptation, anonymization, and an ACL-length rewrite rather than new causal claims.

### Product B: comprehensive ACL long paper

Use one empirical question:

> Can a prospectively frozen reward-homogeneity triage policy reduce canonical-RLVR rollout cost without exceeding a prespecified held-out-performance regret?

This turns the missing application into a concrete one: training operators must decide whether to continue, stop, resample prompts, warm-start, or alter group size. The output is not a diagnostic dashboard; it is an evaluated decision policy with cost and regret.

## Minimum empirical design for Product B

### Primary estimand

For controller `c` relative to fixed canonical GRPO `b`, report

`Delta_cost = E[charged rollouts_c - charged rollouts_b]`

and

`Delta_perf = E[heldout_c - heldout_b]`.

Predeclare a non-inferiority margin `delta` for held-out performance. Success requires an upper-confidence bound showing cost reduction and a lower-confidence bound showing `Delta_perf > -delta`. This is the real application claim.

### Coherent matrix

| Axis | Minimum design |
|---|---|
| Stack | One primary open canonical stack; second stack only as external validation. |
| Algorithms | Canonical GRPO primary; submitted runner as a nested transfer ablation, not a co-equal pooled condition. |
| Models | Two families and two capacity regimes, chosen before running. |
| Tasks | One reasoning task plus one genuinely executed tool/agent task; never mix schema compliance with executed success. |
| Seeds | At least five independent training seeds per primary cell. |
| Horizon | At least 100 steps or a preregistered convergence/budget rule. |
| Held-out | At least 1,000 frozen examples where available, identical prompts/decoder/parser across arms, all seeds reported. |
| Failure controls | Prospectively constructed all-wrong, mixed, and all-correct regimes with feasibility checks before training. |
| Triage actions | Continue, stop, resample/warm-start, or change G; every action charged for rollout and evaluation cost. |
| Validation | Freeze thresholds on development cells; evaluate on held-out model-task or task-family cells. |

The early-triage sample must contain enough independent failures to estimate recall with useful uncertainty. Twenty observed failures is still modest: even 20/20 recall gives a two-sided 95% Clopper-Pearson lower bound of roughly 83%. Report the interval, not only point precision/recall. Compare ZVF against reward mean alone and a simple combined baseline; the May data already show early reward alone was equally separative for the two failures.

### Runner-transfer factorial

Do not treat “canonical” as one binary label. Freeze and independently vary:

1. probability ratio/clipping;
2. completion-only versus prompt-plus-completion masking;
3. reference KL, including beta=0 as an explicit level;
4. group-relative versus value/REINFORCE-style advantage;
5. update epochs per rollout batch.

Use a screening factorial to find material components, then a confirmatory matched comparison on the primary task. For any “same” conclusion, predeclare an equivalence margin and use two one-sided tests or an interval decision. A non-significant difference is not equivalence.

### Held-out protocol

- Freeze the held-out set and decoding before training.
- Record per-example pre/post correctness and all seed-level outcomes.
- Include an untreated or compute-matched control so pre/post drift is interpretable.
- Use paired bootstrap or McNemar for per-example changes and seed-level intervals for training replication.
- Never select checkpoints by online reward for the primary estimate.
- Report online reward, verifier score, held-out capability, and cost in separate columns.

### Evidence ledger

Every reported row must carry:

`claim_id, run_id, commit, model, tokenizer, task_split, reward_parser, stack/version, objective fingerprint, seed, horizon, checkpoint rule, evaluation rule, status, evidence tier, source hash`.

The paper build should fail if:

- a table value is not generated from the ledger;
- two sources disagree without an explicit precedence rule;
- a supposedly matched cell differs on an undeclared field;
- a failed/partial run enters a completed aggregate;
- an aggregate combines non-exchangeable models or protocols.

## Eight-page ACL architecture

1. **Abstract (150-180 words):** operational problem, method, one primary result with uncertainty, bounded claim. No equations or thresholds.
2. **Introduction (0.75 page):** the operator's decision and why reward alone is ambiguous; three contributions; one claim boundary.
3. **Related work (0.75 page):** starvation/advantage collapse, adaptive sampling/controllers, GRPO implementation semantics, evaluation auditing. Include a compact “ours versus prior work” table.
4. **Method (1.5 pages):** exact canonical objective, ZVF/GU definitions, controller/action space, cost/regret estimands, conformance checks.
5. **Experimental design (1.25 pages):** primary matrix, controls, held-out split, seeds, power/precision, ledger/provenance.
6. **Results (2.25 pages):** primary cost/non-inferiority result; prospective triage validation; runner-transfer ablation; held-out effect. Per-run detail goes to appendix/artifact.
7. **Analysis and limitations (0.75 page):** where the policy fails, task/model transfer, verifier errors, no universal GRPO conclusion.
8. **Conclusion (0.25 page):** one result and one scope statement.

No section should exist solely to explain what another table meant. No table should average different models/protocols. Internal filenames belong in the reproducibility appendix, not the argument.

## Submission stop/go gates

Do not submit Product B as a comprehensive ACL long paper unless all are true:

- the positive/failure controls are feasible and frozen before confirmatory runs;
- every primary cell has at least five completed seeds;
- the prospective failure set supports an uncertainty interval that is useful for the claimed decision;
- early ZVF adds value beyond early reward alone on held-out cells;
- the controller reduces charged cost while meeting the held-out non-inferiority margin;
- the canonical runner and submitted-runner relationship is resolved by a matched component ablation or all claims stay runner-specific;
- every primary held-out result uses a frozen, non-selected evaluation protocol;
- every table is regenerated from the claim ledger and all discrepancies are resolved or quarantined;
- the paper fits the ACL page limit with no overfull boxes, clipped text, undefined citations/references, or internal path clutter;
- an adversarial reviewer can state the method and the real application after reading only the introduction and method.

If any gate fails, submit Product A in its honest methods/reproducibility scope or wait. Adding more heterogeneous rows will make the May problem worse, not better.

## Bottom line

The May reviews are not evidence that the diagnostic idea failed. They show that the paper asked readers to infer a clean contribution from a corpus that could not support a clean causal story. The July flagship fixes the epistemology and presentation. The ACL paper now needs one missing piece: a prospective, canonical, multi-seed decision experiment that converts “ZVF is observable” into “this action saves measured compute at bounded held-out regret.” Until that exists, the strongest publishable paper is the narrower conformance/reproducibility work—not the omnibus audit.

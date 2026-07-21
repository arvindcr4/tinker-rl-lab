# Competitive red team: variance-starvation flagship

**Review date / literature cutoff:** 2026-07-20  
**Review posture:** adversarial NeurIPS/ICLR/ICML main-track review  
**Scope:** the frozen E1 evidence, the frozen-but-unstarted flagship protocol, the synthetic policy comparison, the current `ZVFController`, and primary-source verification of the closest work.

## Executive verdict

**No path currently has an earned main-track claim.** The repository has unusually strong provenance for a narrow audit, but it has no new flagship outcome: S1--S4 are all unstarted, the controller simulation is explicitly synthetic, and the current controller is not the controller in the frozen protocol. A preregistration is evidence of discipline, not evidence for H1--H4.

There are still two credible *conditional* main-track routes:

1. **Best risk-adjusted scientific route: Path 7 joined to Path 4.** Prove a nontrivial information/regret result showing when a variance-only statistic is decision-insufficient, identify the minimal action-relevant statistic, and causally validate the predicted intervention difference through exact objective/gradient conformance on multiple stacks. A one-line observation that ZVF cannot distinguish all-zero from all-one rewards is not enough.
2. **Highest-upside but lowest-probability route: Path 5.** The unified paper is credible only if every causal link passes and the controller beats current external methods, not merely the five internal baselines in the frozen six-arm screen.

Path 4 alone can become a main-track analysis paper only if it finds a general, consequential semantic divergence and repairs it causally. Path 6 is the strongest paper supported by evidence already held, but its natural destinations are Datasets & Benchmarks, MLSys, TMLR, or a reproducibility venue unless it is made much broader. Path 2 has a narrow chance with a decisive simplicity/efficiency win. Paths 1 and 3 should not lead a main-track submission.

## What the repository proves now—and what it does not

### Evidence actually held

- E1 is complete and independently verified for **40/40 units**: GRPO plus DAPO, GSPO, Dr.GRPO, and AERO, eight paired seeds each, pinned Qwen3-8B/GSM8K, 30 optimizer steps, and 500 held-out examples. The frozen aggregate says DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO remain `INCONCLUSIVE`.
- This is a strong provenance and bounded-survival result. It is not a general ranking of the algorithms. It covers one model, one binary-reward task, one short horizon, LoRA, and one execution surface.
- The flagship protocol is frozen at `frozen-screening-not-started`. Its hypotheses, seed split, endpoints, margins, and ordered expansion gates are design commitments, not positive evidence.
- The policy comparison is a synthetic allocation model. Its most damaging finding for a universal-controller story is that registered wins split among `static_g8`, `boundary_aware`, and `full_triage`, while the mean AUC of G8 and full triage is nearly tied. Sparse and compute-constrained cases favor G8.

### Evidence not held

- No S1 objective/gradient differential has passed on TRL and verl.
- No registered controller arm has run on a real training job.
- There is no prospective H1 calibration result, randomized boundary-action contrast, controller AUC effect, measured FLOPs-to-target win, graded-reward validation, or secondary-stack sign replication.
- There is no boundary-information theorem or lower bound in the reviewed flagship materials.
- The frozen screen does not contain direct AVSPO, VIP, HORA, CoDaPO, AR3PO, DEPO, or DynaMO arms. Passing H2/H3 against the current internal controls would therefore not by itself establish a 2026 state-of-the-art algorithm claim.

## Implementation reality: the current controller cannot instantiate the flagship claim

The current `zvf-program/zvf-triage/src/zvf_triage/controller.py` does two different things:

1. `_classify` uses mean reward to label high-ZVF batches as cold-start collapse or saturation.
2. `_adaptive_group_size` uses only rolling aggregate ZVF, so high-ZVF all-wrong and high-ZVF all-correct batches receive the same group-size response. Per-prompt dropping is triggered by repeated zero variance, not by an explicit all-wrong/all-correct action split.

Thus the code has boundary-aware **labels** but a symmetric adaptive-G **actuator**. In particular, saturation can emit `lift_difficulty` while adaptive G still increases from the same high rolling ZVF. This is not the frozen rule “expand all-wrong; do not expand and retire all-correct,” and it does not implement Wilson gating or the frozen compute-aware action rule. Using this class as the S1 treatment would invalidate the intended H3 contrast. New action-level fixtures must target the six registered policies themselves.

## Verified collision map

The following are primary-source facts, not similarity-by-title:

| Work | Verified contribution | Collision with this program |
|---|---|---|
| [Gradient Starvation (2605.07689)](https://arxiv.org/abs/2605.07689) | Names the binary-reward GRPO failure, proves the group-mean-centered zero signal, gives a Jensen heterogeneity result, reports 0.69 degeneracy at G=4, and supplies a fixed-reference Sign intervention with a seven-seed GSM8K gain. | Directly occupies mechanism-only starvation, exact binary-event analysis, and a mechanism-to-intervention story. |
| [AVSPO / Advantage Collapse (2605.21125)](https://arxiv.org/abs/2605.21125) | Defines Advantage Collapse Rate, injects virtual rewards without new model rollouts, treats all-wrong and all-correct groups differently, and reports separate Error-Only, Correct-Only, and full ablations. The official [ICML 2026 downloads list](https://icml.cc/Downloads/2026) confirms the paper. | Directly occupies a diagnostic-plus-intervention paper and empirically isolates the two boundaries. It is the most dangerous collision for Paths 1, 2, 5, and 7. |
| [VIP (2602.01601)](https://arxiv.org/abs/2602.01601) | Predicts per-prompt success probability with a Gaussian process, derives gradient-variance dependence, and solves a hard-budget convex rollout-allocation problem. Its [OpenReview record](https://openreview.net/forum?id=Z5sWYACAop) is an ICLR 2026 poster. | Occupies prospective probability/variance estimation and principled budgeted allocation. |
| [HORA (2605.07114)](https://arxiv.org/abs/2605.07114) | Uses a Beta posterior and a globally optimal greedy allocator for posterior hit utility; explicitly avoids saturated prompts and targets frontier prompts while keeping the estimator unchanged. | Occupies learning-free posterior allocation and much of the “simple rule under a hard budget” space. |
| [AERO (2602.14338)](https://arxiv.org/abs/2602.14338) | Uses a probe stage, correctness strata, iterative rescue for zero-success prompts, Bayesian stabilization for zero-advantage groups, and rejection sampling; reports roughly 48% training-compute reduction. | Already implements differentiated boundary handling plus adaptive rollout and curation. |
| [CoDaPO (2606.07950)](https://arxiv.org/abs/2606.07950) | Separates easy, hard, and learnable questions using confidence and empirical difficulty, then reweights and resamples learnable questions under fixed compute. The official [ICML 2026 downloads list](https://icml.cc/Downloads/2026) includes it. | Occupies competence-band selection and difficulty-aware update allocation. |
| [DAPO (2503.14476)](https://arxiv.org/abs/2503.14476) | Dynamically oversamples and symmetrically filters prompts whose sampled accuracy is 0 or 1, alongside other objective/system changes. | Establishes the zero/one filtering baseline and warns that algorithm labels bundle multiple levers. |
| [GSPO (2507.18071)](https://arxiv.org/abs/2507.18071) | Replaces token-level importance semantics with sequence-level likelihood ratios, clipping, and optimization. | Demonstrates that gradient semantics—not just reward variance—can dominate a nominal algorithm comparison. |
| [Dr.GRPO (2503.20783)](https://arxiv.org/abs/2503.20783) | Diagnoses response-length and question-difficulty biases, removes response-length and group-standard-deviation normalization, and documents related open-source reduction behavior. | Occupies objective-bias auditing and supports the need for, but also raises the novelty bar for, Path 4. |
| [AR3PO (2509.25808)](https://arxiv.org/abs/2509.25808) | Combines adaptive rollout with reuse of previously correct responses and reports up to 4.2x lower rollout cost than DAPO. | Occupies efficient adaptive allocation plus replay. |
| [DEPO (2602.06375)](https://arxiv.org/abs/2602.06375) | Uses an online difficulty estimator to filter likely low-utility prompts before rollout and reports up to 2x rollout-cost reduction. | Occupies pre-rollout learnability filtering. |
| [DynaMO (2602.19208)](https://arxiv.org/abs/2602.19208) | Proves uniform allocation suboptimal under its assumptions, derives variance-minimizing allocation using Bernoulli variance, and adds gradient-aware advantage modulation. | Occupies “theory + allocator + gradient intervention,” especially any maximal-controller or unified claim. |

The papers differ materially, but collectively they remove the broad claims “we discover starvation,” “we introduce adaptive G,” “variance predicts allocation,” and “easy and hard prompts need different treatment.” Novelty must live in a stronger causal or information-theoretic statement.

## Seven-path adversarial review

| Path | Strongest reject argument | Exact missing proof / experiment | Sharpest collision | Rescue condition |
|---|---|---|---|---|
| 1. Mechanism-only ZVF | Zero variance gives zero centered advantage is already named, proved, measured, and mitigated; a new metric name is not a contribution. | Correlated/graded-reward theory; prospective unseen-cell calibration; conditioned gradient direction tests; cross-stack causal intervention. | Gradient Starvation; AVSPO; AERO; VIP/DynaMO. | Discover a genuinely new cross-regime causal law, or demote ZVF to the fixture/statistic used by Paths 4/7. |
| 2. One-formula asymmetric controller | Retry-lower/retire-upper looks like a hand-written compression of AERO/AVSPO/HORA/VIP. | Cost-sensitive derivation/regret; exact actuator; S1; matched-token/FLOP head-to-head against faithful current allocators; graded and second-stack replication. | AERO and AVSPO most directly; HORA/VIP on principled allocation. | Be materially simpler **and** strictly better on the quality-compute frontier with distinct, stable actions. |
| 3. Maximal controller | An unidentifiable kitchen sink whose breadth exceeds the task/ablation matrix. | Formal constrained-control problem, oracle/lower bound, broad factorial effects, overhead accounting, multi-family transfer. | The union of VIP, HORA, AERO, AR3PO, DEPO, CoDaPO, DynaMO, GSPO, and Dr.GRPO. | Reframe as a benchmark/API or fund a substantially broader control study. |
| 4. Causal cross-stack audit | At present it is unrun unit testing, vulnerable to “version-specific engineering” rejection. | Formal equivalence taxonomy; public multi-version fixtures; a real semantic divergence; bf16/distributed validation; paired repair that changes the outcome as predicted. | Dr.GRPO objective/implementation bias; GSPO ratio semantics; DAPO's system bundle. | Establish formula -> code -> gradient -> outcome causality and release an extensible conformance standard. |
| 5. Unified chain | Three crowded papers stapled together; the frozen internal screen lacks current external baselines. | Strong theory, causal conformance, mechanism intervention, all S1--S4 outcomes, plus faithful AVSPO/AERO/VIP/HORA-class comparisons. | Gradient Starvation/AVSPO on mechanism-intervention; VIP/DynaMO on theory-allocation; AERO/HORA/CoDaPO on control. | Every causal link and external head-to-head must pass; otherwise publish the surviving bounded component only. |
| 6. Algorithm audit benchmark | Forty verified records are narrow coverage, not a general benchmark; one common-stack reimplementation may be unfaithful and the E1 budgets differ. | Published-recipe and common-stack estimands; open/multiple stacks; longer curves; more tasks/rewards; matched costs; powered equivalence. | DAPO's open system plus existing objective analyses/comparisons. | Pair with Path 4 and show semantic certification explains survival/reversal across a broad versioned benchmark. |
| 7. Boundary-information insufficiency + causality | The weak statement “ZVF=1 for all-zero and all-one” is a one-line non-injectivity fact already acted on empirically. | A decision model, observational-equivalence construction, positive minimax-regret lower bound, minimal sufficient statistic, and randomized boundary actions across stacks/regimes. | AVSPO's Error-Only/Correct-Only/full ablation; AERO correctness strata; HORA posterior utility; DAPO endpoints. | Make it a real information/regret theorem and causally validate exactly the lost information through Path 4. |

### Path 1 — Mechanism-only ZVF

**Strongest reject argument.** This is a renamed and narrower version of work already available. For binary GRPO, zero within-group reward variance implying zero centered advantage is algebra, not a discovery. Gradient Starvation already owns the name-level mechanism, the exact failure event, a heterogeneity theorem, logged prevalence, and an intervention. AVSPO already presents ACR as a real-time diagnostic and reports a successful mitigation. A ZVF paper whose main result is `p^G + (1-p)^G` or a correlation with gradient norm is below the main-track novelty bar.

**Exact missing proof/experiment.** A viable mechanism paper would need all of the following:

- a result beyond i.i.d. binary rewards—e.g. identified bounds under correlated decoding/verifier error and a non-vacuous graded-reward extension;
- prospective prediction on unseen task/model/G cells, not reconstruction of the group used to compute ZVF;
- direct gradient magnitude **and direction** tests after conditioning on length, clipping, importance ratios, reward scale, and entropy;
- an intervention that changes the proposed mechanism while holding non-treatment semantics fixed, replicated on at least two stacks.

**Novelty collision.** Gradient Starvation is direct; AVSPO and AERO cover diagnosis and mitigation; VIP and DynaMO connect Bernoulli variance to allocation or gradient variance; DAPO already operationalizes the two zero-variance endpoints.

**Rescue condition.** Do not submit ZVF as a standalone mechanism. Use its exact identity as a fixture oracle inside Path 4 or as the lossy statistic in Path 7. A mechanism-only main-track route survives only if it discovers a new cross-regime causal law that the named competitors do not state and that predicts real gradient behavior prospectively.

**Verdict:** **No-go as a standalone main-track paper.**

### Path 2 — One-formula asymmetric boundary controller

**Strongest reject argument.** “Retry all-wrong and retire all-correct” is sensible, but by July 2026 it looks like an obvious compression of existing methods, not a new algorithm. AERO already stratifies by correctness and rescues zero-success prompts; AVSPO separately intervenes on all-wrong and all-correct collapse; HORA allocates by posterior hit utility; VIP solves a principled budget allocation; CoDaPO/DEPO target the learnable band. A reviewer can fairly call the formula a hand-written threshold policy with fewer features.

**Exact missing proof/experiment.** The paper needs:

- a cost-sensitive derivation of the formula and a theorem establishing when its action is optimal or its regret is bounded;
- an implementation that truly separates lower-boundary rescue, upper-boundary retirement, and base-G behavior;
- S1 equality/difference fixtures on both stacks before training;
- compute-matched comparisons against static G8/G16, symmetric ZVF, failure-only, **and faithful AVSPO, AERO, VIP/HORA-class allocators**, with controller/probe/rejected-rollout overhead charged;
- multi-seed learning-curve AUC, final-quality non-inferiority, measured FLOPs-to-target, action-disagreement matrices, graded reward, and secondary-stack sign replication.

The frozen v1 screen alone is insufficient because its “best naive arm” is not the best known external method.

**Novelty collision.** AERO is closest in action structure; AVSPO is closest in explicit two-boundary isolation; HORA/VIP are strongest principled simple allocators; AR3PO, DEPO, DynaMO, and CoDaPO occupy adjacent efficiency claims.

**Rescue condition.** The one-line rule must be materially simpler than these methods and still lie strictly above their quality-compute frontier under the preregistered accounting. It must also make meaningfully different actions—far more informative than merely clearing the current `<95% same decisions` falsifier. If it only ties a naive heuristic, the correct outcome is a negative controller result, not a main-track algorithm claim.

**Verdict:** **Conditional, low-probability main-track path. Run only as the minimal controller.**

### Path 3 — Maximal general compute controller

**Strongest reject argument.** This is a kitchen-sink policy whose components and hyperparameters cannot be causally identified with the available matrix. Prompt choice, G, temperature, clipping, continuation, replay, and retirement each collide with existing work. Three tasks and two model sizes cannot support “general,” and any gain can be attributed to tuning or uncharged controller overhead.

**Exact missing proof/experiment.** It would require a formally specified constrained control problem, observable state and action semantics, an oracle or lower bound, component-wise marginal effects, a broad factorial/ablative study, hardware-aware end-to-end accounting, and transfer across materially different reward/task families and stacks. The current preregistration does not define or power that program.

**Novelty collision.** VIP/HORA/DynaMO cover allocation; AERO/AR3PO cover adaptive sampling and curation/replay; CoDaPO/DEPO cover question selection; GSPO/Dr.GRPO cover objective surfaces. Combining them is not novelty.

**Rescue condition.** Reframe it as an open controller benchmark/API with common accounting and multiple submitted policies, or derive a genuinely new constrained-control result and fund a much broader study. Neither is the present flagship.

**Verdict:** **No-go for this protocol and main-track cycle.**

### Path 4 — Causal cross-stack objective/gradient audit

**Strongest reject argument.** In its current state this is planned unit testing. Float64 equality on synthetic fixtures can be dismissed as software QA; two stack versions can be dismissed as version-specific; intentional framework differences are not bugs; and no evidence yet shows that a fixture divergence causes a different learning conclusion. S1 is unstarted.

**Exact missing proof/experiment.** A research contribution needs:

- a formal equivalence/treatment taxonomy specifying masks, reductions, ratios, clipping, group selection, loss scale, and gradient identity;
- public canonical fixtures with metamorphic and negative-control tests across multiple versions and preferably a third stack;
- at least one real, undocumented semantic divergence—not only an injected fault;
- a paired repair experiment showing formula -> code -> gradient -> training-outcome causality under matched non-treatment fields;
- robustness from float64 reference through bf16/distributed execution, with a justified tolerance hierarchy.

**Novelty collision.** Dr.GRPO already converts objective/implementation bias into an algorithm and documents reduction behavior; GSPO attacks importance-ratio granularity; DAPO is a full stack-plus-algorithm system. These works make “we audited equations” insufficient, but none supplies a general fail-closed conformance standard.

**Rescue condition.** Show that the same public algorithm label produces materially different gradients across widely used stacks, that the suite predicts the difference, and that repairing it changes a survival verdict or learning curve in the predicted direction. Release a versioned standard others can extend. Then the contribution is causal measurement science, not testing.

**Verdict:** **Best near-term conditional main-track analysis path; no positive claim until S1 and downstream repair validation exist.**

### Path 5 — Unified theory + intervention + controller

**Strongest reject argument.** A reviewer will see three crowded contributions stapled together: elementary starvation theory, an intervention already covered by Sign/AVSPO, and an allocator already covered by AERO/VIP/HORA/DynaMO. Failure of any link makes the narrative look post hoc. Worse, the frozen screen compares only internal controller variants, so even a pass does not establish competitiveness against accepted 2026 methods.

**Exact missing proof/experiment.** It needs the Path 7 theorem or an equivalently strong theory contribution, Path 4 causal conformance, a targeted mechanism-restoring intervention, all S1--S4 outcomes, and direct contemporary external baselines. Every stage must preserve the same causal estimand. The controller effect needs positive paired AUC, cell-wise non-inferiority, lower measured FLOPs-to-target, and secondary-stack sign preservation; the mechanism prediction must pass held-out calibration on binary and graded regimes.

**Novelty collision.** Gradient Starvation and AVSPO already offer mechanism-to-intervention chains; VIP and DynaMO offer theory-to-allocation chains; AERO, HORA, CoDaPO, AR3PO, and DEPO cover controller components.

**Rescue condition.** Earn a chain competitors do not close: a pre-fit information/mechanism theorem predicts a held-out failure; a surgical treatment changes the predicted objective/gradient quantity; and a minimal controller converts that change into a matched-compute outcome across stacks and reward regimes. Before any scored run, create a clearly versioned protocol addition or separate confirmatory comparison for the missing external baselines; do not quietly reinterpret the frozen v1 arms.

**Verdict:** **Credible high-upside main-track route only if every gate and external head-to-head passes. Currently no-go.**

### Path 6 — Algorithm audit benchmark

**Strongest reject argument.** Forty impeccable records do not create broad scientific coverage. E1 is 30 steps on one Qwen3-8B/GSM8K/LoRA setting. DAPO used roughly 1.5k--2.1k rollouts per seed while GRPO/GSPO used 480, so the audit is not itself a common-budget efficiency benchmark. Reimplementing named methods on one common stack can remove system components necessary to their published effect. Three arms are underpowered, and “inconclusive” cannot be marketed as failure.

**Exact missing proof/experiment.** An archival benchmark needs two separate estimands:

1. faithful published-recipe reproduction, and
2. common-stack algorithm isolation after objective conformance.

It also needs open primary execution, longer curves, multiple models/tasks/reward types, at least two stacks, matched tokens/FLOPs where efficiency is claimed, powered equivalence tests, versioned treatment fingerprints, and governance for new variants.

**Novelty collision.** DAPO already provides an open end-to-end reproducible system; Dr.GRPO and GSPO show that objective semantics matter; many papers compare subsets of these algorithms. The distinctive opportunity is not another leaderboard but a survival/equivalence methodology with executable semantic certification.

**Rescue condition.** Pair Path 6 with Path 4, keep `INCONCLUSIVE` intact, and demonstrate that semantic certification explains cross-paper survival or reversal. That can be a strong benchmark/systems contribution. Main-track algorithms novelty requires a general empirical law, not only a reliable table.

**Verdict:** **Strongest evidence-backed paper now; strong non-main-track fit, conditional main-track only after major expansion.**

### Path 7 — Boundary-information insufficiency theorem plus causal validation

**Strongest reject argument.** The weak theorem is trivial: for binary rewards, `[0,...,0]` and `[1,...,1]` both have zero variance and ZVF=1, so a controller that observes only ZVF must take the same action. Reviewers already know this, DAPO filters both endpoints, AERO conditions on correctness, and AVSPO reports separate all-wrong/all-correct interventions. Calling non-injectivity an “information insufficiency theorem” without a decision consequence will look inflated.

**Exact missing proof.** The defensible theorem must do substantially more:

- define a decision problem with state, allowed observation, action set (retry/keep/retire/intervene), compute cost, and learning utility;
- construct observationally equivalent environments under ZVF but different Bayes-optimal actions and prove a positive minimax-regret or impossibility lower bound for every ZVF-only controller;
- characterize a minimal additional statistic that is sufficient under explicit assumptions—e.g. correct count or mean reward together with G and uncertainty—and say when that statistic ceases to be sufficient under correlated rollouts, verifier noise, graded rewards, or policy drift;
- connect the lower bound to an estimable quantity and a preregistered action rule. Merely appending mean reward to the existing controller is not a theorem.

**Exact missing experiment.** Randomize action at both boundary types under matched token/FLOP budgets; estimate lower-boundary retry and upper-boundary retire effects separately; compare ZVF-only, failure-only, fully boundary-aware, and posterior allocators; measure direct objective/gradient consequences; and replicate the predicted sign across TRL/verl and binary/graded rewards. The current H3 “different decisions in >5% of eligible steps” is too weak to validate an information lower bound.

**Novelty collision.** AVSPO's Error-Only/Correct-Only/full mechanism isolation is the sharpest empirical collision; AERO already performs correctness-conditioned rescue; HORA's posterior hit utility supplies a principled missing statistic; DAPO operationalizes both endpoints. The safe novelty is a formal information/regret boundary plus causal validation—not the fact that the endpoints differ.

**Rescue condition.** Make this the theory spine of Path 4: prove the loss of action-relevant information, certify the statistic and action semantics in code, then show that restoring exactly that information changes gradients and learning utility as predicted. A result that survives verifier noise or graded rewards would be much more defensible than a Bernoulli-only lemma.

**Verdict:** **Most promising new idea, but main-track credible only in its strong lower-bound + causal form. Weak form is a no-go.**

## Main-track decision matrix

| Path | Main-track novelty now | Conditional ceiling | Go/no-go |
|---|---|---|---|
| 1. Mechanism-only ZVF | None: directly occupied | Workshop/appendix unless a new cross-regime causal law appears | **Drop standalone** |
| 2. One-formula asymmetric controller | Not earned | Main track only with a decisive simplicity-adjusted win over current external allocators | **Run minimal test; do not promise paper** |
| 3. Maximal controller | None for current protocol | Benchmark/systems program after major redesign | **Stop** |
| 4. Causal cross-stack audit | Plausible gap, zero outcome evidence | Main-track analysis if a general divergence and causal repair are found | **Priority 1 empirical gate** |
| 5. Unified chain | No current claim | Highest ceiling if all causal and competitive gates pass | **Earned flagship only** |
| 6. Algorithm audit benchmark | Strong artifact novelty, weak algorithm novelty | Excellent benchmark/MLSys/TMLR; main track after broad causal expansion | **Package with Path 4** |
| 7. Information insufficiency + causality | Strong idea only in nontrivial form | Credible main-track theory/empirics spine with regret lower bound and causal validation | **Priority 1 theory route** |

## Required pre-compute corrections to the scientific story

These are review conclusions, not edits to the frozen protocol:

1. **Do not treat `ZVFController` as the registered treatment.** Implement and differential-test the six action policies explicitly.
2. **Separate protocol validity from competitive sufficiency.** The frozen v1 six-arm result can answer H2/H3 as written, but it cannot support a “best current allocator” claim. Because no scored run has begun, any added external comparison should be a new, clearly versioned preregistration or a separately frozen study—not a silent change.
3. **Use external baselines at their faithful granularity.** AVSPO changes advantages, AERO changes rollout and curation, VIP/HORA change allocation, and CoDaPO changes weighting/resampling. A superficial emulation is not a fair comparison.
4. **Make S1 a paper-grade causal gate.** Test mathematical identity, intended differences, negative controls, and bf16/distributed realization; then demonstrate that at least one certified semantic delta changes training in the predicted direction.
5. **Predefine the strong Path 7 theorem before looking at controller outcomes.** Otherwise the information-theory narrative will read as a post-hoc explanation for whichever arm wins.
6. **Charge all costs.** Generated tokens, rejected and probe rollouts, environment executions, controller inference, forward/backward FLOPs, wall time, and memory must enter the quality-compute frontier.

## Final recommendation

The safest program is not “finish the controller paper.” It is:

1. complete S1 as an independently publishable causal conformance study;
2. formalize Path 7 as a real decision-information lower bound, or drop the theorem language;
3. implement only the minimal asymmetric rule and preserve the frozen internal screen;
4. add a separately frozen contemporary-baseline study before making any main-track algorithm claim;
5. promote Path 5 only if prediction, causal treatment, controller utility, external competitiveness, and cross-stack replication all pass.

As of 2026-07-20, **Path 7 + Path 4 is the only route with a defensible, uncrowded scientific center.** Path 5 remains a legitimate conditional flagship, but the probability that all its gates pass *and* survive AVSPO/VIP/HORA/AERO-class comparison is low. E1 should be published for what it is—a rigorous bounded audit—not used to imply that a new controller has already been validated.

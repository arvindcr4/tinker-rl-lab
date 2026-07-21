# Flagship user-fit panel: simulated stakeholder interviews

**Date:** 2026-07-20

**Status:** structured persona simulation, not human-subject research

**Decision:** which scientific path has both defensible literature differentiation and enough user value to justify the flagship experiment?

## Executive decision

The panel's strongest user-fit path is **a one-formula, reward-aware asymmetric controller, with the causal objective/gradient audit as a mandatory trust layer**. The controller is the thing users can act on; the audit is the proof that its measured behavior is real. Exact ZVF theory should support the controller's prediction and calibration, but is too weak as a standalone user proposition. A reusable algorithm-audit benchmark is the best secondary artifact, not the main scientific headline.

This is deliberately narrower than a “maximal general controller.” The literature already contains dynamic sampling, pre-rollout filtering, Bayesian rollout allocation, and advantage shaping for zero-variance prompts [W1–W4]. A new controller earns differentiation only if it demonstrates that **all-wrong starvation and all-correct saturation require different actions**, beats static `G=16` and the best simple heuristic at matched generated tokens, and retains the effect across reward regimes and stacks. Those conditions are already close to the preregistered H2–H4 gates.

Panel-wide option ranking (lower mean rank is better):

| Rank | Candidate path | Mean rank across 8 personas | User-fit reading |
|---:|---|---:|---|
| 1 | **B. One-formula reward-aware asymmetric controller** | **1.75** | Clearest operational decision and compute value; adoption depends on simplicity and head-to-head evidence. |
| 2 | **D. Causal objective/gradient audit** | **2.00** | Highest trust and maintainer value; essential validation layer, but less direct end-user performance value on its own. |
| 3 | **E. Unified theory + intervention + controller** | **3.13** | Best paper narrative if aggressively bounded; easily becomes an overclaimed bundle. |
| 4 | **F. Algorithm-audit benchmark** | **3.38** | Strong artifact and community value; expensive to maintain and a weaker flagship mechanism claim. |
| 5 | **A. Exact ZVF theory only** | **5.13** | Useful diagnostic/calibration result, but low adoption value and high “known phenomenon, new notation” risk. |
| 6 | **C. Maximal general controller** | **5.63** | Universally disliked: too many knobs, hard to debug, difficult to attribute, and easy to collide with adjacent work. |

The recommended product-shaped claim is:

> Given a prompt's observed reward outcomes and uncertainty, a small asymmetric rule chooses among **retry/escalate**, **keep**, and **retire**. It predicts and increases useful-gradient yield while preserving held-out quality under a fixed generated-token budget. Every action path is checked against a canonical float64 objective/gradient reference on TRL and verl.

Do not market this as “solving zero variance.” That problem is already explicit in DAPO, GRESO, AERO, and RL-ZVP [W1–W4]. Market it, if the evidence passes, as **boundary-aware compute routing with causal execution proof**.

## Method and honesty boundary

No real people were interviewed. The “quotes” below are concise simulated utterances from eight role personas. They are inference devices, not testimony and not evidence of market demand. Each persona is grounded in public primary sources—papers, official framework documentation, and official reviewing/artifact criteria—and in the frozen local evidence. The simulation asks the same three questions of every persona:

1. **Problem discovery:** What job is this person trying to complete, and what currently blocks it?
2. **Reaction to options:** Which candidate paths help, confuse, or fail that job?
3. **Willingness to adopt:** What concrete proof would change behavior?

The resulting rankings are transparent judgments. They are not survey statistics. Mean ranks summarize this panel only and must not be reported as population preferences.

## Evidence anchor: what the project actually has

### Frozen local evidence

The panel was constrained by the following repository facts:

- The completed E1/R08 audit is `COMPLETE`: 40 arm–seed units, eight paired seeds for GRPO, DAPO, GSPO, Dr.GRPO, and AERO, all under one Qwen3-8B/GSM8K/Tinker stack.
- Against GRPO, DAPO's held-out delta is `+0.0010` with 95% CI `[-0.0045, +0.00675]` and the preregistered verdict `DISAPPEARS`. GSPO (`+0.0050`), Dr.GRPO (`-0.0020`), and AERO (`-0.00075`) are `INCONCLUSIVE`; no arm collapsed.
- E1 is a survival audit, not a controller validation or general leaderboard. Its scope is one model, task, stack, 30-step horizon, and binary exact-match reward.
- E1 also exposes the trust problem: a prior loss flag was present but unwired, and recovery/finalizer bookkeeping defects required explicit correction receipts. Plausible outcome traces were not sufficient proof that the intended algorithm executed.
- The flagship protocol is frozen but screening has not started. S1 requires float64 losses, masks, importance ratios, action selections, and flattened gradients to agree across a canonical reference, TRL, and verl before any GPU screening.
- The preregistered controller comparison already includes static `G=8`, static `G=16`, symmetric ZVF escalation, all-wrong-only escalation, boundary-aware routing, and full triage. It covers GSM8K binary reward, MATH-500 sparse binary reward, and MBPP graded executable reward.
- Expansion requires positive matched-budget learning-curve AUC versus static `G=16` and the best naive arm, final-quality non-inferiority on every task, and at least a 10% improvement in useful-gradient fraction or tokens/FLOPs-to-target.

Local source surfaces:

- `zvf-program/flagship/preregistration.json`
- `zvf-program/flagship/README.md`
- `zvf-program/audit/results/audit.json`
- `zvf-program/audit/reproducibility_audit.tex`
- `execution-notes.md`

### Web-verified public pain points

| Public evidence | Pain point relevant to this panel | Implication for differentiation |
|---|---|---|
| DAPO uses dynamic sampling to filter groups with no reward variance and spends additional sampling to obtain informative groups [W1]. | Flat groups waste optimizer opportunities, but “filter/resample zero variance” is established. | A symmetric retry controller is not novel enough. |
| GRESO predicts zero-variance prompts before rollout to avoid sampling overhead [W2]. | Avoiding doomed rollout work is already a direct efficiency target. | A pre-filter alone collides with prior work. |
| AERO uses adaptive rollout allocation, a Bayesian success-rate posterior, and selective sample handling, reporting lower compute and wall time [W3]. | Users want useful signal per unit of compute, not ZVF telemetry for its own sake. | A complex adaptive allocator needs a sharply different action rule and matched-budget comparison against AERO-like behavior. |
| RL-ZVP argues that flat prompts can be made informative through advantage shaping rather than discarded [W4]. | “Zero variance means useless” is not universally valid once the learning objective is changed. | Claims must be objective-conditional; the controller must distinguish observation, intervention, and target objective. |
| Dr.GRPO identifies optimization biases caused by reward standard-deviation and response-length normalization [W5]. | Objective details can dominate apparent algorithm behavior. | Objective/gradient equivalence is a real user need, not appendix polish. |
| TRL exposes many GRPO knobs: reward functions, reward scaling, loss types, importance-sampling levels, vLLM modes, and distributed settings [W6]. | Flexible frameworks create a large semantic configuration surface. | A small executable fixture suite is easier to adopt than another prose checklist. |
| HybridFlow/verl and OpenRLHF optimize flexible algorithm expression, device placement, generation/training throughput, and usability [W7, W8]. | Maintainers and infrastructure owners optimize both semantic flexibility and systems efficiency. | Controller overhead and integration burden must be measured, not assumed away. |
| AReaL shows that synchronous rollout/training coupling wastes accelerator time and introduces a freshness–throughput tradeoff [W9]. | At scale, a policy that changes rollout allocation interacts with scheduling and staleness. | Static single-worker gains are insufficient for large-lab adoption. |
| Open-R1 documents reproducibility work plus practical response-length and training-configuration challenges [W10]. | Small/open teams need recipes that fit existing trainers and do not multiply tuning burden. | The simple controller and audit fixtures have more fit than a maximal controller. |
| DeepSeek-R1 reports that pure RL can improve reasoning while creating readability and language-mixing failures [W11]. | Outcome accuracy alone does not capture all applied quality regressions. | Adoption proof needs quality guardrails beyond a single exact-match score. |
| NeurIPS asks authors to expose assumptions, limitations, reproducibility details, statistical uncertainty, and compute [W12]. | Reviewers need claims whose evidence boundary is auditable. | A preregistered multi-stack, multi-regime result is more credible than a broad controller story. |
| ICLR's official reviewer guide emphasizes sound, constructive assessment; the review form asks for soundness, presentation, significance, and originality [W13]. | A crowded idea needs a crisp originality statement plus technically sound evidence. | “ZVF controller” without nearest-neighbor contrasts will be read as incremental. |
| ACM artifact criteria separate availability, functionality, reusability, and result reproduction [W14]. | An artifact evaluator wants a runnable path, stable inputs, expected outputs, and reusable components. | The audit benchmark has strong value only if packaged independently of the paper's private evidence. |

## Candidate paths presented to the panel

To avoid label drift, every persona ranked the same options:

- **A — Exact ZVF theory only:** formalize the probability/calibration relationship between prompt success, group size, zero-variance events, and useful-gradient availability; no learned controller claim.
- **B — One-formula reward-aware asymmetric controller:** one small decision rule that treats all-wrong and all-correct flat groups differently and chooses retry/escalate, keep, or retire.
- **C — Maximal general controller:** a configurable controller spanning reward types, uncertainty models, schedules, objectives, stacks, and multiple actions.
- **D — Causal objective/gradient audit:** canonical fixtures and differential checks proving that named algorithms and controller actions change exactly their declared loss, masks, ratios, gradients, and group decisions.
- **E — Unified theory + intervention + controller:** a single scientific paper joining prediction, targeted interventions, controller utility, and cross-stack validation.
- **F — Algorithm-audit benchmark:** a reusable benchmark for testing whether GRPO-family improvements survive on a shared stack with frozen provenance and evaluation rules.

## Simulated interview 1: outcome-RL researcher

**Grounding:** zero-variance handling and GRPO objective corrections are active research topics [W1–W5]. This persona values a mechanism that changes scientific understanding, not merely a trainer callback.

### Round 1 — problem discovery

**Simulated response:** “I need to know when group-relative RL is starved of comparative signal, whether the cause is task difficulty or saturation, and which intervention follows from that diagnosis. I do not need another name for flat rewards.”

- **Job to be done:** explain and predict when an outcome-reward policy receives useful within-group contrast, then identify a falsifiable intervention.
- **Current blockers:** reward density, group size, parser behavior, objective normalization, and policy skill are entangled; many papers report final accuracy without the signal path.
- **Adoption value:** a calibrated mechanism can improve experiment design and prevent false algorithm stories.
- **Confusing terminology:** “zero-variance” can mean reward variance, advantage variance, gradient variance, or batch variance; “useful gradient” must be defined operationally; “boundary-aware” is opaque without naming the two boundaries.

### Round 2 — reaction to options

- **E** is most scientifically attractive if the intervention directly tests the theory and the controller is a consequence rather than a bag of heuristics.
- **B** is attractive because the all-wrong/all-correct asymmetry is a crisp empirical bet, but it must be contrasted with DAPO, GRESO, AERO, and RL-ZVP.
- **D** is necessary because an objective mismatch can manufacture the mechanism.
- **A** is acceptable as a bounded theory paper only if it predicts unseen task/model/reward-regime cells; the binary identity alone feels tautological.
- **F** is useful infrastructure but does not answer the mechanism question.
- **C** is the least attractive because too many adaptive components destroy attribution.

**Option ranking:** `E > B > D > A > F > C`.

### Round 3 — willingness to adopt / required proof

**Willingness:** would cite and build on E or B; would use D to validate implementations.

**Must-have evidence:**

1. prospective calibration on held-out task/model/group-size cells, not a post-hoc fit;
2. explicit decomposition of all-wrong vs all-correct flat events;
3. an intervention that changes the predicted mediator (useful-gradient yield) before claiming final-quality gains;
4. seed-level inference and learning curves under equal generated-token budgets;
5. nearest-neighbor table versus DAPO, GRESO, AERO, and RL-ZVP;
6. negative-result path showing what remains publishable if the controller does not win.

**Hard blocker:** if the “one formula” is only a threshold wrapper around observed ZVF and is not prospectively calibrated, it will not change research practice.

## Simulated interview 2: open-source framework maintainer

**Grounding:** TRL exposes many semantic and systems choices, while verl/HybridFlow and OpenRLHF emphasize flexible expression and efficient execution [W6–W8].

### Round 1 — problem discovery

**Simulated response:** “My failure mode is not lack of clever algorithms. It is accepting a feature whose CLI says one thing while its loss, mask, sampler, or distributed path does another.”

- **Job to be done:** merge algorithm extensions without silently changing unrelated behavior, keep APIs stable, and diagnose regressions across backends.
- **Current blockers:** incomplete reference fixtures, multiple execution modes, distributed/non-distributed divergence, ambiguous configuration names, and tests that check outputs but not gradients.
- **Adoption value:** objective-level differential tests reduce maintainer review burden and downstream bug reports.
- **Confusing terminology:** “canonical objective” needs a versioned mathematical contract; “stack matched” must enumerate tolerated differences; `ZVF` is not an intuitive public API name.

### Round 2 — reaction to options

- **D** solves the maintainer's highest-cost problem and can be accepted independently of a controller win.
- **B** is implementable if it is a stateless or small-state hook with bounded overhead and no sampler fork.
- **F** is useful if fixtures become conformance tests that external frameworks can run.
- **E** is too paper-shaped unless its components are separable packages.
- **C** creates a permanent compatibility and support surface.
- **A** does not reduce implementation risk.

**Option ranking:** `D > B > F > E > C > A`.

### Round 3 — willingness to adopt / required proof

**Willingness:** high for D as tests; conditional for B as an optional integration; low for C.

**Must-have evidence:**

1. tiny deterministic CPU fixtures plus float64 reference outputs;
2. checks for losses, token masks, importance ratios, selected group sizes, and flattened gradients;
3. treatment-path coverage proving every option differs from baseline on a targeted fixture and matches on frozen non-treatment fields;
4. TRL and verl adapters with a stable minimal API;
5. tests for empty/malformed reward groups, graded rewards, resumptions, and distributed aggregation;
6. microbenchmarks for wall-time and memory overhead.

**Hard blocker:** a controller that requires patching core generation loops differently per stack rather than using supported extension points.

## Simulated interview 3: large-lab scaling researcher

**Grounding:** AReaL, HybridFlow, and OpenRLHF focus on accelerator utilization, asynchronous generation/training, placement, and throughput [W7–W9]. AERO already claims major compute savings from adaptive rollout allocation [W3].

### Round 1 — problem discovery

**Simulated response:** “I care about tokens and wall time to a quality target across hundreds of workers. A controller that saves rollouts but introduces bubbles, stale trajectories, or irregular batches can lose at system scale.”

- **Job to be done:** increase learning utility per accelerator-hour while preserving convergence under distributed or asynchronous execution.
- **Current blockers:** generation/training imbalance, stragglers, variable-length completions, staleness, resharding, and controllers whose logical savings do not become cluster savings.
- **Adoption value:** a low-overhead allocation rule could redirect rollout capacity to useful prompt-steps without destabilizing scheduling.
- **Confusing terminology:** “matched compute” must separate generated tokens, optimizer FLOPs, environment executions, accelerator-hours, and wall time; “FLOPs-to-target” must say how FLOPs are measured.

### Round 2 — reaction to options

- **B** is best because it could be implemented as a cheap routing signal and scheduled at scale.
- **D** is essential for ensuring sharded/asynchronous paths implement the same objective.
- **E** is interesting if the controller remains small and the cross-stack result survives scale-up.
- **F** helps compare algorithms, but maintaining a benchmark is not the immediate scaling objective.
- **C** adds scheduling entropy and tuning cost.
- **A** does not by itself improve utilization.

**Option ranking:** `B > D > E > F > C > A`.

### Round 3 — willingness to adopt / required proof

**Willingness:** would pilot B after open-stack single-node proof; would require a scale-aware systems appendix before production.

**Must-have evidence:**

1. equal generated-token ceilings plus realized optimizer FLOPs, GPU-hours, wall time, peak memory, and environment executions;
2. throughput traces showing that controller decisions do not create device bubbles or severe batch fragmentation;
3. sensitivity to asynchronous policy lag and stale success-rate estimates;
4. results at two model scales and more than one completion-length distribution;
5. failure behavior when reward parsers are delayed or partially unavailable;
6. direct comparison to AERO-like adaptive allocation, not only static group size.

**Hard blocker:** savings visible only in nominal rollout counts while end-to-end wall time or cost worsens.

## Simulated interview 4: small academic lab

**Grounding:** Open-R1 exists to make the R1 pipeline reproducible and highlights response-length and training-configuration challenges [W10]. Public systems papers repeatedly trade usability against performance [W7, W8].

### Round 1 — problem discovery

**Simulated response:** “I have a small number of GPUs and cannot burn a week discovering that most groups had no contrast or that the implementation flag was inert. Give me an early warning and a safe default.”

- **Job to be done:** decide whether an RLVR run is worth continuing and choose a defensible group/rollout policy within a fixed budget.
- **Current blockers:** limited seeds, expensive long completions, fragile environments, unclear defaults, and insufficient budget for a large hyperparameter sweep.
- **Adoption value:** an early-stop/route signal and executable conformance tests protect scarce compute.
- **Confusing terminology:** `full_triage`, `Wilson uncertainty gating`, and `boundary-aware` sound like a complicated subsystem; “retire mastered prompts” is clearer than “all-correct saturation action.”

### Round 2 — reaction to options

- **B** is the only controller path likely to fit the lab's tuning budget.
- **D** prevents wasting the entire allocation on an invalid implementation.
- **F** is valuable if it ships small fixtures and reference runs rather than requiring a cluster.
- **A** can guide planning but does not tell the lab what to do next.
- **E** looks expensive and may hide which component mattered.
- **C** is unusable without a large sweep budget.

**Option ranking:** `B > D > F > A > E > C`.

### Round 3 — willingness to adopt / required proof

**Willingness:** high if B is one config block in TRL/verl and has a fail-safe fallback to static `G`.

**Must-have evidence:**

1. 1.7B-scale screening with total GPU-hour and storage accounting;
2. default thresholds that transfer across at least the three preregistered reward regimes;
3. a dashboard or log line translating decisions into “retry hard prompt,” “keep,” or “retire mastered prompt”;
4. checkpoint-safe resume and deterministic replay of controller state;
5. quality non-inferiority, not just compute savings;
6. a small reference recipe reproducible on commonly available academic hardware.

**Hard blocker:** per-task retuning that costs more rollouts than the controller saves.

## Simulated interview 5: reproducibility chair / artifact evaluator

**Grounding:** NeurIPS explicitly asks for assumptions, uncertainty, compute, code/data access, and reproduction details [W12]. ACM distinguishes available, functional, reusable, and reproduced artifacts [W14].

### Round 1 — problem discovery

**Simulated response:** “I need to determine whether an independent evaluator can recover the claimed result, understand deviations, and reuse the core artifact without access to the authors' private training history.”

- **Job to be done:** verify provenance, execute a bounded workflow, compare outputs with expected checksums/tolerances, and assess reusability.
- **Current blockers:** missing model/data revisions, mutable remote runs, undocumented recovery, huge compute requirements, and claims that depend on private dashboards.
- **Adoption value:** a fail-closed benchmark with small conformance fixtures plus immutable evidence receipts maps directly to artifact evaluation.
- **Confusing terminology:** “survival” might sound like universal replication; “independently accepted” needs a named validator; “private Hub” is provenance, not public availability.

### Round 2 — reaction to options

- **F** is the most natural artifact contribution if it contains reusable, open, bounded evaluation units.
- **D** provides the functional core and deterministic oracle.
- **B** is assessable if the intervention is separable and replayable.
- **E** can be strong but risks coupling artifact evaluation to a large GPU matrix.
- **A** is easy to verify but low-value as an artifact.
- **C** explodes the evaluation matrix.

**Option ranking:** `F > D > B > E > A > C`.

### Round 3 — willingness to adopt / required proof

**Willingness:** high for F/D if the public package can be evaluated without private credentials; conditional for the full result-reproduction badge.

**Must-have evidence:**

1. exact model, dataset, stack, container, CUDA/PyTorch, parser, and prompt-split fingerprints;
2. a one-command smoke path, a bounded CPU conformance path, and a documented GPU reproduction path;
3. expected outputs, numerical tolerances, checksums, and explicit failure messages;
4. immutable run manifests and correction/resume receipts;
5. license and data-access instructions;
6. a statement separating “artifact functional” from “all expensive claims reproduced.”

**Hard blocker:** an artifact whose critical evidence is accessible only through the authors' private W&B or Hugging Face accounts.

## Simulated interview 6: conference reviewer

**Grounding:** official NeurIPS and ICLR materials foreground soundness, originality/significance, limitations, reproducibility, and statistical evidence [W12, W13]. The nearest-neighbor literature is already dense [W1–W5].

### Round 1 — problem discovery

**Simulated response:** “I need one claim I can state precisely, distinguish from four close papers, and verify from the experiment design. A six-part system with one positive aggregate number is difficult to trust.”

- **Job to be done:** judge originality, technical correctness, significance, and whether the evidence supports the advertised scope.
- **Current blockers:** crowded terminology, post-hoc controller choices, too many endpoints, low seed counts, and baselines that are not compute matched.
- **Adoption value:** a preregistered causal chain—prediction, action disagreement, useful-gradient mediation, and held-out utility—could be a strong contribution.
- **Confusing terminology:** `ZVF` needs expansion at first use; “exact” must not imply universal; “causal audit” can be mistaken for causal inference rather than intervention-controlled execution testing.

### Round 2 — reaction to options

- **E** offers the strongest main-track arc if it is one mechanism and one controller, not a platform paper.
- **D** is unusually credible because it directly addresses silent objective misexecution, but needs general value beyond one local incident.
- **B** is clean and useful, but novelty depends on asymmetric actions and action-disagreement evidence versus AERO/DAPO/GRESO.
- **F** fits an artifact/dataset track better than a mechanism main track.
- **A** risks being a simple identity around an already-recognized phenomenon.
- **C** is very hard to review causally.

**Option ranking:** `E > D > B > F > A > C`.

### Round 3 — willingness to adopt / required proof

**Willingness:** likely positive only if E is bounded to B + D + predictive theory and all gates pass; otherwise prefers a focused D or B paper.

**Must-have evidence:**

1. explicit contribution matrix against DAPO, GRESO, AERO, RL-ZVP, Dr.GRPO, and relevant framework conformance work;
2. frozen hypotheses and disjoint screening/confirmatory seeds;
3. seed as the independent unit, paired intervals, multiplicity handling, and honest `INCONCLUSIVE` labels;
4. ablations that separate asymmetric boundary information from uncertainty gating and compute-aware scheduling;
5. matched-token and measured-FLOP comparisons, plus non-inferior final quality;
6. a results-independent negative route.

**Hard blocker:** claiming general RL efficiency from Qwen/GSM8K/Tinker-only evidence or treating a confidence interval crossing zero as equivalence.

## Simulated interview 7: applied reasoning-model engineer

**Grounding:** DeepSeek-R1 reports that RL can improve reasoning while also creating readability and language-mixing failures [W11]. Open-R1 reports practical issues with long responses and training configurations [W10].

### Round 1 — problem discovery

**Simulated response:** “I need the model to get better on the verifier without becoming longer, less readable, reward-hacky, or brittle on held-out tasks. Training efficiency is valuable only behind those guardrails.”

- **Job to be done:** improve reasoning accuracy reliably while controlling response quality, length, and operational cost.
- **Current blockers:** sparse or hackable verifiers, long-tail completion lengths, reward-parser bugs, unstable generalization, and metrics that miss qualitative regressions.
- **Adoption value:** routing rollout effort away from mastered prompts and toward uncertain failures may improve both speed and curriculum quality.
- **Confusing terminology:** “all-wrong” may include parser failure, environment failure, or genuinely hard prompts; “retire” must not mean permanently forgetting examples; “quality target” must name the held-out metric.

### Round 2 — reaction to options

- **B** maps directly to a training decision and is easiest to reason about operationally.
- **D** protects against parser/objective/integration bugs.
- **E** is valuable if it demonstrates why the action works and includes quality guardrails.
- **F** helps vendor/framework comparison but is secondary to shipping a model.
- **C** introduces too many knobs into an already fragile pipeline.
- **A** is interesting but not sufficient for adoption.

**Option ranking:** `B > D > E > F > C > A`.

### Round 3 — willingness to adopt / required proof

**Willingness:** would shadow-run B and compare controller decisions before allowing it to change allocation.

**Must-have evidence:**

1. held-out accuracy/AUC and tokens-to-target, plus response length, KL, entropy, clip fraction, and reward-hacking checks;
2. separate treatment of parser failure from genuine zero reward;
3. replayable per-prompt decisions with reason codes;
4. evidence on binary math, sparse math, and graded executable rewards;
5. rollback to static policy when uncertainty or telemetry quality is insufficient;
6. no regression at final quality under the common token ceiling.

**Hard blocker:** the controller improves verifier reward while worsening true task success, response quality, or operational latency.

## Simulated interview 8: compute / infrastructure owner

**Grounding:** HybridFlow, OpenRLHF, and AReaL measure success in throughput, placement efficiency, utilization, and wall-clock time—not merely algorithmic rollout counts [W7–W9].

### Round 1 — problem discovery

**Simulated response:** “I allocate accelerators against a budget and SLA. I need predictable memory, checkpointing, utilization, and recovery. An adaptive policy is welcome only if its savings survive the scheduler.”

- **Job to be done:** deliver a target-quality run within accelerator-hour, storage, and reliability constraints.
- **Current blockers:** variable completion lengths, OOMs, idle inference/training workers, checkpoint churn, environment bottlenecks, and non-reproducible resumes.
- **Adoption value:** fewer wasteful generations and earlier stopping can reduce cost, provided controller state is cheap and recoverable.
- **Confusing terminology:** “compute-aware” is meaningless without an accounting model; “matched budget” needs explicit ceilings and a treatment of failed/resumed work.

### Round 2 — reaction to options

- **B** offers the clearest path to resource savings.
- **D** reduces expensive reruns caused by silent semantic failure.
- **F** supports procurement and stack comparison if resource metrics are standardized.
- **E** is acceptable only after the components demonstrate independent value.
- **A** gives planning information but no direct savings.
- **C** threatens predictability and test-matrix size.

**Option ranking:** `B > D > F > E > A > C`.

### Round 3 — willingness to adopt / required proof

**Willingness:** conditional production pilot after deterministic shadow mode and a bounded failure policy.

**Must-have evidence:**

1. generated tokens, optimizer FLOPs, accelerator-hours, wall time, environment executions, peak memory, checkpoint bytes, and retry waste;
2. less than 1% token mismatch for fixed-budget comparisons or a separately labeled tokens-to-target endpoint;
3. exact-resume receipts that preserve prompt order and controller state;
4. behavior under worker loss, delayed rewards, and partial batches;
5. scheduler-visible reduction in cost, not merely logical action counts;
6. no purchase of extra capacity hidden inside the claimed efficiency result.

**Hard blocker:** irregular controller actions reduce utilization or make cost forecasting materially worse.

## Cross-panel extraction

### Jobs to be done

Six jobs recur across personas:

1. **Diagnose:** tell whether a flat group is a hard-prompt failure, mastered-prompt saturation, parser failure, or an objective artifact.
2. **Decide:** turn the diagnosis into a small, inspectable action—retry/escalate, keep, or retire.
3. **Trust:** prove the named objective and controller action actually executed across training stacks.
4. **Budget:** improve quality per generated token and accelerator-hour, not only a proxy metric.
5. **Compare:** separate algorithm deltas from stack deltas with shared fixtures and frozen provenance.
6. **Reproduce:** give a bounded public path from configuration to expected output and correction/resume receipt.

### Blocking conditions that recur

- **Novelty collision:** DAPO, GRESO, AERO, and RL-ZVP already act on flat or predicted-flat groups [W1–W4].
- **Semantic ambiguity:** reward variance, advantage variance, and gradient usefulness are repeatedly conflated.
- **Proxy/value mismatch:** fewer rollouts may not mean less wall time, lower cost, or better learning.
- **Objective dependence:** a flat outcome group is uninformative for standard within-group centering, but RL-ZVP explicitly changes advantage shaping [W4].
- **Integration risk:** a controller can change sampler, prompt order, token budget, or distributed aggregation along with its declared action.
- **Evaluation burden:** a maximal controller creates an unreviewable ablation and compatibility matrix.
- **Private-evidence ceiling:** immutable private W&B/Hub records strengthen provenance but do not satisfy public artifact availability.
- **Scope drift:** E1's single-stack 30-step result cannot support universal controller or framework claims.

### Must-have evidence before a main-track claim

The panel converges on the following proof stack:

1. **S1 objective differential passes:** canonical float64 reference agrees with TRL and verl on loss, masks, ratios, actions, and flattened gradients.
2. **Prospective prediction:** a pre-fit starvation model calibrates on unseen task/model/group-size cells and distinguishes all-wrong from all-correct events.
3. **Action disagreement:** the asymmetric controller makes materially different decisions from symmetric ZVF, failure-only retry, and AERO-like allocation. H3's `<95% same actions` condition is a minimum; publish the confusion matrix.
4. **Mediator movement:** controller use increases useful-gradient group fraction or decreases tokens/FLOPs-to-target before final capability is interpreted.
5. **Matched resource accounting:** generated tokens are the primary match, with measured optimizer FLOPs, environment executions, GPU-hours, wall time, memory, storage, and retry waste reported.
6. **Learning utility:** seed-paired learning-curve AUC improves versus static `G=16` and the best naive arm; final quality is non-inferior in every preregistered cell.
7. **Reward-regime breadth:** binary moderate, sparse binary, and graded executable rewards; parser/environment failures reported separately from genuine all-wrong groups.
8. **Scale and stack:** two model scales for confirmation and a sign-preserving secondary-stack replication after differential tests.
9. **Quality guardrails:** response length, KL, entropy, clip fraction, parser validity, reward-hacking probes, and task success beyond raw training reward.
10. **Reusable artifact:** CPU fixtures, one-command smoke test, expected outputs/tolerances, immutable manifests, exact-resume receipts, and a public package that does not require private credentials.

### Terminology to change before external presentation

| Current term | Likely confusion | Preferred external phrasing |
|---|---|---|
| `ZVF` | Unknown acronym; may imply gradient variance. | “fraction of prompt groups with identical rewards”; introduce `ZVF` only after the phrase. |
| `useful gradient` | Sounds normative and objective-independent. | “non-zero within-group comparative signal under the frozen objective,” plus the exact measured criterion. |
| `boundary-aware` | Boundary of what? | “different actions for all-wrong and all-correct groups.” |
| `full triage` | Product/medical metaphor; hides components. | “uncertainty-gated asymmetric rollout routing.” |
| `retire` | May imply permanent data deletion. | “skip currently mastered prompts until the recheck condition.” |
| `causal objective/gradient audit` | Can be mistaken for causal inference. | “intervention-controlled objective and gradient conformance tests.” |
| `survival` | May sound universally replicated. | “effect under a frozen shared stack.” |
| `matched compute` | Ambiguous unit. | Name “matched generated tokens” and report measured FLOPs/GPU-hours separately. |
| `exact ZVF theory` | “Exact” may be read as universally complete. | “exact binary-reward accounting plus held-out calibration tests.” |
| `maximal general controller` | Signals complexity without a user outcome. | Do not use externally; specify each action and supported reward regime. |

## Literature differentiation test by option

### A. Exact ZVF theory only

**Differentiation risk: high.** The field already recognizes homogeneous-reward groups as an efficiency problem and directly intervenes on them [W1–W4]. A binary probability identity is unlikely to be sufficient as a flagship contribution.

**Could become defensible if:** the theory predicts held-out zero-variance and useful-signal rates across group sizes and reward regimes; quantifies calibration failure under correlated/non-binary rewards; and produces a decision-relevant bound that existing methods do not provide.

**User value:** experiment planning, early diagnosis, and a negative-result mechanism paper. **Adoption ceiling:** theory alone does not change a training loop.

### B. One-formula reward-aware asymmetric controller

**Differentiation risk: medium, but tractable.** DAPO filters flat groups, GRESO avoids predicted-flat prompts, AERO adaptively reallocates rollouts, and RL-ZVP reshapes advantages [W1–W4]. The novel bet is not adaptation; it is **a minimal rule whose action depends on which reward boundary produced flatness**, with uncertainty and compute cost made explicit.

**Required differentiation proof:**

- formal decision rule and action surface;
- action-disagreement matrix against symmetric retry, all-wrong retry, static `G`, and an AERO-like posterior allocator;
- head-to-head matched-token utility and measured systems cost;
- evidence that all-correct retirement, not merely extra sampling, produces value;
- graded-reward extension defined without forcing binary thresholds.

**User value:** highest. It answers “what should the trainer do now?”

### C. Maximal general controller

**Differentiation risk: very high.** Breadth overlaps multiple existing mechanisms while making causal attribution and maintenance harder.

**Panel verdict:** do not choose this path. Generality should be earned after a small controller works across the three frozen regimes, not declared in version one.

### D. Causal objective/gradient audit

**Differentiation risk: low-to-medium.** Frameworks already emphasize flexible algorithm expression and efficient execution [W6–W9], while Dr.GRPO shows that seemingly small objective details matter [W5]. The local unwired-loss incident gives a concrete failure class. The differentiated artifact is an **executable, cross-stack intervention oracle** that checks exact losses, masks, ratios, gradients, and controller actions—not another reporting checklist.

**Required differentiation proof:** demonstrate at least one realistic silent misexecution caught by the fixtures; show agreement on TRL and verl; version the mathematical contract; and make the test suite easy for maintainers to adopt.

**User value:** extremely high for maintainers, reviewers, and labs avoiding invalid runs. **Adoption ceiling:** it prevents loss rather than directly improving model capability.

### E. Unified theory + intervention + controller

**Differentiation risk: medium-high.** This is the strongest paper shape only if “unified” means one short causal chain:

`predicted boundary state -> asymmetric action -> more comparative signal -> better matched-budget learning utility`.

It fails user fit if it bundles unrelated theory, a maximal controller, a framework, and a benchmark. The recommended bounded version is B as the headline, D as the validity layer, and only the theory needed to make the controller prospective.

### F. Algorithm-audit benchmark

**Differentiation risk: medium.** E1/R08 supplies a strong prototype, and public framework diversity makes the need credible [W6–W9]. But a single-stack private evidence collection is not yet a reusable benchmark.

**Required differentiation proof:** public frozen fixtures; declared-lever taxonomy; stack-diff checks; cheap conformance tier plus expensive reproduction tier; at least two independent framework adapters; clear artifact badges/claims; and governance for new method submissions.

**User value:** high for evaluation and maintenance, moderate for a mechanism-paper audience.

## Recommended flagship packaging

### Scientific headline

Choose **B with D as a non-negotiable foundation**:

> A reward-aware asymmetric rollout rule distinguishes hard-prompt starvation from mastered-prompt saturation and improves learning utility under matched generated tokens. Intervention-controlled conformance tests establish that the objective and all action paths execute identically on two open stacks.

### Paper shape

1. **Problem:** identical-reward groups are common but causally ambiguous; prior work already filters, predicts, reallocates, or reshapes them.
2. **Minimal theory:** predict boundary-specific flat-group probabilities and state the assumptions under which comparative signal vanishes.
3. **Execution proof:** canonical fixtures and TRL/verl gradient/action agreement.
4. **Intervention:** one asymmetric retry/keep/retire rule; uncertainty gating shown separately.
5. **Matched-budget result:** learning-curve AUC, final-quality non-inferiority, tokens/FLOPs-to-target, and systems metrics.
6. **Scope:** three reward regimes, two scales, two stacks; no universal GRPO-family claim.
7. **Negative route:** if prediction holds but control does not win, publish the mechanism and negative controller result; if S1 fails, stop.

### Artifact shape

- Primary artifact: controller + deterministic objective/gradient/action conformance suite.
- Secondary artifact: a small algorithm-audit benchmark layer built from the same fixtures.
- Keep E1/R08 as bounded motivating evidence, never pool it with the prospective flagship analysis.

### Kill criteria from user fit

Stop calling this a controller flagship if any of the following occurs:

- S1 cannot establish objective/action agreement across TRL and verl;
- the asymmetric rule makes the same decision as the best simple heuristic in at least 95% of eligible prompt-steps;
- it needs per-task threshold sweeps whose cost erases the saved rollout budget;
- rollout savings do not improve end-to-end tokens/FLOPs/GPU-hours-to-target;
- final quality is inferior in any frozen task/model cell;
- parser or environment failures account for the apparent all-wrong signal;
- the closest-work table cannot state a concrete action-level difference from AERO, DAPO, GRESO, and RL-ZVP.

## Source URLs

Primary and official sources used to ground the simulations:

- **[W1] DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** Dynamic sampling, clip-higher, token-level loss, and overlong reward shaping. https://arxiv.org/abs/2503.14476
- **[W2] Efficient Reinforcement Learning for LLM Reasoning via Adaptive Prompt Selection (GRESO).** Predicts/filters zero-variance prompts before rollout. https://arxiv.org/abs/2506.02177
- **[W3] Train Less, Learn More: Adaptive Efficient Rollout Optimization for Group-Based Reinforcement Learning (AERO).** Adaptive rollout allocation and success-rate posterior. https://arxiv.org/abs/2602.14338
- **[W4] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping (RL-ZVP).** Changes the learning signal for flat prompts. https://arxiv.org/abs/2509.21880
- **[W5] Understanding R1-Zero-Like Training: A Critical Perspective (Dr.GRPO).** Objective/normalization biases in GRPO-style training. https://arxiv.org/abs/2503.20783
- **[W6] Hugging Face TRL GRPO Trainer documentation.** Public semantic and systems configuration surface. https://huggingface.co/docs/trl/main/en/grpo_trainer
- **[W7] HybridFlow: A Flexible and Efficient RLHF Framework (the framework underlying verl).** Flexible hybrid control, placement, and throughput. https://arxiv.org/abs/2409.19256
- **[W8] OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework.** Usability, Ray/vLLM integration, and training efficiency. https://arxiv.org/abs/2405.11143
- **[W9] AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning.** Generation/training decoupling, utilization, and staleness-aware optimization. https://arxiv.org/abs/2505.24298
- **[W10] Open-R1: Update #1.** Open reproduction work and practical GRPO/long-response training challenges. https://huggingface.co/blog/open-r1/update-1
- **[W11] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** Outcome-RL capability plus readability and language-mixing limitations. https://arxiv.org/abs/2501.12948
- **[W12] NeurIPS Paper Checklist Guidelines.** Official requirements and guidance for assumptions, limitations, uncertainty, compute, and reproducibility. https://neurips.cc/public/guides/PaperChecklist
- **[W13] ICLR 2026 Reviewer Guide.** Official reviewer process and review-quality expectations. https://iclr.cc/Conferences/2026/ReviewerGuide
- **[W14] ACM Artifact Review and Badging training.** Availability, functionality, reusability, and result-reproduction criteria. https://reviewers.acm.org/training-course/artifact-review-and-badging

## Rerun inputs

```yaml
workflow: firecrawl-research-papers
topic: user fit and literature differentiation for ZVF variance-starvation flagship paths
source_constraints:
  - primary papers
  - official framework documentation
  - official conference and artifact-review guidance
panel_personas:
  - outcome-RL researcher
  - open-source framework maintainer
  - large-lab scaling researcher
  - small academic lab
  - reproducibility chair/artifact evaluator
  - conference reviewer
  - applied reasoning-model engineer
  - compute/infrastructure owner
rounds:
  - problem discovery
  - reaction to candidate paths
  - willingness to adopt and required proof
candidate_paths:
  - exact ZVF theory only
  - one-formula reward-aware asymmetric controller
  - maximal general controller
  - causal objective/gradient audit
  - unified theory plus intervention plus controller
  - algorithm-audit benchmark
output: markdown panel with explicit simulation disclaimer, rankings, proof requirements, and source URLs
```

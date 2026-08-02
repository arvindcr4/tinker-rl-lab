# Literature and competitive landscape for the ZVF flagship

Date of search: 2026-07-20  
Scope: 2024--2026 work on GRPO-family objectives, outcome-only RL/RLVR,
zero-variance group degeneracy, adaptive rollout and group sizing, difficulty
curricula, variance reduction, compute-aware RL, and implementation audits.  
Evidence rule: major competitive claims below are tied to primary papers,
official project pages, or official repositories. ArXiv-only work is explicitly
called a preprint; accepted papers are identified only where the primary source
does so.

## Executive decision

The standalone claim that binary-reward group degeneracy occurs with probability

\[
\Pr(\mathrm{ZVF}\mid p,G)=p^G+(1-p)^G
\]

and that unanimous groups contribute no group-relative reward gradient is no
longer a viable flagship novelty. The same degeneracy formula already appears
in 2026 work, the zero-gradient observation is the premise of DAPO, AERO,
Goldilocks RL, GRESO, SPEED-RL, PODS, VIP, and Selective Rollout, and a July 2026
preprint now explicitly unifies GRPO, Dr.GRPO, and DAPO through the group reward
standard deviation. A theory-only ZVF paper therefore has a near-zero competitive
half-life unless it proves substantially more than this Bernoulli identity.

The controller lane is also crowded. DAPO filters zero-variance groups after
sampling; GRESO predicts and skips uninformative prompts before rollout; SPEED-RL
and CDAS target intermediate difficulty; PODS selects maximal-variance subsets;
VIP optimizes prompt-wise rollout allocation under a hard budget; AERO combines
adaptive rollout allocation, rejection, and a Bayesian posterior; AGPO jointly
controls clipping and sampling temperature; Goldilocks uses an adaptive teacher;
and Selective Rollout stops agentic trajectories before their terminal reward.
Any "general adaptive compute controller" without a sharply different estimand
will look like a bundle of already published mechanisms.

The defensible flagship is therefore not *ZVF discovers flat groups*. It is one
of two narrower claims:

1. **Mechanism-to-audit flagship (preferred):** an exact, cross-stack objective
   conformance and causal-gradient audit, followed by an extensible single-stack
   survival benchmark. This directly builds on E1, for which DAPO's gain
   disappears under the shared stack while GSPO, Dr.GRPO, and AERO remain
   inconclusive. The unique object is the *causal attribution protocol*:
   byte/fingerprint-locked stacks, float64 differential fixtures, targeted
   objective interventions, gradient-direction checks, and prospective survival
   verdicts—not a new optimizer name.
2. **Mechanism-to-controller flagship (conditional):** a pre-fitted,
   boundary-aware model that predicts held-out useful-gradient availability and
   uses a deliberately minimal asymmetric action rule: spend more only near a
   recoverable all-wrong boundary; retire mastered all-correct prompts. This is
   credible only if it beats static G16, DAPO-style symmetric rejection,
   failure-only retry, AERO/VIP-like allocation, and a strong difficulty
   curriculum at equal generated tokens and measured FLOPs. Without that win,
   publish the calibration/audit result and do not claim a controller advance.

The preregistered unified route remains scientifically coherent, but it is too
broad to be the default paper shape. Its controller component faces direct 2026
competition and its theory component is already crowded. It should be earned by
the S1--S4 gates, not assumed as the narrative in advance.

## What the repository already establishes

This landscape uses the repository's final, frozen evidence boundary rather
than treating planned work as completed.

- E1 is `COMPLETE`: five arms, eight paired seeds, Qwen3-8B, GSM8K, 30 optimizer
  steps, and a fixed 500-example held-out set under one Tinker/LoRA stack. All 40
  units pass local, W&B, private-Hub, checkpoint, treatment-fingerprint, and
  held-out-hash verification.
- Relative to GRPO, DAPO has delta `+0.00100`, 95% CI
  `[-0.00450,+0.00675]`; its former preregistered `DISAPPEARS` equivalence
  verdict is **superseded** (2026-08-02 exact-t reanalysis: MDE80 0.0101 >
  0.01 margin, and the multiplicity step had never run — see
  `zvf-program/audit/STATISTICAL_REANALYSIS.md`), so DAPO is `INCONCLUSIVE`.
  GSPO is `+0.00500 [-0.00125,+0.01200]`, Dr.GRPO is
  `-0.00200 [-0.00950,+0.00725]`, and AERO is
  `-0.00075 [-0.00825,+0.00675]`; all four arms are `INCONCLUSIVE`. No arm
  collapsed.
- The result is a bounded single-stack survival audit, not a universal algorithm
  ranking. It is short-horizon, one-model, one-task, LoRA, and relies on a closed
  execution stack. This is strong provenance evidence but weak external-validity
  evidence.
- The earlier 505-task binary accounting establishes
  `pass@G - p^G = 1 - ZVF`; the strongest measured group-size result is the
  matched-token G2-vs-G16 panel; `(1-ZVF)/sqrt(G)` selecting G4 is conditional on
  that proxy and cohort; and 92.3% of retrospective controller fires were
  all-correct. Those facts motivate asymmetry, but they do not demonstrate that
  any controller preserves or improves learning.
- The frozen flagship protocol correctly gates new claims: exact TRL/verl
  objective differentials first, then six-arm screening, then disjoint-seed
  confirmatory training, then a secondary-stack replication. Its three reward
  regimes (binary moderate density, binary sparse, and graded executable reward)
  are important because nearly all simple ZVF theory assumes Bernoulli rewards.

## Field map: what changed from 2024 to 2026

### 1. GRPO became a baseline, not a novelty

DeepSeekMath introduced GRPO as a critic-free PPO variant for mathematical
reasoning and memory efficiency ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).
DeepSeek-R1 then made large-scale outcome-verifier RL a central reasoning recipe,
including R1-Zero without an SFT warm start
([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)). By 2025--2026, the
research question had moved from "does GRPO work?" to which normalization,
importance-ratio granularity, clipping, sampling, and data-allocation choices
actually cause the result.

The official implementations reinforce that point. TRL exposes a GRPO trainer
([official repository](https://github.com/huggingface/trl)); verl supports GRPO,
DAPO, Dr.GRPO, RLOO, PRIME, and other variants
([official repository](https://github.com/volcengine/verl)); and OpenRLHF exposes
GRPO, Dr.GRPO, RLOO, PPO, and REINFORCE-family estimators in a shared framework
([official repository](https://github.com/OpenRLHF/OpenRLHF)). A contribution
must therefore specify the precise estimator and execution semantics, not merely
say "we use GRPO."

### 2. Objective corrections already occupy the obvious axes

- **Dr.GRPO.** *Understanding R1-Zero-Like Training* identifies response-length
  and question-difficulty biases induced by GRPO's normalizations and removes
  the relevant denominators. Its broader message is also competitive with an
  audit paper: template and base-model choices can masquerade as RL gains
  ([Liu et al., 2025](https://arxiv.org/abs/2503.20783); official
  [code](https://github.com/sail-sg/understand-r1-zero)).
- **DAPO.** DAPO combines decoupled asymmetric clipping, dynamic sampling that
  removes accuracy-0/1 groups, token-level policy-gradient loss, and overlong
  reward shaping. It reports a fully open 32B training system and 50 on AIME
  2024 ([Yu et al., 2025](https://arxiv.org/abs/2503.14476); official
  [repository](https://github.com/BytedTsinghua-SIA/DAPO)). Thus, "filter flat
  groups" is prior art, not a new controller.
- **GSPO.** GSPO moves importance ratios, clipping, rewards, and optimization to
  the sequence level and reports stability and efficiency benefits, especially
  for MoE RL ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)). A
  gradient audit must therefore test sequence-level and token-level semantics,
  not treat every group-relative objective as interchangeable.
- **AD-GRPO is not an adaptive-difficulty method in the relevant primary
  source.** In BNPO, `AD-GRPO` means *GRPO with advantage decomposition* for
  multi-component rewards. The reported average gain over GRPO is modest; BNPO
  itself uses an adaptive Beta normalization and supplies a variance-reduction
  argument ([Xiao et al., 2025](https://arxiv.org/abs/2506.02864)). This acronym
  must be defined in the flagship to avoid confusing it with difficulty-adaptive
  sampling.
- **AERO.** AERO directly targets zero-advantage dead zones with adaptive
  rollouts, selective rejection, and a Bayesian posterior. It reports roughly
  48% less training compute and 45% less step time while matching or improving
  Pass@8/Avg@8 across three Qwen configurations
  ([Zhang et al., 2026, preprint](https://arxiv.org/abs/2602.14338)). This is the
  closest direct competitor to a ZVF controller.

### 3. Zero-variance/group degeneracy is now an established problem statement

Several independent lines make the standalone ZVF diagnosis non-novel:

- DAPO's dynamic sampling removes groups whose sampled accuracy is exactly zero
  or one ([paper](https://arxiv.org/abs/2503.14476)).
- AERO explicitly states that all-correct and all-wrong groups yield zero
  group-normalized advantage and waste compute
  ([paper](https://arxiv.org/abs/2602.14338)).
- *Goldilocks RL* adapts training data toward the student's current difficulty
  frontier and reports gains at equal compute
  ([Mahrooghi et al., 2026, preprint](https://arxiv.org/abs/2602.14868)).
- *Spectral Policy Optimization* specifically targets all-negative groups by
  adding AI feedback that differentiates incorrect responses
  ([Chen et al., 2025, preprint](https://arxiv.org/abs/2505.11595)). This is an
  important counterexample to the idea that all-wrong groups should only receive
  more samples; they can also be repaired by changing the reward information.
- *Selective Rollout* identifies terminally uniform groups as wasted compute and
  stops likely-degenerate agentic groups from partial trajectories
  ([Zhai and Wang, 2026, preprint](https://arxiv.org/abs/2605.05802)).
- *GRPO, Dr.GRPO, and DAPO Are Three Operations on One Number* proves a
  group-standard-deviation identity, connects the methods through the same
  statistic, and claims implications for difficulty weighting and group size
  ([Bay and Yearick, 2026, preprint](https://arxiv.org/abs/2607.00152)). This is
  the most immediate collision with exact-ZVF theory.
- *Demystifying GRPO* represents the group-relative gradient as a U-statistic,
  derives finite-sample MSE and suboptimality results, and gives a group-size
  scaling law ([Zhou et al., 2026, preprint](https://arxiv.org/abs/2603.01162)).
  A new group-size theorem will be compared against this theory.

The residual opportunity is therefore not detecting unanimity. It is predicting
unanimity *before paying for all rollouts*, calibrating that prediction outside
the data used to fit it, separating all-wrong from all-correct actions, extending
beyond Bernoulli rewards, or proving that an intervention changes held-out
learning at a matched compute budget.

### 4. Adaptive rollout and curriculum work is highly competitive

The relevant methods occupy distinct but overlapping control surfaces:

| Control surface | Primary prior art | What it already claims |
|---|---|---|
| Post-rollout group filtering | [DAPO](https://arxiv.org/abs/2503.14476) | Remove accuracy-0/1 groups before the update |
| Pre-rollout prompt skipping | [GRESO](https://arxiv.org/abs/2506.02177) | Use temporal reward consistency to skip uninformative prompts; up to 2.0x total-training speedup without accuracy loss |
| Prompt curriculum | [SPEED-RL](https://arxiv.org/abs/2506.09016) | Select intermediate-difficulty prompts using a signal-to-noise argument; 2x--6x faster training |
| Competence/difficulty alignment | [CDAS](https://arxiv.org/abs/2505.17652) | Estimate stable historical difficulty and align it to current competence; faster than DAPO dynamic sampling |
| Difficulty prediction plus replay | [Sun et al.](https://arxiv.org/abs/2506.05316) | Attention-based adaptive difficulty and rollout replay; 25%--65% less time to target |
| Rollout subset selection | [PODS](https://arxiv.org/abs/2504.13818) | Generate broadly but update on a maximal-reward-variance subset |
| Budget allocation over prompts | [VIP, ICLR 2026](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf) | Predict per-prompt success with a GP and solve a hard-budget variance-minimization allocation |
| Adaptive group/rejection/posterior | [AERO](https://arxiv.org/abs/2602.14338) | Adapt rollout count, reject selectively, and use a Bayesian posterior to avoid dead zones |
| Teacher-generated difficulty | [Goldilocks RL](https://arxiv.org/abs/2602.14868) | Continuously match generated task difficulty to the student's evolving competence |
| Clip plus temperature control | [AGPO](https://arxiv.org/abs/2605.20722) | Use reward and entropy statistics to adapt clipping and decoding temperature at equal generated-token budget |
| Mid-trajectory termination | [Selective Rollout](https://arxiv.org/abs/2605.05802) | Stop likely-degenerate multi-turn groups before terminal reward |

This table is the baseline set for any controller paper. Beating only static G8
and G16 is no longer enough.

### 5. Variance reduction is broader than ZVF

RLOO showed that simple REINFORCE-style, leave-one-out baselines can outperform
more elaborate PPO-style RLHF at lower cost
([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)). BNPO proposes a
policy-adaptive Beta normalization and an explicit variance-reduction analysis
([Xiao et al., 2025](https://arxiv.org/abs/2506.02864)). A 2026 theory paper
argues that the group-relative estimator is a second-order U-statistic with an
oracle relationship ([Zhou et al.](https://arxiv.org/abs/2603.01162)), while
*Your Group-Relative Advantage Is Biased* argues that group-relative advantages
underweight hard prompts and proposes history-aware difficulty weighting
([Yang et al., 2026, preprint](https://arxiv.org/abs/2601.08521)). These claims
are not automatically consistent because they use different estimators,
assumptions, and targets. A flagship should exploit this tension with exact
fixtures rather than cite "variance reduction" generically.

Practical objective semantics are also unsettled. *Group-Relative REINFORCE Is
Secretly an Off-Policy Algorithm* reports that clipping can matter more than
importance weighting in several settings and frames common algorithms as
regularized REINFORCE variants
([ICLR 2026 paper](https://arxiv.org/abs/2509.24203)). *Clipping-Free Policy
Optimization* reports materially different off-policy behavior in TRL and verl
and runs 80 models across the frameworks
([paper PDF](https://gglab-ku.github.io/assets/pdf/2601.22801v1.pdf)). These are
direct competitors to a broad "objective audit" but also evidence that exact
cross-framework conformance testing remains useful.

### 6. Outcome-only RL is powerful, but its information limit matters

DeepSeek-R1 and ProRL support the claim that verifiable outcome rewards can
produce or expose strong reasoning behavior
([DeepSeek-R1](https://arxiv.org/abs/2501.12948);
[ProRL](https://arxiv.org/abs/2505.24864)). At the same time, outcome-only
groups cannot distinguish two equally wrong trajectories. PRIME addresses the
credit-assignment and efficiency limit by learning implicit process rewards
online from policy rollouts and outcome labels
([Cui et al., 2025](https://arxiv.org/abs/2502.01456)), while Spectral Policy
Optimization adds AI feedback within all-negative groups
([Chen et al., 2025](https://arxiv.org/abs/2505.11595)).

This matters for the flagship's causal language. ZVF measures whether an
outcome-only group contains within-group outcome contrast; it does not measure
whether the sampled reasoning is diverse, whether token-level credit is
correct, or whether a richer verifier could recover signal. Claims should say
"outcome-contrast starvation under the specified reward" rather than
"no learning signal exists."

## Candidate flagship paths

The competitive half-life below is an estimate of how long the present novelty
claim is likely to remain distinct if no additional evidence is produced. It is
not a prediction of paper-review duration.

### Path 1 — Exact ZVF theory only

**Candidate claim.** For Bernoulli rewards, derive the exact probability of a
zero-variance group, its complement as useful contrast probability, finite-G
bounds, and a group-size recommendation.

**Closest prior art.** The closest collisions are the group-standard-deviation
identity paper ([Bay and Yearick, 2026](https://arxiv.org/abs/2607.00152)),
Goldilocks RL's sparse-reward/difficulty analysis
([Mahrooghi et al., 2026](https://arxiv.org/abs/2602.14868)), VIP's
success-probability-to-gradient-variance analysis
([ICLR 2026](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf)),
and the U-statistic/group-size analysis
([Zhou et al., 2026](https://arxiv.org/abs/2603.01162)). DAPO and AERO already
operationalize the zero-variance event.

**Claimed novelty that is still safe.** A rigorously stated *calibration theorem*
for the specific repository metrics, exact confidence bounds under estimation of
`p`, decomposition into all-wrong and all-correct boundary events, and an
extension to graded/executable rewards could still be new. The raw Bernoulli
formula and "unanimous groups have zero centered advantage" are not.

**Remaining gap.** Existing work does not yet give a widely accepted,
distribution-shift-aware calibration result showing that a model fitted before
training predicts future useful-gradient availability across models, tasks,
stacks, reward cardinalities, and group sizes. This gap belongs more naturally
to Path 7 than to a pure theorem paper.

**Likely reviewer objections.** The core identity is elementary; the key event is
already used by DAPO/AERO; useful-gradient probability is not gradient quality;
i.i.d. Bernoulli rollouts ignore correlated decoding; and the proposed
`(1-ZVF)/sqrt(G)` objective is chosen rather than derived from end-to-end compute
or learning utility. Reviewers will also ask why a theory-only paper is needed
when VIP and the U-statistic paper derive stronger optimization consequences.

**Falsifiers.** Reject the path if (a) the flagship's theorems reduce to
`p^G+(1-p)^G` plus monotonicity; (b) the optimal-G result changes under a
reasonable compute cost; (c) empirical calibration fails on more than the
preregistered held-out tolerance; or (d) graded rewards do not admit a useful,
testable extension.

**Competitive half-life.** **Already expired** for a main-track standalone
claim; at most 0--2 months for a short technical note before further 2026 theory
absorbs it.

**Publication fit.** Workshop or appendix theorem in a larger empirical paper.
Not credible as a NeurIPS/ICLR/ICML main-track flagship in its current form.

**Verdict.** **Drop as standalone.** Keep the exact identity as notation and a
test oracle.

### Path 2 — One-formula asymmetric boundary controller

**Candidate claim.** Use one scalar decision score with opposite signs for the
two ZVF boundaries: extra compute may be valuable for a plausibly recoverable
all-wrong group, while an all-correct group should be retired rather than
resampled. The minimal controller changes only group size or prompt admission.

**Closest prior art.** DAPO's dynamic sampling is symmetric at accuracy 0/1
([paper](https://arxiv.org/abs/2503.14476)); GRESO skips historically
uninformative prompts ([paper](https://arxiv.org/abs/2506.02177)); SPEED-RL and
CDAS select a competence-aligned middle
([SPEED-RL](https://arxiv.org/abs/2506.09016);
[CDAS](https://arxiv.org/abs/2505.17652)); AERO already combines adaptive
rollouts, rejection, and a posterior
([paper](https://arxiv.org/abs/2602.14338)); and VIP allocates rollouts from
predicted success probabilities under a hard budget
([ICLR 2026](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf)).

**Claimed novelty that is still safe.** A *minimal, explicitly asymmetric,
pre-fitted* rule whose action difference is prospectively tested can be distinct:
escalate only the lower boundary when posterior rescue value exceeds rollout
cost; retire the upper boundary; leave ambiguous/mid-difficulty prompts at the
base G. The contribution would be causal parsimony and boundary decomposition,
not adaptive sampling in general.

**Remaining gap.** Most published curricula optimize symmetric informativeness
or intermediate difficulty. A clean experiment that isolates lower-boundary
rescue from upper-boundary retirement, while matching generated tokens and
measured FLOPs, remains valuable. The repository's 92.3% all-correct fire audit
is an unusually strong motivation for exactly this contrast, but it is not yet
an outcome.

**Likely reviewer objections.** AERO may already implement a substantively
equivalent categorization; the formula may just be a hand-written Bayes decision
rule; all-wrong prompts can be impossible rather than recoverable; all-correct
prompts can still benefit diversity/calibration; and one formula may overfit
binary math rewards. Reviewers will demand AERO, VIP, DAPO, SPEED/CDAS, static
G8/G16, symmetric-ZVF, and failure-only baselines—not only static G.

**Falsifiers.** Reject the controller claim if it makes the same action as the
best naive heuristic on at least 95% of eligible steps; fails to improve paired
learning-curve AUC over the best baseline; loses final-quality non-inferiority;
does not reduce measured FLOPs/tokens-to-target; or changes sign on the second
stack. Also reject "asymmetry matters" if lower-only and full-boundary rules are
indistinguishable within the preregistered margin.

**Competitive half-life.** **2--4 months.** Direct pressure from AERO, VIP,
AGPO, Goldilocks, and likely follow-ups is intense.

**Publication fit.** Main-track ML only with a decisive, multi-regime,
cross-stack matched-compute win and a transparent one-line rule. Otherwise a
Findings/workshop efficiency paper.

**Verdict.** **Run as the smallest controller test.** Do not enlarge the method
until this rule survives strong baselines.

### Path 3 — Maximal general compute controller

**Candidate claim.** A single controller jointly chooses prompt, group size,
sampling temperature, clipping radius, rollout continuation/termination,
replay, and retirement from a general telemetry state, optimizing learning
utility per token/FLOP across binary, graded, code, and agentic tasks.

**Closest prior art.** AGPO already couples adaptive clipping and temperature
([paper](https://arxiv.org/abs/2605.20722)); AERO controls rollouts and rejection
([paper](https://arxiv.org/abs/2602.14338)); VIP solves budgeted rollout
allocation ([ICLR 2026](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf));
PODS selects update subsets ([paper](https://arxiv.org/abs/2504.13818));
Sun et al. combine difficulty targeting and replay
([paper](https://arxiv.org/abs/2506.05316)); and Selective Rollout handles
mid-trajectory termination ([paper](https://arxiv.org/abs/2605.05802)). PRIME
changes the reward-information channel itself
([paper](https://arxiv.org/abs/2502.01456)).

**Claimed novelty that is still safe.** A formally specified constrained-control
problem with one state representation, an auditable action space, and a Pareto
frontier over quality, generated tokens, optimizer FLOPs, wall time, and memory
could be novel. Merely composing the published mechanisms is not.

**Remaining gap.** The literature lacks a standard, cross-stack controller
benchmark that compares all control surfaces under common accounting. This is
more naturally a benchmark contribution than a claim that one maximal policy is
universally best.

**Likely reviewer objections.** The method is an unidentifiable bundle; every
ablation is underpowered; reward regimes need different state/action semantics;
wall time is hardware-specific; controller inference and probe costs are omitted;
and the generality claim is unsupported by three tasks. Reviewers will ask why a
large learned controller is preferable to the simple asymmetric rule or VIP's
convex allocator.

**Falsifiers.** Reject the path if no component has a stable marginal benefit;
the controller's benefit disappears after charging probe/control overhead; a
static or one-formula baseline lies on the same quality-compute frontier; actions
fail to transfer across tasks/stacks; or the controller requires post-hoc
threshold changes.

**Competitive half-life.** **1--3 months** for novelty by feature aggregation;
6--12 months only if reframed as an open control benchmark with durable APIs.

**Publication fit.** ML systems or benchmark track with an extensive artifact;
main-track algorithms only with a formal control objective and unusually broad
evidence.

**Verdict.** **Do not lead with this path.** It is too broad for the current
evidence and maximizes competitive exposure.

### Path 4 — Causal objective/gradient audit

**Candidate claim.** Algorithm labels are insufficient causal variables. Exact
float64 fixtures and stack-differential tests can determine whether TRL and verl
implement the same mathematical treatment, whether a treatment changes only its
declared lever, and which loss/gradient difference causes an observed learning
delta.

**Closest prior art.** Dr.GRPO is an objective-bias audit
([Liu et al., 2025](https://arxiv.org/abs/2503.20783)); GSPO challenges
token-level ratio semantics ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071));
the ICLR 2026 REINFORCE analysis decomposes clipping, importance sampling,
regularization, and data weighting
([paper](https://arxiv.org/abs/2509.24203)); and CFPO directly documents
different off-policy semantics in TRL and verl while running 80 models
([paper PDF](https://gglab-ku.github.io/assets/pdf/2601.22801v1.pdf)). Official
frameworks expose overlapping algorithm names but different execution models
([TRL](https://github.com/huggingface/trl),
[verl](https://github.com/volcengine/verl),
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)).

**Claimed novelty that is still safe.** An executable *objective conformance
standard* is distinct from another analytic paper: canonical synthetic
minibatches; exact masks, ratios, selected groups, reductions, and flattened
gradients; treatment-specific metamorphic tests; negative controls; stack
fingerprints; and a causal chain from formula to code to gradient to outcome.
The strongest novelty would be demonstrating that apparently identical labels
produce materially different gradients and that fixing those differences changes
a survival verdict.

**Remaining gap.** Existing papers analyze objectives or compare frameworks, but
there is no widely adopted fail-closed conformance suite that certifies objective
identity before expensive RLVR runs. The repository's unwired Dr.GRPO flag is a
concrete failure mode that output telemetry did not catch. This path directly
addresses a real technology gap.

**Likely reviewer objections.** "This is unit testing, not research"; synthetic
float64 equality may not predict bf16 distributed behavior; implementation
differences can be intentional; only two frameworks are tested; and discovered
bugs may be version-specific. The answer must be a formal taxonomy, a public
fixture specification, nontrivial empirical failures, and an outcome-level causal
validation—not only a test harness.

**Falsifiers.** Reject the paper claim if both stacks already agree on all
targeted semantics; injected fixture differences do not predict training
differences; no real framework defect or undocumented semantic divergence is
found; or the causal intervention fails to alter the expected gradient/outcome.
The S1 tolerance (`rtol<=1e-6`, `atol<=1e-8`) is itself a preregistered falsifier.

**Competitive half-life.** **6--12 months.** Objective papers are moving fast,
but conformance infrastructure is more durable than a new optimizer heuristic.

**Publication fit.** TMLR, MLSys, NeurIPS Datasets & Benchmarks, or a strong
reproducibility/artifact venue. Main-track ML is plausible if the audit reveals a
general causal phenomenon rather than isolated bugs.

**Verdict.** **Preferred near-term paper core.** Complete S1 before spending on
controller screening.

### Path 5 — Unified theory + intervention + controller

**Candidate claim.** A single paper moves from a calibrated theory of variance
starvation, through a causal intervention that restores useful gradients, to a
boundary-aware compute controller that improves held-out learning utility at
matched compute across stacks and reward regimes.

**Closest prior art.** The theory overlaps the group-standard-deviation identity
and U-statistic papers
([Bay and Yearick](https://arxiv.org/abs/2607.00152);
[Zhou et al.](https://arxiv.org/abs/2603.01162)); the intervention overlaps
Spectral Policy Optimization and PRIME
([SPO](https://arxiv.org/abs/2505.11595);
[PRIME](https://arxiv.org/abs/2502.01456)); and the controller overlaps AERO,
VIP, Goldilocks, GRESO, SPEED-RL, and AGPO
([AERO](https://arxiv.org/abs/2602.14338);
[VIP](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf);
[Goldilocks](https://arxiv.org/abs/2602.14868);
[GRESO](https://arxiv.org/abs/2506.02177);
[SPEED-RL](https://arxiv.org/abs/2506.09016);
[AGPO](https://arxiv.org/abs/2605.20722)).

**Claimed novelty that is still safe.** The *validated chain* can be novel even
when each component is not: a frozen pre-fit model predicts an unseen cell; a
targeted intervention changes the predicted gradient mechanism; and the minimal
controller converts that mechanism into matched-compute learning gains. Cross-
stack sign preservation and binary-to-graded reward generalization would make the
chain substantially stronger than existing one-task controllers.

**Remaining gap.** Few papers close all three levels—probabilistic prediction,
objective/gradient causality, and end-to-end compute utility—under one
preregistered protocol. This is the flagship's best high-upside gap.

**Likely reviewer objections.** The paper is three papers stitched together;
the theory is elementary; the intervention/controller is not novel relative to
AERO/VIP; the experiment matrix is too large to power; and negative components
make the story incoherent. A clear ordered gate and one primary causal diagram
are essential. The paper must remain publishable if H2/H3 fail: the mechanism
claim and negative controller result need their own clean estimand.

**Falsifiers.** The preregistration already states them: H1 calibration misses
more than 20% of cells or pooled absolute error exceeds 0.05; H2 has no positive
paired AUC effect/non-inferiority or no FLOP saving; H3 does not beat the best
naive heuristic or acts identically at least 95% of the time; H4 changes sign or
fails non-inferiority on the second stack. Any of these must remove the
corresponding headline.

**Competitive half-life.** **3--6 months.** The integrated causal chain has more
protection than the controller alone, but every component faces active 2026
competition.

**Publication fit.** NeurIPS/ICLR/ICML main track only if all gates pass with
strong multi-regime evidence. If controller gates fail, split to a TMLR/Findings
mechanism-and-negative-result paper rather than forcing the unified story.

**Verdict.** **Keep as the earned flagship route, not the default promise.**

### Path 6 — Algorithm audit benchmark

**Candidate claim.** A method's published gain should be separated into an
algorithmic delta and a stack delta. A common-stack, fail-closed benchmark with
paired seeds and equivalence-aware verdicts measures whether GRPO-family gains
retain, survive, disappear, reverse, or remain inconclusive.

**Closest prior art.** DAPO supplies a reproducible end-to-end system
([paper](https://arxiv.org/abs/2503.14476)); CFPO compares objectives across TRL
and verl ([paper PDF](https://gglab-ku.github.io/assets/pdf/2601.22801v1.pdf));
OpenRLHF and verl provide shared algorithm surfaces
([OpenRLHF](https://github.com/OpenRLHF/OpenRLHF);
[verl](https://github.com/volcengine/verl)); and a 2025 comparative preprint
reports that DAPO dynamic sampling can fail to help under its setup
([Lian, 2025](https://arxiv.org/abs/2512.07611)). The repository's R08/E1 is
already the direct prototype and currently has stronger provenance than most
small comparisons.

**Claimed novelty that is still safe.** The survival-verdict framework,
strict machine contract, full local/W&B/HF reconciliation, treatment
fingerprints, held-out completion hashes, and equivalence-aware fail-closed
aggregation form a distinctive benchmark methodology. The E1 finding—four
arms `INCONCLUSIVE` after the 2026-08-02 exact-t reanalysis superseded the
DAPO `DISAPPEARS` verdict, with no arm declared a failure—is also a
disciplined negative result.

**Remaining gap.** To be an archival benchmark rather than a bounded case
study, it needs an open primary stack, objective-differential tests, longer
learning curves, multiple tasks/reward regimes, and at least one second stack.
It should compare two estimands: faithful published-recipe reproduction and
common-stack algorithm isolation. Otherwise reviewers can argue that a method
was amputated from the stack it requires.

**Likely reviewer objections.** Thirty steps on Qwen3-8B/GSM8K cannot test the
published claims of large-scale methods; Tinker is closed; LoRA can change
method behavior; the published deltas are not metric-compatible; eight seeds
still leave three arms underpowered; and common-stack reimplementation may be
unfaithful. The benchmark also needs governance for fast-moving variants and
versioned stack semantics.

**Falsifiers.** Reject the archival claim if an open-stack replication reverses
the DAPO equivalence result; objective fixtures show the treatment was
misimplemented; published-recipe reproduction fails for non-algorithmic reasons;
longer curves diverge after step 30; or cross-task heterogeneity makes a pooled
survival label meaningless. Inconclusive arms must remain inconclusive until
powered; non-significance is not disappearance.

**Competitive half-life.** **9--18 months** if the artifact is extensible and
versioned; 2--4 months if it remains only the current five-arm snapshot.

**Publication fit.** NeurIPS Datasets & Benchmarks, MLSys, TMLR, ReScience C, or
a reproducibility track. This is the best fit for the evidence already in hand.

**Verdict.** **Strongest evidence-backed route.** Pair it with Path 4 so the
benchmark can certify its own objective semantics.

### Path 7 — Predictive sentinel and calibration benchmark

**Candidate claim.** Before or early in training, a frozen model of success
probability, group size, uncertainty, and reward regime predicts future
zero-variance rate, useful-gradient availability, and learning-curve failure on
unseen task/model/G cells. The contribution is calibrated prediction and
selective abstention, not a controller win.

**Closest prior art.** VIP uses a Gaussian process to predict per-prompt success
and convert it to gradient-variance allocations
([ICLR 2026](https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf));
GRESO uses historical temporal consistency
([paper](https://arxiv.org/abs/2506.02177)); CDAS estimates difficulty from
historical discrepancies ([paper](https://arxiv.org/abs/2505.17652)); and
Goldilocks learns a teacher of current difficulty
([paper](https://arxiv.org/abs/2602.14868)). AERO's Bayesian posterior is also a
direct baseline ([paper](https://arxiv.org/abs/2602.14338)).

**Claimed novelty that is still safe.** A strict held-out *calibration* study
across group sizes, models, stacks, and binary versus graded rewards, with
pre-fit parameters and uncertainty intervals, is more diagnostic than the
controller-oriented prior work. The graded MBPP regime and direct
gradient-cosine telemetry are especially important: they can establish when
ZVF is a sufficient proxy and when it is not.

**Remaining gap.** Current methods mostly optimize performance; few make the
prediction itself the primary estimand with coverage, calibration error,
abstention, temporal lead, and out-of-domain failure boundaries. This is a
publishable negative result even if control does not help.

**Likely reviewer objections.** Predicting ZVF from rollouts used to compute ZVF
is tautological; success probabilities drift after every update; prompt-level
rows are not independent; a calibrated event rate does not imply gradient
direction or learning; and a GP/posterior baseline may dominate the simple
formula. The study must predict future/unseen cells, not reconstruct current
groups, and seed must remain the training replicate.

**Falsifiers.** Use H1 exactly: more than 20% held-out interval misses or pooled
absolute calibration error above 0.05. Also reject a useful-gradient claim if
predicted ZVF has weak or unstable association with gradient norm/cosine after
controlling for reward scale, clipping, and length, or if performance is no
better than a historical-frequency/GP baseline.

**Competitive half-life.** **4--8 months.** Prediction is crowded but less
commoditized than control; graded-reward and cross-stack calibration improve
durability.

**Publication fit.** TMLR, ACL/EMNLP Findings, or a NeurIPS/ICLR main-track
analysis paper if the cross-regime failure boundary is strong.

**Verdict.** **Best controller-independent fallback.** It should be produced by
S2 even if the expansion gate fails.

## Comparative scorecard

Scores are 1 (weak) to 5 (strong) and reflect the present repository state, not
the hypothetical fully completed study.

| Path | Novelty survival | Evidence already held | Additional compute burden | Reviewer defensibility | Recommended priority |
|---|---:|---:|---:|---:|---:|
| 1. Exact ZVF theory only | 1 | 4 | 1 | 1 | 7 |
| 2. One-formula asymmetric controller | 3 | 2 | 3 | 3 | 3 |
| 3. Maximal general compute controller | 2 | 1 | 5 | 2 | 6 |
| 4. Causal objective/gradient audit | 4 | 3 | 2 | 4 | 1 |
| 5. Unified theory + intervention + controller | 4 if all gates pass, 2 otherwise | 2 | 5 | 4 if preregistered | 4 |
| 6. Algorithm audit benchmark | 4 | 5 | 3 | 4 | 2 |
| 7. Predictive sentinel/calibration | 3 | 3 | 3 | 4 | 3 |

The scorecard implies a staged paper strategy:

1. Finish the causal objective/gradient gate and make its conformance artifact
   independently publishable.
2. Upgrade E1 into an open, longer-horizon audit benchmark; preserve
   `INCONCLUSIVE` rather than manufacturing winners.
3. In screening, test the minimal asymmetric rule before a maximal controller.
4. Regardless of controller success, publish the preregistered calibration and
   failure-boundary analysis.
5. Promote the unified flagship only if every causal link survives.

## Reviewer-proof novelty boundary

The paper should explicitly separate claims that the current literature has
already absorbed from claims the program can still own.

| Unsafe headline | Defensible replacement |
|---|---|
| "We discover that unanimous GRPO groups have zero gradient." | "We prospectively calibrate when outcome-contrast starvation occurs on unseen cells and test whether it predicts measured gradient utility." |
| "We derive the probability of a zero-variance binary group." | "We use the exact Bernoulli identity as a test oracle, then quantify correlation, drift, graded-reward failure, and uncertainty." |
| "We introduce adaptive group sizing." | "We isolate a one-line asymmetric boundary action against AERO, VIP, DAPO, curricula, and static-G controls at equal tokens/FLOPs." |
| "Our controller is compute efficient." | "It reaches a frozen quality target at lower measured end-to-end FLOPs, including probes, rejected rollouts, retries, and controller overhead." |
| "DAPO/GSPO/Dr.GRPO/AERO do not work." | "Under one frozen stack and short-horizon estimand, DAPO's delta is equivalent to zero; three other deltas remain underpowered. Open-stack external validity is pending." |
| "The algorithm caused the gain." | "Float64 objective fixtures certify the intended semantic delta; paired common-stack intervention identifies the outcome change attributable to that delta." |

## Baseline and ablation requirements by route

### For any controller claim

At minimum include:

- static G8 and static G16;
- DAPO-style symmetric zero-variance dynamic sampling;
- failure-only retry;
- full boundary-aware rule;
- the minimal one-formula asymmetric rule;
- AERO or a faithful AERO-equivalent implementation;
- VIP-style predicted allocation or the strongest implementable hard-budget
  allocator;
- a prompt-curriculum baseline (SPEED-RL, CDAS, or GRESO class);
- a posterior/difficulty-only baseline without ZVF nomenclature;
- equal generated-token ceilings and secondary measured-FLOP accounting;
- ablations charging every probe, rejected rollout, environment execution, and
  controller decision.

Static G16 is necessary but no longer sufficient as the sole strong baseline.

### For the objective/gradient audit

At minimum include:

- GRPO, mean-centered no-std/Dr.GRPO, DAPO sampling and clipping, GSPO
  sequence-level ratios, and AERO action paths;
- on-policy and controlled off-policy fixtures;
- positive, negative, all-correct, all-wrong, graded, padded, unequal-length,
  clipped, and masked fixtures;
- reference float64 losses and flattened gradients;
- metamorphic checks showing treatment fixtures differ and non-treatment fields
  remain invariant;
- TRL and verl adapters, with OpenRLHF as a high-value third implementation;
- bf16/distributed stress tests after float64 conformance;
- at least one intervention where correcting a semantic mismatch changes an
  end-to-end outcome or survival verdict.

### For the audit benchmark

At minimum include:

- a faithful published-recipe stratum and a common-stack isolation stratum;
- Qwen3-1.7B screening and Qwen3-8B confirmation or another preregistered scale
  pair;
- GSM8K, MATH-500, and MBPP reward regimes as frozen;
- longer learning curves than E1's 30 steps;
- paired seeds, equivalence tests, MDE disclosure, and hierarchical task/model
  summaries;
- exact environment, tokenizer, parser, objective, sampler, and checkpoint
  fingerprints;
- public/open execution on at least one stack and directional replication on a
  second;
- an explicit policy for inconclusive verdicts and future variant versions.

## Technology gaps worth claiming

After the literature pass, these gaps remain real and testable:

1. **Objective conformance before compute.** Popular frameworks expose the same
   algorithm labels without a common executable semantic certificate.
2. **Causal formula-to-gradient-to-outcome attribution.** Papers typically stop
   at algebra or training curves; few demonstrate the whole intervention chain.
3. **Boundary asymmetry under matched compute.** The field heavily targets
   intermediate difficulty, but clean lower-rescue versus upper-retirement
   experiments remain scarce.
4. **Calibration across reward cardinality.** Most theory assumes Bernoulli
   rewards. The MBPP graded regime can expose where ZVF is too coarse.
5. **Honest power-aware survival auditing.** Published comparisons rarely use
   equivalence rules and explicit `INCONCLUSIVE` verdicts with achieved MDE.
6. **End-to-end compute accounting.** Rollout papers often report generation or
   wall-time savings; fewer charge probes, rejected samples, environment calls,
   optimizer work, and controller overhead in one estimand.
7. **Faithful-versus-isolated comparison.** The community lacks a standard way
   to report both "does the published recipe reproduce?" and "does the declared
   algorithmic delta survive on a shared stack?"

## Risks that cannot be solved by more citation

- **Correlated rollouts:** `p^G+(1-p)^G` assumes conditional i.i.d. Bernoulli
  outcomes. Shared decoding state, batched RNG behavior, or mode collapse can
  make the effective group correlation nonzero. The empirical calibration must
  test this rather than hide it in a theorem assumption.
- **Outcome-only ambiguity:** an all-wrong group can contain meaningful process
  diversity. ZVF says the outcome verifier cannot distinguish it; it does not say
  the trajectories contain no learnable information.
- **All-correct is not always mastered:** exact-answer success can coexist with
  verbosity, brittle reasoning, poor calibration, or low pass@k diversity. Upper
  retirement must be evaluated on the frozen primary endpoint and secondary
  reasoning-quality measures, not presumed harmless.
- **Group-size endogeneity:** changing G also changes prompt count, optimizer
  cadence, batch composition, wall time, and potentially off-policy lag. Equal
  rollout count alone is not causal isolation.
- **Framework semantic drift:** TRL, verl, and OpenRLHF evolve rapidly. Every
  benchmark result must bind versions and content hashes; a library name is not
  a stable treatment.
- **Short-horizon rank inversion:** E1 can show 30-step equivalence while methods
  diverge later. The flagship needs learning curves and tokens/FLOPs-to-target,
  not only final step 30.

## Primary-source bibliography

### Foundations and core GRPO-family objectives

1. Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical
   Reasoning in Open Language Models*. ArXiv preprint.
   <https://arxiv.org/abs/2402.03300>
2. DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
   via Reinforcement Learning*. ArXiv technical report.
   <https://arxiv.org/abs/2501.12948>
3. Liu et al. (2025), *Understanding R1-Zero-Like Training: A Critical
   Perspective* (Dr.GRPO). ArXiv/preprint and official code.
   <https://arxiv.org/abs/2503.20783>;
   <https://github.com/sail-sg/understand-r1-zero>
4. Yu et al. (2025), *DAPO: An Open-Source LLM Reinforcement Learning System at
   Scale*. ArXiv preprint and official repository.
   <https://arxiv.org/abs/2503.14476>;
   <https://github.com/BytedTsinghua-SIA/DAPO>
5. Zheng et al. (2025), *Group Sequence Policy Optimization*. ArXiv preprint.
   <https://arxiv.org/abs/2507.18071>
6. Xiao, Zhang, and Cao (2025), *BNPO: Beta Normalization Policy Optimization*
   (source of advantage-decomposed AD-GRPO). ArXiv preprint.
   <https://arxiv.org/abs/2506.02864>
7. Zhang et al. (2026), *Train Less, Learn More: Adaptive Efficient Rollout
   Optimization for Group-Based Reinforcement Learning* (AERO). ArXiv preprint.
   <https://arxiv.org/abs/2602.14338>

### Degeneracy, theory, and variance

8. Bay and Yearick (2026), *GRPO, Dr.GRPO, and DAPO Are Three Operations on One
   Number: The Group-Standard-Deviation Identity*. ArXiv preprint.
   <https://arxiv.org/abs/2607.00152>
9. Zhou et al. (2026), *Demystifying Group Relative Policy Optimization: Its
   Policy Gradient is a U-Statistic*. ArXiv preprint.
   <https://arxiv.org/abs/2603.01162>
10. Yang et al. (2026), *Your Group-Relative Advantage Is Biased*. ArXiv
    preprint. <https://arxiv.org/abs/2601.08521>
11. Ahmadian et al. (2024), *Back to Basics: Revisiting REINFORCE Style
    Optimization for Learning from Human Feedback in LLMs*. ArXiv preprint.
    <https://arxiv.org/abs/2402.14740>
12. *Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying
    Some Myths About GRPO and Its Friends*. ICLR 2026 paper.
    <https://arxiv.org/abs/2509.24203>
13. *Clipping-Free Policy Optimization for Large Language Models*. 2026 paper
    PDF. <https://gglab-ku.github.io/assets/pdf/2601.22801v1.pdf>

### Adaptive sampling, curricula, and compute allocation

14. Xu et al. (2025), *Not All Rollouts Are Useful: Down-Sampling Rollouts in
    LLM Reinforcement Learning* (PODS). ArXiv preprint.
    <https://arxiv.org/abs/2504.13818>
15. Zheng et al. (2025), *Act Only When It Pays: Efficient Reinforcement
    Learning for LLM Reasoning via Selective Rollouts* (GRESO). ArXiv preprint.
    <https://arxiv.org/abs/2506.02177>
16. Zhang et al. (2025), *SPEED-RL: Faster Training of Reasoning Models via
    Online Curriculum Learning*. ArXiv preprint.
    <https://arxiv.org/abs/2506.09016>
17. Kong et al. (2025), *Rethinking the Sampling Criteria in Reinforcement
    Learning for LLM Reasoning: A Competence-Difficulty Alignment Perspective*
    (CDAS). ArXiv preprint. <https://arxiv.org/abs/2505.17652>
18. Sun et al. (2025), *Improving Data Efficiency for LLM Reinforcement
    Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout
    Replay*. ArXiv preprint. <https://arxiv.org/abs/2506.05316>
19. Nguyen et al. (2026), *Adaptive Rollout Allocation for Online Reinforcement
    Learning with Verifiable Rewards* (VIP). Published at ICLR 2026.
    <https://zhaoyuzhi.github.io/files/2026-Adaptive-Rollout-Allocation-for-Online-Reinforcement-Learning-with-Verifiable-Rewards.pdf>
20. Mahrooghi, Lotfi, and Abbe (2026), *Goldilocks RL: Tuning Task Difficulty to
    Escape Sparse Rewards for Reasoning*. ArXiv preprint.
    <https://arxiv.org/abs/2602.14868>
21. Hu et al. (2026), *AGPO: Adaptive Group Policy Optimization with Dual
    Statistical Feedback*. ArXiv preprint.
    <https://arxiv.org/abs/2605.20722>
22. Zhai and Wang (2026), *Selective Rollout: Mid-Trajectory Termination for
    Multi-Sample Agent RL*. ArXiv preprint.
    <https://arxiv.org/abs/2605.05802>

### Outcome-only rewards and richer information

23. Cui et al. (2025), *Process Reinforcement through Implicit Rewards* (PRIME).
    ArXiv preprint. <https://arxiv.org/abs/2502.01456>
24. Chen et al. (2025), *Spectral Policy Optimization: Coloring your Incorrect
    Reasoning in GRPO*. ArXiv preprint.
    <https://arxiv.org/abs/2505.11595>
25. Liu et al. (2025), *ProRL: Prolonged Reinforcement Learning Expands
    Reasoning Boundaries in Large Language Models*. ArXiv preprint.
    <https://arxiv.org/abs/2505.24864>

### Framework and reproducibility surfaces

26. Hugging Face TRL, official repository. <https://github.com/huggingface/trl>
27. verl, official repository. <https://github.com/volcengine/verl>
28. OpenRLHF, official repository and 2024 technical report.
    <https://github.com/OpenRLHF/OpenRLHF>;
    <https://arxiv.org/abs/2405.11143>
29. Lian (2025), *Comparative Analysis and Parametric Tuning of PPO, GRPO, and
    DAPO for LLM Reasoning Enhancement*. ArXiv preprint.
    <https://arxiv.org/abs/2512.07611>

## Rerun inputs

```yaml
workflow: firecrawl-research-papers
topic: >
  2024-2026 GRPO-family objectives, outcome-only RLVR, zero-variance group
  degeneracy, adaptive rollout/group sizing, difficulty curricula, variance
  reduction, compute-aware RL, and objective/audit benchmarks
source_policy: primary papers and official repositories only for major claims
searched_on: 2026-07-20
candidate_paths:
  - exact ZVF theory only
  - one-formula asymmetric controller
  - maximal general compute controller
  - causal objective/gradient audit
  - unified theory plus intervention plus controller
  - algorithm audit benchmark
  - predictive sentinel and calibration benchmark
output: markdown landscape with novelty, gaps, objections, half-life, falsifiers,
  and publication fit
```

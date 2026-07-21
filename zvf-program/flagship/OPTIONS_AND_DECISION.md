# Flagship options and decision

**Decision date:** 2026-07-20  
**Evidence base:** completed E1 40/40 audit, literature review, eight-role
simulated user-fit exercise, synthetic policy simulation, adversarial review,
and the preregistered S1--S4 plan.  
**Status:** decision document, not a claim of a new result. S1--S4 have not run.

## Executive decision

The current evidence is not enough for a competitive NeurIPS, ICLR, or ICML
paper about a new adaptive rollout algorithm. The defensible next paper is:

> **Cross-stack causal conformance for group-relative policy optimization:**
> determine whether nominally identical GRPO-family objectives and adaptive
> policies produce the same loss, gradient, decisions, and training outcome in
> TRL and verl; then establish when outcome-only failure statistics are
> insufficient to choose the correct intervention.

The controller remains useful, but only as a gated consequence of this claim.
It should not be the paper's premise.

Recommended composition:

1. **Core:** objective/gradient/decision conformance across TRL and verl.
2. **Theory, if nontrivial:** an identifiability result showing that the same
   observed failure state can require different optimal interventions.
3. **Empirical consequence:** a minimal policy that distinguishes all-failure
   from all-success only after the required extra information is available.
4. **Fallback artifact:** the E1 audit methodology and evidence package.

This route exploits the project's strongest assets: exact provenance, an
existing implementation incident, 40 verified experiment units, explicit
negative-result labels, and a reproducible test framework. It also avoids
claiming novelty already occupied by work on zero-advantage groups, adaptive
rollout allocation, or difficulty-aware sampling.

## Why the headline must change

The relevant literature has advanced beyond the original framing:

- [Gradient Starvation in Binary-Reward GRPO](https://arxiv.org/abs/2605.07689)
  already names the zero-gradient problem, analyzes all-zero/all-one groups,
  and proposes a sign-based remedy.
- [AVSPO](https://arxiv.org/abs/2605.21125), accepted at ICML 2026, introduces
  an advantage-collapse metric and reports a mitigation that learns from
  homogeneous groups.
- [AERO](https://arxiv.org/abs/2602.14338) already combines adaptive rollouts,
  selective retention, and difficulty estimation for efficiency.
- DAPO, GRESO, VIP, HORA, and RL-ZVP further crowd the space of filtering,
  difficulty estimation, adaptive allocation, and learning from otherwise
  uninformative groups.

Therefore, these are not durable headline claims:

- homogeneous reward groups yield zero centered advantage;
- the probability of a homogeneous group is `p^G + (1-p)^G`;
- rollout allocation should depend on difficulty or group informativeness;
- all-failure and all-success should be handled differently.

The differentiated question is whether an observed statistic is sufficient to
select an intervention and whether major implementations realize that
intervention as specified.

## The seven candidate routes

### 1. Rebrand ZVF as the central phenomenon

**What:** center the paper on the rate of homogeneous reward groups and its
effect on learning.  
**Why it was attractive:** it is mathematically clean, measurable, and already
supported by the E1 instrumentation.  
**Why not:** the phenomenon and direct fixes now have explicit prior work.  
**Rescue:** keep the metric as a diagnostic and stratification variable, not a
contribution.  
**Decision:** reject as the flagship; retain in background and analysis.

### 2. Minimal asymmetric rollout controller

**What:** treat all-failure and all-success separately, spending extra samples
only where their expected value exceeds their cost.  
**Why it was attractive:** users want an actionable policy; it was the top
option in the simulated user-fit exercise.  
**Why not yet:** AERO/VIP/HORA/GRESO occupy closely related territory, and the
current controller's group-size path uses a shared homogeneous-group rate, so
the implementation is not evidence for the intended six-policy comparison.  
**Rescue:** compare a one-equation policy against strong external baselines
after objective and decision conformance pass.  
**Decision:** fourth priority; empirical consequence, not headline.

### 3. Full triage controller

**What:** combine retry, keep, drop/recheck, and dynamic group size in one
policy.  
**Why it was attractive:** potentially maximizes quality per unit compute.  
**Why not:** it is difficult to identify causally, expensive to ablate, and
maximally exposed to prior-art collisions.  
**Rescue:** none until simpler interventions succeed individually.  
**Decision:** reject for the first flagship submission.

### 4. Cross-stack causal conformance

**What:** test whether a canonical mathematical specification, TRL, and verl
produce identical losses, gradients, adaptive decisions, and short training
effects.  
**Why:** the project has direct evidence that an implementation can appear to
run while failing to realize the intended mechanism. The literature generally
compares algorithms, not an auditable chain from equation to implementation to
training effect.  
**Differentiation:** conformance is a method and empirical result, not another
GRPO variant.  
**Kill criterion:** if TRL and verl agree on all material cases and all
discrepancies are cosmetic or artificial, this is not a main-track paper.  
**Decision:** first priority and paper core.

### 5. Unified theory + conformance + controller

**What:** prove when the diagnostic is insufficient, verify implementations,
and demonstrate an efficient controller.  
**Why:** this is the only route with the scope of a strong flagship paper.  
**Risk:** it is three papers' worth of claims and requires every link to hold.  
**Rescue:** use a modular paper structure in which each stage has its own stop
condition and fallback venue.  
**Decision:** aspirational route only if Routes 4 and 7 both succeed.

### 6. Audited reproduction benchmark

**What:** publish the 40/40 E1 audit and the `SURVIVES` / `DISAPPEARS` /
`INCONCLUSIVE` protocol as a reproducibility artifact.  
**Why:** this is the strongest completed evidence and the lowest-risk asset.  
**Why not the first flagship:** one model, one principal task, short training,
and heterogeneous method budgets limit the breadth of the scientific claim.  
**Rescue:** add a second framework and a small faithful-versus-controlled
comparison.  
**Decision:** second-best artifact route; likely better aligned with a
reproducibility, datasets/benchmarks, or systems venue unless broadened.

### 7. Information insufficiency / identifiability theorem

**What:** construct observationally matched training states with the same
`(p_hat, G, homogeneous-group rate, history)` but different optimal actions;
identify the minimal extra observation that resolves the ambiguity.  
**Why:** it explains why outcome-only triage policies disagree and gives the
conformance study a scientific thesis.  
**Risk:** a generic no-free-lunch construction is trivial and unpublishable.  
**Kill criterion:** the theorem must hold under realistic restrictions and
predict a measurable action reversal; otherwise drop it.  
**Decision:** second scientific priority and conditional theory contribution.

## Comparative decision

| Route | Novelty durability | Existing evidence | New compute | Collision risk | Submission role |
|---|---:|---:|---:|---:|---|
| 4. Cross-stack conformance | High | Medium | Low | Low--medium | **Core** |
| 7. Identifiability theorem | High if nontrivial | Low | Low--medium | Medium | **Conditional theory** |
| 6. Audited benchmark | High | Very high | Medium | Medium | **Fallback artifact** |
| 2. Minimal controller | Medium | Low | Medium--high | Very high | **Gated consequence** |
| 5. Unified route | High only if complete | Low | Very high | High | **Flagship upside** |
| 1. ZVF phenomenon | Low | High | Low | Very high | Background only |
| 3. Full triage | Low--medium | Low | Very high | Very high | Stop |

The user-fit exercise and policy simulation support the modular choice but do
not constitute evidence. Eight simulated stakeholder roles ranked an actionable
minimal policy first and conformance second. In the synthetic simulation, no
single policy dominated: fixed group size, minimal asymmetry, and full triage
each won some regimes. That is exactly why the paper must first establish what
information distinguishes the regimes.

## The proposed paper

Working title:

> **Same Signal, Different Action: Causal Conformance for Group-Relative RL**

Primary claim to test:

> Nominally identical GRPO-family algorithms and adaptive policies can differ
> materially in loss, gradient, and decisions across implementations; semantic
> conformance tests detect these differences and prevent false algorithmic
> conclusions.

Conditional theory claim:

> Outcome-only homogeneous-group statistics identify loss of within-group
> learning signal but are not generally sufficient to select whether to retry,
> keep, recheck, or alter the learning signal.

Conditional controller claim:

> A controller using the additional identified information improves learning
> per unit total cost, or reaches matched quality at lower cost, across at least
> two models and three task regimes.

The paper should claim only the first statement unless the corresponding later
gates pass.

## Execution plan

### Gate 0 — preserve and freeze the completed evidence

- Do not modify the 40 E1 unit artifacts.
- Snapshot the current repository, environment, W&B identifiers, Hugging Face
  commits, and decision document.
- Re-run the existing audit verifier only; do not regenerate training data.

**Exit:** E1 still verifies 40/40 and all hashes match.

### Gate 1 — write the scientific specification before training

Specify canonical objectives and policy decisions for:

- all-correct, all-incorrect, mixed, and tied rewards;
- equal variance with different reward distributions;
- clipping boundaries and zero denominators;
- noisy, missing, and delayed rewards;
- per-token versus per-sequence normalization;
- retry, keep, recheck, and stop decisions;
- every policy listed in the preregistration.

Add property tests for invariance, permutation, translation/scaling where
appropriate, and finite-gradient behavior.

**Kill:** no precise distinction can be written between the candidate policies.

### Gate 2 — conformance on TRL and verl

For each case, compare:

1. canonical reference output;
2. TRL output;
3. verl output;
4. analytical or finite-difference gradient;
5. adaptive decision and total charged cost.

Require predefined tolerances and version pins. Investigate discrepancies,
classify them as specification, implementation, default, or numerical effects,
and produce a minimal reproduction.

**Main-track go:** at least one material, real-world discrepancy changes a
gradient or policy decision and survives maintainers' strongest explanation.

**Fallback:** if stacks agree, release the suite as infrastructure and stop the
conformance-paper claim.

### Gate 3 — nontrivial identifiability result

Prove or disprove sufficiency of the observable state under explicit,
realistic assumptions. The useful result has this form:

- two latent regimes yield the same observable state;
- retry is optimal in one and keep/recheck is optimal in the other;
- any policy using only that state incurs positive regret;
- one named extra observation resolves or bounds the ambiguity;
- the construction maps to executable task regimes.

**Kill:** the result needs arbitrary rewards, adversarially chosen latent states,
or an observation unavailable in practice.

### Gate 4 — small empirical action-reversal pilot

Do not launch the existing 189-run S2--S4 plan immediately. Replace its first
compute gate with a new preregistration after review:

- one 1.7B model;
- two carefully matched task regimes;
- four policies: fixed `G=8`, fixed `G=16`, minimal asymmetric controller, and
  one strong external adaptive baseline;
- three seeds;
- short but power-justified horizon;
- charged total tokens/FLOPs, including probes and rejected samples.

This is **24 units** for two fixed controls plus two adaptive policies across
two regimes and three seeds. If only one fixed control is needed after a paired
power calculation, it can be reduced to 18 units. Do not reduce seeds.

**Go:** preregistered action-by-regime interaction, effect-direction reversal,
and a material gradient or learning-efficiency difference.

**Kill:** homogeneous-group rate alone performs as well as the proposed added
information, or the strong external baseline dominates.

### Gate 5 — flagship confirmation

Only after Gates 2--4 succeed:

- two model scales or families;
- three task/reward regimes;
- at least five confirmation seeds;
- faithful implementations of the strongest relevant baselines;
- learning curves, matched-quality compute, quality at matched compute, and
  calibrated uncertainty;
- held-out predictions of when each policy should win;
- ablations of every additional observation and controller branch.

At this point, revise the existing S2--S4 preregistration instead of silently
reusing it. Preserve the old plan and checksum as historical records.

## Submission decision tree

- **Conformance discrepancy + nontrivial theory + confirmed controller:**
  NeurIPS/ICLR/ICML flagship.
- **Conformance discrepancy, controller fails:** conformance paper; submit to a
  main ML or strong systems/ML venue according to empirical breadth.
- **Theory predicts action reversals, stacks agree:** theory/empirical paper;
  conformance suite becomes supporting infrastructure.
- **No material conformance discrepancy, no nontrivial theory:** do not spend on
  the full controller matrix; publish the E1 audit through a reproducibility or
  systems route.
- **Only benchmark breadth improves:** datasets/benchmarks, reproducibility, or
  systems route; do not market it as a new optimization algorithm.

## Immediate next work

1. Freeze this decision and the 40/40 evidence state.
2. Implement the canonical objective and decision specifications.
3. Run CPU-only conformance tests on TRL and verl.
4. Draft the identifiability theorem and attempt to break it.
5. Review Gates 2 and 3 together.
6. Only then write a replacement pilot preregistration and request compute.

No new GPU work is justified before steps 2--5 are complete.

## Supporting analyses

- [Literature landscape](./research/literature_landscape.md)
- [Simulated stakeholder exercise](./user_fit.md)
- [Synthetic policy comparison](./research/simulation.md)
- [Adversarial option review](./research/adversarial_review.md)
- [Decision synthesis](./research/decision_synthesis.md)


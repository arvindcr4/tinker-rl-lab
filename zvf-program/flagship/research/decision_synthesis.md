# Independent flagship decision synthesis

**Decision date:** 2026-07-20  
**Evidence boundary:** frozen preregistration and execution notes; completed
literature, simulated user-fit, and policy-simulation artifacts; plus a fresh
primary-source web check, an independent senior-area-chair challenge to the
provisional ranking, and the completed competitive red team read in full. The
ranking remains an independent synthesis: it adopts the red team's verified
collisions and falsifiers but distinguishes the best next paper core (the
conformance audit) from the highest-upside conditional theory spine (boundary-
information insufficiency). The simulation and
persona panel are decision aids, not experimental or human-subject evidence.
No S1--S4 result exists yet.

## Decision

**No option currently has an earned main-track claim.** S1--S4 are unstarted;
the following is a risk-adjusted choice of what to test and package next.

Lead now with the **cross-stack objective/gradient conformance audit**. Treat
**boundary-information insufficiency plus causal validation** as the higher-
upside promotion route only if it first passes a sharp relevance test in a
natural RLVR model class. Package the **algorithm audit benchmark** as the
results-independent fallback. Test the one-formula asymmetric controller only
as a deliberately small empirical challenge, not as the default headline.

The immediate priority is to prove that named treatments execute as specified.
The repository's unwired-loss incident makes this a demonstrated failure class,
not a hypothetical concern. The higher-upside conceptual move is to stop
treating the identical-reward boundary as if it identifies an action. The
repository has strong evidence that it is a useful diagnostic. It does **not**
have evidence that `retry`, `skip`, `retire`, or `change the reward information`
is generally optimal conditional on that diagnostic. The synthetic comparison
makes this concrete: among 12
scenarios, `static_g8`, `boundary_aware`, and `full_triage` each win four; across
the eight core stress regimes, mean synthetic AUC is nearly tied (`0.6436`,
`0.6420`, and `0.6437` respectively). Static G8 wins the sparse and
compute-constrained cases, while the adaptive rules win selected mixed or
mostly-correct cases. That is a model result, not training evidence, but it is
exactly the pattern an information-insufficiency claim predicts.

The simulated user panel points in a different but compatible direction:
the one-formula controller ranks first (mean rank `1.75`) and conformance second
(`2.00`) because users want an action and proof that it executed. That is a good
product priority. It is not, by itself, the best novelty priority: adaptive
allocation and zero-variance repair are now heavily occupied by DAPO, GRESO,
AERO, VIP, HORA, AVSPO, RL-ZVP, SGPO, EP-GRPO, MDP-GRPO, ISPO,
and related work. The theorem route becomes distinct only if it goes beyond
splitting all-wrong from all-correct: even a history-aware statistic containing
`p_hat`, `G`, boundary counts, and uncertainty must remain observationally
equivalent across natural regimes with opposite optimal actions. A generic
two-hidden-world no-free-lunch construction would not clear the novelty bar.

### Recommended paper thesis now

> **Shared RLVR algorithm names do not establish shared mathematical
> interventions. We introduce a versioned, fail-closed conformance standard for
> losses, masks, importance ratios, action selection, and flattened gradients;
> apply it to TRL and verl; and test whether correcting certified semantic
> mismatches changes end-to-end effect-under-a-frozen-stack verdicts.**

If and only if the boundary-matched action-reversal test passes, promote the
stronger thesis: **even `(p_hat, G, boundary counts/history)` is insufficient to
identify the optimal RLVR intervention in a natural model class; a sharp regret
bound and a minimal extra statistic predict when resampling, reward enrichment,
or temporary suspension wins.** This remains a conformance-certified causal
study, not a controller-win paper.

## Ranked recommendation

| Rank | Path | Decision | Why now |
|---:|---|---|---|
| **1** | **Causal cross-stack conformance audit** | **Select now; execute first** | Durable technology gap, directly motivated by the unwired-loss incident, and required before any causal outcome claim. |
| **2** | **Boundary-information insufficiency theorem + causal validation** | **Conditional high-upside promotion** | Potentially strongest novelty and explains the simulation, but only if stronger than a definitional two-world construction and empirically sign-reversing. |
| **3** | **Algorithm audit benchmark** | **Strong fallback / secondary artifact** | Best use of evidence already held; E1 has exceptional provenance, but needs open, longer, multi-regime extension. |
| **4** | **One-formula asymmetric controller** | **Run only as the smallest action test** | Highest simulated user fit, but fierce collision pressure and no universal simulated win. It must beat modern allocators, not only static G. |
| **5** | **Unified theory + intervention + controller** | **Promote only if every gate passes** | Highest upside, highest matrix and narrative risk. Present evidence cannot support the integrated headline. |
| **6** | **Mechanism-only ZVF** | **Keep as notation/calibration fallback** | The exact Bernoulli identity and unanimous-group observation are already absorbed by current work. |
| **7** | **Maximal controller** | **Drop** | An unidentifiable bundle in the most crowded lane, with the worst adoption and ablation burden. |

### Comparative scorecard

Scores are present-state judgments: `5` is favorable, except compute burden and
competitive pressure where `5` is costly/high.

| Path | Novelty durability | Evidence held | User fit | Compute burden | Competitive pressure | Reviewer defensibility |
|---|---:|---:|---:|---:|---:|---:|
| Boundary insufficiency + causal validation | 1 in weak form; **5 if strengthened** | 1--2 | 4 | 2--3 | 3 | 2 now; **5 if empirically nontrivial** |
| Cross-stack conformance audit | 4 | 3 | 4 | 2 | 2--3 | 4 |
| Algorithm audit benchmark | 4 | **5** | 4 | 3 | 3 | 4 |
| One-formula asymmetric controller | 2--3 | 2 | **5** | 4 | **5** | 3 only after strong wins |
| Unified chain | 4 if all gates pass; 2 otherwise | 2 | 3 | **5** | 5 | 4 if complete; 2 if partial |
| Mechanism-only ZVF | 1 | 4 | 2 | 1 | **5** | 1 |
| Maximal controller | 2 | 1 | 1 | **5** | **5** | 1--2 |

## Reviewer dossiers

### Boundary-information insufficiency theorem plus causal validation (rank 2)

**What and why.** Formalize the boundary observation as
`S = (p_hat, G, boundary counts/history, uncertainty)`. Actions include additional
same-verifier rollouts, continuing unchanged, temporarily suspending a prompt,
or acquiring richer reward/process information. Construct two latent RLVR
environments that induce the same distribution over `S` but have different
utility-optimal actions. If the utility gap is `Delta_1` in one environment and
`Delta_2` in the other, the best randomized `S`-only rule has worst-case regret
at least `Delta_1 Delta_2 / (Delta_1 + Delta_2)` in the two-action construction.
This is a small Blackwell-style information result, not another derivation of
`p^G + (1-p)^G`.

The empirical contribution must show that the construction describes real
failure modes. Matching `p`, `G`, and boundary frequency while changing rollout
dependence, verifier validity, or within-failure process information should
make the best intervention reverse. The existing binary-moderate, binary-sparse,
and graded-executable regimes give a natural test bed.

**Strongest novelty claim.** A decision-relevant impossibility result for RLVR:
boundary statistics can predict outcome-contrast starvation yet be insufficient
for selecting a compute or information intervention. The causal experiment then
identifies which added signal closes the gap.

**Closest prior.** The mathematical ancestor is Blackwell's
[comparison of experiments](https://doi.org/10.1525/9780520411586-009).
In RLVR, [AERO](https://arxiv.org/abs/2602.14338) and
[VIP](https://openreview.net/forum?id=Z5sWYACAop) act from success/variance
beliefs, whereas [RL-ZVP](https://arxiv.org/abs/2509.21880),
[SGPO](https://arxiv.org/abs/2505.11595),
[PRIME](https://arxiv.org/abs/2502.01456),
[EP-GRPO](https://arxiv.org/abs/2605.04960), and
[ISPO](https://arxiv.org/abs/2606.08815) introduce richer token/process or
intrinsic information. Direct 2026 pressure is stronger than the completed
landscape captured: [Gradient Starvation in Binary-Reward GRPO](https://arxiv.org/abs/2605.07689)
already analyzes all-fail/all-pass degeneracy and a fixed-reference sign
advantage; [AVSPO, ICML 2026](https://arxiv.org/abs/2605.21125) diagnoses
advantage collapse and restores gradients with virtual reward samples; and
[HORA](https://arxiv.org/abs/2605.07114) allocates by posterior hit utility.
Thus, boundary decomposition itself is not the novelty. The fresh search found
no primary RLVR paper establishing the strengthened natural-class observational-
equivalence/regret result.

**Kill criterion.** Kill the paper route if the lower bound is only a vacuous
"two hidden states can differ" no-free-lunch statement; if matching the
observable boundary requires pathological environments; or if realistic
boundary-matched cells do not produce a preregistered intervention-by-regime
interaction and action-ranking reversal. A proof that survives only by allowing
arbitrary rewards or unconstrained actions is not enough.

**Smallest discriminating test.** After S1 conformance, run one Qwen3-1.7B,
one base group size, three screening seeds, and two matched prompt strata. Match
their pre-intervention `(p_hat, all-wrong rate, G)` but make one stratum
resampleable (weakly correlated failures) and the other information-limited
(correlated failures with graded/stepwise distinctions). Randomize equal-budget
`+8 same-verifier rollouts` versus `richer reward information`; test the
seed-paired interaction on comparative-signal yield and learning-curve AUC.
The minimum success is a stable sign-reversing action-by-regime interaction,
mediated by gradient cosine/comparative-signal yield and learning AUC on both
stacks, not merely a main effect. Kill or demote the theorem if a reward-mean or
history baseline resolves the aliasing. Add the all-correct clean-versus-lenient-
verifier suspension test only after this lower-bound witness works.

**Precise path thesis.** Boundary prediction is possible, but boundary-only
control is not identifiable; added dependence, verifier, or process information
is necessary to select the intervention.

### Causal cross-stack objective/gradient conformance audit (rank 1)

**What and why.** Treat algorithm names and configuration flags as hypotheses,
not treatments. Versioned float64 fixtures certify loss values, masks,
importance ratios, selected groups/actions, reductions, and flattened gradients
for GRPO-family objectives and every controller action in a canonical reference,
TRL, and verl. A targeted correction must then change the predicted gradient
and at least one end-to-end outcome.

**Strongest novelty claim.** A fail-closed executable semantic certificate that
links formula -> implementation -> gradient -> outcome, including metamorphic
negative controls. This is more than unit testing if it yields a taxonomy of
semantic divergences and validates a general causal attribution protocol.

**Closest prior.** [Dr.GRPO](https://arxiv.org/abs/2503.20783) diagnoses
objective biases; [Group-Relative REINFORCE](https://arxiv.org/abs/2509.24203)
reinterprets clipping, regularization, and data weighting; official
[TRL](https://github.com/huggingface/trl),
[verl](https://github.com/volcengine/verl), and
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) expose overlapping algorithm
surfaces. These works raise the semantic problem, but they are not a common
versioned conformance oracle.

**Kill criterion.** Both stacks already agree on all realistic target semantics;
only injected toy bugs are found; float64 differences do not predict bf16 or
distributed behavior; or correcting a mismatch fails to alter the declared
gradient/outcome. Under the frozen protocol, failing `rtol <= 1e-6` and
`atol <= 1e-8` is a stop signal, not a publishable controller result.

**Smallest discriminating test.** Build the CPU float64 suite for GRPO,
mean-centered/no-std, DAPO filtering/clipping, sequence-level ratios, and the
registered boundary actions; run it on TRL and verl; demonstrate that it catches
the known class of inert/unwired treatment and one realistic semantic mismatch;
then repair one mismatch and run a small paired training cell showing the
predicted gradient or survival verdict changes.

**Precise path thesis.** Shared algorithm labels do not establish shared
interventions; executable conformance is necessary for causal claims about
RLVR algorithms and adaptive allocation.

### 3. Algorithm audit benchmark

**What and why.** Extend E1's survival-verdict methodology into a public,
versioned benchmark that separates faithful published-recipe reproduction from
common-stack algorithm isolation. Preserve `SURVIVES`, `DISAPPEARS`, `REVERSES`,
and `INCONCLUSIVE`, with paired seeds, equivalence margins, achieved power, and
full provenance.

**Strongest novelty claim.** The combination of declared-lever fingerprints,
fail-closed aggregation, local/W&B/HF reconciliation, immutable held-out hashes,
correction/resume receipts, and equivalence-aware verdicts is unusually strong.
E1's current disciplined result---DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO
remain `INCONCLUSIVE`---is valuable precisely because it does not turn low
power into a leaderboard.

**Closest prior.** DAPO provides an open end-to-end recipe; official shared
frameworks expose multiple objectives; objective papers compare selected
implementations. The closest competition is any broad shared-stack comparison,
but none of the primary sources checked combines faithful-vs-isolated strata
with E1's machine-contract and equivalence-aware survival logic.

**Kill criterion.** The open-stack replication reverses the DAPO equivalence;
fixtures show an E1 treatment was misimplemented; longer curves invert the
30-step conclusion; or task heterogeneity makes a pooled survival label
misleading. The current audit is also not a common-budget efficiency benchmark:
the red team notes that DAPO used roughly 1.5k--2.1k rollouts per seed while
GRPO/GSPO used 480. A private, one-task, 30-step snapshot is not an archival
benchmark.

**Smallest discriminating test.** On an open stack, compare only GRPO and DAPO
in two strata---faithful recipe and isolated algorithm---on Qwen3-1.7B,
GSM8K plus one sparse task, and longer curves with paired seeds. Require
conformance receipts and an equivalence-aware conclusion before adding arms.

**Precise path thesis.** Published RLVR gains decompose into recipe/stack and
algorithm effects; a conformance-certified, power-aware audit can state which
effects survive a shared implementation without converting uncertainty into a
negative claim.

### 4. One-formula asymmetric controller

**What and why.** Use a transparent rule with different actions for all-wrong
and all-correct groups: retry only when posterior rescue value exceeds full
cost; temporarily suspend currently mastered all-correct prompts; keep ambiguous
groups at base G. This is the strongest product proposition and the smallest
controller worth testing.

**Strongest novelty claim.** Not adaptive sampling, but causal parsimony: isolate
lower-boundary rescue from upper-boundary suspension with a pre-fitted rule,
explicit action disagreement, and end-to-end compute accounting.

**Closest prior.** [DAPO](https://arxiv.org/abs/2503.14476) filters accuracy-0/1
groups; [GRESO](https://arxiv.org/abs/2506.02177) skips predicted-uninformative
prompts; [AERO](https://arxiv.org/abs/2602.14338) combines adaptive rollouts,
rejection, and a Bayesian posterior; [VIP](https://openreview.net/forum?id=Z5sWYACAop)
uses GP success prediction and hard-budget variance allocation;
[HORA](https://arxiv.org/abs/2605.07114) allocates by posterior hit utility;
and [AVSPO](https://arxiv.org/abs/2605.21125) reports error-only,
correct-only, and combined interventions. Newer pressure also includes
[MDP-GRPO](https://arxiv.org/abs/2606.06058), which changes sampling and
advantages to restore gradients in homogeneous groups. The completed red team
also identifies CoDaPO, AR3PO, DEPO, and DynaMO as adjacent allocation,
selection, replay, and theory-plus-allocation collisions. A main-track test must
compare against at least one richer-information method, because all-wrong does
not imply that more identical outcome samples are the right fix.

**Kill criterion.** Same action as the best naive rule on at least 95% of
eligible steps; no paired AUC improvement over the strongest allocator;
final-quality inferiority in any frozen cell; no measured FLOP/GPU-hour saving;
per-task retuning erases savings; or secondary-stack sign reversal.

**Smallest discriminating test.** Before the full matrix, add the explicit
one-formula arm and one strong AERO/VIP-class baseline to a protocol version
frozen before scored runs. On Qwen3-1.7B and three reward regimes, use the
three disjoint screening seeds and require action disagreement, >=10% mediator
movement, positive paired AUC, and final-quality non-inferiority. The current
six-arm screen is sufficient for internal triage, but not for a 2026
main-track controller novelty claim.

**Precise path thesis.** A pre-fitted asymmetric boundary rule yields more
held-out learning utility per measured end-to-end compute than static,
symmetric, failure-only, and modern posterior-allocation baselines.

### 5. Unified theory + intervention + controller

**What and why.** Join calibrated boundary prediction, causal gradient
intervention, and controller utility in one ordered chain. This is attractive
only if every component independently survives and the controller remains the
minimal rule above.

**Strongest novelty claim.** A complete prospective chain from prediction on
unseen cells, through mediator-changing intervention, to matched-compute held-out
utility with sign preservation across stacks and binary-to-graded rewards.

**Closest prior.** Its pieces collide separately with the
[group-standard-deviation identity](https://arxiv.org/abs/2607.00152), the
[GRPO U-statistic/group-size theory](https://arxiv.org/abs/2603.01162),
RL-ZVP/SGPO/PRIME-style signal restoration, and AERO/VIP/GRESO-style allocation.
Integration is the only defensible novelty.

**Kill criterion.** Any frozen link fails: H1 calibration, S1 conformance, H2
utility/non-inferiority/efficiency, H3 action value, or H4 stack sign. Do not
keep the unified headline by replacing a failed component post hoc.

**Smallest discriminating test.** Do not run a separate test. Promote this
route only after the smallest tests for the insufficiency theorem, conformance
audit, and minimal controller all pass on disjoint data/seeds, then execute the
frozen S3/S4 confirmation.

**Precise path thesis.** A pre-fitted boundary model, a conformance-certified
causal intervention, and a minimal asymmetric controller form one validated
mechanism-to-utility chain across reward regimes and stacks.

### 6. Mechanism-only ZVF

**What and why.** Retain the exact binary accounting, calibration intervals,
boundary decomposition, and graded-reward generalization as a diagnostic layer.
Do not submit the raw identity as the flagship.

**Strongest novelty claim.** Only a prospective calibration/failure-boundary
result remains plausibly new: fit before training, predict unseen task/model/G
cells, quantify correlated-rollout and graded-reward failures, and relate the
prediction to measured gradient cosine/norm under a frozen objective.

**Closest prior.** The July 2026
[group-standard-deviation identity paper](https://arxiv.org/abs/2607.00152)
explicitly unifies GRPO, Dr.GRPO, and DAPO through the same statistic; the
[U-statistic paper](https://arxiv.org/abs/2603.01162) provides stronger
finite-sample/group-size theory; DAPO, AERO, and multiple 2026 methods already
operationalize zero-variance collapse.

**Kill criterion.** The theorem reduces to `p^G + (1-p)^G` plus monotonicity;
the group-size optimum depends on the chosen proxy; held-out miss rate exceeds
20% or pooled calibration error exceeds 0.05; or ZVF does not predict gradient
telemetry after reward scale, clipping, length, and dependence controls.

**Smallest discriminating test.** Freeze a pre-fit model and test H1 on unseen
task/model/G cells, including MBPP's graded reward and explicit correlated-
rollout diagnostics. This is already a useful result, but should be an analysis
paper or a section of the insufficiency/unified routes, not the flagship
mechanism claim.

**Precise path thesis.** Exact binary-reward accounting is a calibrated
diagnostic with measurable failure boundaries; it is not a universal learning
theory or action rule.

### 7. Maximal controller

**What and why.** A policy jointly controlling prompt admission, group size,
temperature, clipping, replay, continuation, retirement, and reward-information
acquisition is a research program, not a credible first paper from the current
evidence.

**Strongest novelty claim.** Only a formal constrained-control benchmark with a
versioned state/action API and quality/compute Pareto frontier could be distinct.
Feature aggregation is not novelty.

**Closest prior.** AERO, VIP, GRESO, PODS, DAPO, RL-ZVP, SGPO, EP-GRPO,
MDP-GRPO, ISPO, and current dynamic-allocation work collectively occupy nearly
every proposed control surface.

**Kill criterion.** Any static/minimal rule lies on the same Pareto frontier;
component marginal effects are unstable; probe/controller overhead erases
gains; actions do not transfer; or the ablation matrix cannot power attribution.

**Smallest discriminating test.** None now. A maximal controller becomes
reasonable only after several independently validated small interventions share
a common state/action representation. Until then, simulate it only as an oracle
ceiling outside the frozen protocol.

**Precise path thesis.** No defensible present thesis. Reframe as a future open
control benchmark if the minimal interventions first earn generality.

## Execution order and publication gates

1. **Finish S1 and make it independently publishable.** No GPU screening before
   canonical-reference, TRL, and verl conformance passes for every action path.
   Do not use the existing generic `ZVFController` as the registered treatment:
   the completed red team confirms that its actuator responds symmetrically to
   rolling ZVF even though its labels distinguish collapse from saturation. The
   six registered action policies require explicit implementations and fixtures.
2. **Run the two-stratum information-insufficiency witness.** The decisive
   endpoint is an action-ranking reversal under matched boundary observables.
3. **Choose the paper from evidence, not aspiration.** If the reversal is real,
   lead with information insufficiency and use conformance as the trust layer.
   If the reversal fails but S1 finds general semantic divergences, lead with
   conformance. If the divergences are limited but the open longer-horizon audit
   succeeds, lead with the algorithm audit benchmark.
4. **Run the minimal controller challenge only after its modern baselines are
   frozen.** If it beats the strongest naive and posterior allocator at matched
   tokens and measured end-to-end compute, the one-formula route becomes viable.
5. **Promote the unified route only after H1--H4 pass.** Otherwise keep the corresponding
   negative result and do not substitute a larger controller.

The controller baseline set is the one place where the current preregistration
and current publication burden differ. The six frozen arms are appropriate for
scientific screening, but a controller headline now also needs an AERO/VIP-class
allocator, a difficulty/curriculum baseline, and a richer-information baseline.
Because no scored screening run has started, any expanded confirmatory claim
should be frozen as a new protocol version before execution, never patched after
results.

## Reviewer-facing claim discipline

- Say **"identical outcome rewards under the frozen objective"**, not "no
  learning signal exists." RL-ZVP, SGPO, PRIME, EP-GRPO, and ISPO explicitly
  recover other signals.
- Say **"temporary prompt suspension with a recheck rule"**, not permanent
  retirement or mastery.
- Say **"matched generated tokens; measured FLOPs/GPU-hours reported"**, not
  generic matched compute.
- Say **"DAPO's effect is equivalent to zero in the frozen E1 estimand"**, not
  "DAPO does not work." The other three E1 arms remain inconclusive.
- Treat the simulation's G16 domination and controller wins as hypotheses about
  allocation, never as evidence for H1--H4.

## Primary-source verification notes

The fresh web check confirms the main strategic collisions:

- [Bay and Yearick (2026)](https://arxiv.org/abs/2607.00152) explicitly state
  the group-standard-deviation identity and connect GRPO, Dr.GRPO, and DAPO.
- [Zhou et al. (2026)](https://arxiv.org/abs/2603.01162) cast the GRPO gradient
  as a U-statistic and claim finite-sample and group-size results.
- [AERO](https://arxiv.org/abs/2602.14338) already combines adaptive rollout,
  selective rejection, and a Bayesian posterior, reporting compute and wall-time
  reductions under its setup.
- [VIP, ICLR 2026](https://openreview.net/forum?id=Z5sWYACAop) predicts
  per-prompt success with a GP and solves hard-budget variance allocation.
- [Gradient Starvation](https://arxiv.org/abs/2605.07689) already analyzes
  binary all-fail/all-pass collapse and a fixed-reference sign intervention;
  [AVSPO, ICML 2026](https://arxiv.org/abs/2605.21125) diagnoses advantage
  collapse and injects virtual rewards; and
  [HORA](https://arxiv.org/abs/2605.07114) supplies a learning-free posterior
  hit-utility allocator. These make a scalar-ZVF or endpoint-split theorem
  insufficiently novel.
- [RL-ZVP, ICLR 2026](https://arxiv.org/abs/2509.21880),
  [SGPO, TMLR](https://arxiv.org/abs/2505.11595),
  [EP-GRPO](https://arxiv.org/abs/2605.04960),
  [MDP-GRPO, ACL 2026](https://arxiv.org/abs/2606.06058), and
  [ISPO](https://arxiv.org/abs/2606.08815) all reinforce the same warning:
  identical outcome rewards do not imply that every useful intervention is
  another rollout-allocation rule.

One bibliographic correction matters for reviewer trust: arXiv `2505.11595` is
**Stepwise Guided Policy Optimization**, not "Spectral Policy Optimization."
The current primary record identifies it as accepted by TMLR. External-facing
materials should use the current title and acronym SGPO.

## Bottom line

The program can own **executable cross-stack conformance** now. It may be able to
own **"a boundary is a diagnostic, not a decision state"** after the natural-
class theorem and sign-reversing causal witness pass. That staged claim is more
durable than another adaptive-G method and more honest about the simulation's
conditional winners. The audit benchmark remains the evidence-backed fallback;
the minimal asymmetric controller remains a valuable test of where the
insufficiency boundary stops mattering.

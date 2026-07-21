# Flagship policy simulation comparison

## Decision summary

This is a **synthetic allocation simulation**, not an experiment, a
preregistration result, or confirmatory evidence for H1--H4. Its useful result
is narrower: under an explicit matched-token model, there is no universal
winner. Fixed `G=8` wins the severely sparse, mostly-wrong, and
compute-constrained cases because it buys more updates; boundary-aware control
wins the mixed-difficulty and larger/easier scaling cells; full triage has the
best registered AUC in transitional, mostly-correct, noisy-verifier, and
distribution-shift cases, but generally by small amounts. `G=16` is dominated
in this model's AUC-per-token accounting.

The simulated result therefore supports the frozen design's existing posture:
run all six screening arms, retain `static_g8`, `static_g16`, symmetric, and
failure-only controls, and advance only if the preregistered *training*
endpoints pass. It does **not** justify promoting a stateful bandit to the
flagship; that diagnostic adds state/features and a new decision rule beyond
the frozen six-arm protocol.

Reproduce the generated data with:

```bash
python3 zvf-program/flagship/research/simulation/run_simulation.py \
  --replicates 96 --sensitivity-replicates 36 --seed 20260720
```

Artifacts: [`run_simulation.py`](simulation/run_simulation.py),
[`scenarios.json`](simulation/scenarios.json),
[`policy_regime_summary.csv`](simulation/policy_regime_summary.csv),
[`sensitivity_summary.csv`](simulation/sensitivity_summary.csv), and
[`simulation_results.json`](simulation/simulation_results.json).

## What was compared

I interpret “six candidate flagship paths” as the six arms frozen in
`flagship/preregistration.json`. The first four rows are the requested concrete
allocation policies; the last two are the stronger registered variants. The
bandit is an extra diagnostic ceiling, clearly outside the protocol.

| Path / allocation rule | Action after an initial `G=8` group | Status in simulation |
|---|---|---|
| `static_g8` | Never expand | Frozen baseline |
| `static_g16` | Sample `G=16` directly | Frozen baseline |
| `symmetric_zvf` | Add eight rollouts after **any** observed zero-variance group | Frozen naive heuristic |
| `failure_only` | Add eight only after an all-observed-wrong group; do not expand all-correct | Frozen naive heuristic |
| `boundary_aware` | Failure-only expansion; replace all-correct prompts with fresh prompts from the same synthetic task distribution | Frozen candidate controller |
| `full_triage` | Boundary-aware plus a bin-level Wilson upper-bound/compute gate for retries | Frozen candidate controller |
| `stateful_bandit_diagnostic` | Boundary-aware plus Thompson-sampled stratum retry scores | Non-registered diagnostic only |

The “replacement” convention is a model of retiring mastered prompts, not a
claim that current training infrastructure already performs that action.

## Inputs, evidence boundary, and existing theory

The simulator reads the completed E1 JSON and copies the exact subset it uses
to [`e1_frozen_inputs.json`](simulation/e1_frozen_inputs.json). E1 was a
30-step, three-seed Qwen2.5-0.5B-Instruct measurement. Its frozen correlations
were gradient norm versus `p(1-p)`: 0.71, versus GU: 0.63, and versus ERF:
0.169. The five reported difficulty strata had mean reward probabilities
0.264--0.458. These numbers are used only as a weak Beta-prior centre and a
declared scale convention for synthetic learning dynamics. They are not
pooled with the new protocol, do not form an interval here, and do not
constitute support for the controller.

This is intentionally conservative with respect to the current theory. T3 in
[`zvf_theory.tex`](../../theory/zvf_theory.tex) establishes that the declared
signal-per-rollout proxy has universal optimum `G in {2,3}`; it does *not*
derive a data-dependent actuator. Thus a simulated adaptive win cannot turn
the controller into a theorem. It merely illustrates when a richer,
matched-budget objective might be worth testing prospectively.

There is also an implementation warning for S1. The existing generic
`ZVFController` smooths batch ZVF and increases group size when it is high.
Although it labels saturation separately, its default group-size action does
not itself implement the frozen “all-wrong expand / all-correct no-expansion
and retire” action path. Its per-prompt drop bookkeeping also keys on repeated
zero variance, not an explicitly verified all-wrong/all-correct action split.
S1 should therefore differential-test the *six registered actions*, rather
than treating the generic controller's default adaptive-G output as proof of
their equivalence.

## Simulation model

Each replicate contains a persistent pool of synthetic prompts. A prompt with
difficulty `d` is correct with probability
`sigmoid(global_skill - d)`. Responses are Bernoulli draws; the verifier can
independently flip a true positive or a true negative according to configured
sensitivity and specificity. A group contributes positive synthetic learning
only when both the latent rewards and observed verifier rewards have within-
group contrast. Its magnitude is `4p(1-p)`, attenuated by observed label
error. This deliberately penalizes false contrast rather than treating any
verifier variation as useful signal.

For each policy/scenario pair, the ceiling is exactly the cost of 42 updates
of 48 prompts at `G=16` (or 18 updates in the compute-constrained case).
Policies may take more small-group updates but cannot exceed the same number
of generated rollouts. The result records:

- **AUC per generated-token budget:** normalized trapezoidal area under the
  simulated held-out success curve. This is the primary ranking quantity.
- **Usable-gradient yield:** latent-and-observed-mixed groups per 1,000
  generated rollouts. False contrast is observed-mixed but latent-uniform.
- **Cost:** generated tokens and an explicit *proxy* FLOP calculation of
  `8 × parameter_count × generated_tokens`; it is not measured hardware
  FLOPs.
- **Regret:** AUC difference from the best *tested* policy in the same
  synthetic scenario. It is not regret against a real or oracle policy.

The stress cases are mostly-wrong, transitional, mostly-correct, sparse
reward, noisy verifier, distribution shift, prompt-difficulty mixture,
compute-constrained, and four model/task-scale cells (1.7B/8B ×
GSM8K-like/MATH-like). Their parameters are fully versioned in
[`scenarios.json`](simulation/scenarios.json).

## Main results

The table reports the best **registered** arm by simulated AUC, with `G=8` as
the fixed-token reference. The diagnostic bandit is excluded from selection.
Numbers are Monte Carlo means over 96 synthetic replicates; they are not
statistical confidence intervals for a real training effect.

| Regime | Best registered path | AUC/token | `full_triage - G8` | Useful groups / 1k rollouts of winner | Readout |
|---|---:|---:|---:|---:|---|
| Mostly wrong | static G8 | 0.4036 | -0.0187 | 96.4 | Extra retries cost more updates than they salvage. |
| Transitional | Full triage | 0.8411 | +0.0055 | 80.5 | Small conditional benefit; not remotely a confirmation-sized margin. |
| Mostly correct | Full triage | 0.9335 | +0.0021 | 53.3 | Retiring saturation is conditionally useful. |
| Sparse reward | static G8 | 0.1148 | -0.0079 | 62.8 | The controller cannot make truly scarce successes appear. |
| Noisy verifier | Full triage | 0.7880 | +0.0015 | 91.3 | Near tie; 21.6% false contrast under this noise model is the dominant risk. |
| Distribution shift | Full triage | 0.7395 | +0.0044 | 97.2 | Conditional gating is slightly more robust than unconditional retries. |
| Difficulty mixture | Boundary-aware | 0.8067 | +0.0168 | 84.5 | The clearest modelled benefit: avoid revisiting easy saturated prompts. |
| Compute constrained | static G8 | 0.5517 | -0.0027 | 113.4 | Few steps magnify the value of cheap updates. |

The scaling cells make the same conditional point rather than a scale-law
claim: boundary-aware wins the 1.7B GSM8K-like cell (0.8257 versus G8 0.8199),
8B GSM8K-like (0.8731 versus 0.8673), and 8B MATH-like (0.7764 versus
0.7697); G8 wins the 1.7B MATH-like sparse cell (0.6168). Those values are
outputs of this model's initial-skill assumptions, not observations at the
frozen Qwen checkpoints.

At the standard 1.7B-token profile every arm spends 8,257,536 generated
tokens, with the stated proxy costing `1.123e17` FLOPs. The
compute-constrained profile spends 3,538,944 tokens (`4.813e16` proxy FLOPs).
The 8B scaling cells use the same generated tokens but `5.285e17` proxy
FLOPs. Exact per-policy costs and simulated 95% percentile ranges are in the
CSV; matching is exact in these runs because the last batch is truncated to
the remaining rollout budget.

The six registered policies split the 12 scenario-level AUC wins evenly:
full triage 4, boundary-aware 4, and static G8 4. Across only the eight core
stress regimes, mean synthetic AUC is nearly tied for G8 (0.6436), full triage
(0.6437), and boundary-aware (0.6420), while G16 is 0.5473. Averaging across
heterogeneous regimes is not a proposed endpoint; it is included only to make
the lack of a universal controller win explicit.

## Cost, failure modes, and sensitivity

The simulator resolves a potentially misleading pattern: `G=16` usually has
a larger *fraction* of useful groups but fewer useful groups per rollout and
half as many updates under a generated-token ceiling. For example in the
mostly-wrong case it produces 55.7 useful groups/1k rollouts versus G8's 96.4
and its AUC is 0.2018 versus 0.4036. This follows directly from the specified
learning model and agrees qualitatively with the theory's warning that the
simple per-rollout proxy favours small groups; it does not measure real GRPO
learning.

Symmetric expansion is consistently weak in the model because it pays to
expand all-correct groups even though they are saturation, not cold-start
starvation. Failure-only removes that obvious waste but still loses in severe
sparsity: a single all-wrong observation need not mean that retrying is the
best use of a fixed budget. Boundary-aware wins when the persistent prompt pool
contains a meaningful easy component to retire. Full triage's extra gate helps
only when its state estimate is sufficiently calibrated; that is precisely the
assumption to test in S1/S2.

Seven sensitivity probes perturb the least defensible modelling choices:
learning gain, E1-prior strength, verifier reliability, and shift severity.
No policy wins all probes. Full triage leads clean verifier, harder shift, and
both learning-gain probes; failure-only has the top AUC under severe verifier
noise (0.7293, only 0.0001 above full triage); and G8 leads under both weak and
strong E1-prior variants of the mostly-wrong case. This sensitivity is a reason
to keep the naive arms in screening, not evidence to tune thresholds after
seeing results.

The major unmodelled or fragile assumptions are:

- Latent `p` and independently flipped verifier labels are much simpler than
  correlated completions, reward-parser bugs, and executable-verifier errors.
- Retiring an all-correct prompt assumes it is genuinely mastered and that a
  replacement prompt is representative. Either can fail under contamination,
  reward hacking, or curriculum feedback.
- The bandit assumes stable, observable difficulty strata. Distribution shift
  can make its posterior stale, and the policy is not part of the frozen arm
  set.
- A skill increment proportional to verified contrast is a simulation
  convention, not a GRPO gradient derivation. Real update quality also depends
  on sequence likelihoods, clipping, KL, advantage normalization, and
  optimizer state.
- A Monte Carlo percentile band over synthetic random seeds is neither the
  preregistered seed-paired bootstrap nor a real uncertainty interval.

## Strategic path comparison

Ratings are decision-oriented (Low/Medium/High), not results.

| Path | Technical novelty | Experiment burden | Compute risk | Interpretability / adoption | Venue risk | Simulation-informed role |
|---|---|---|---|---|---|---|
| Static G8 | Low | Low | Low | High / easy | High: baseline only | Essential efficient baseline; wins sparse/short-budget cases. |
| Static G16 | Low | Low | High | High / easy | High: baseline only | Required comparator and target, not a likely flagship. |
| Symmetric ZVF | Medium | Medium | Medium-high | High / easy | High: ignores boundary semantics | Necessary ablation; weak expected winner. |
| Failure-only | Medium | Medium | Medium | High / easy | Medium | Important naive H3 control; robust fallback under severe noise. |
| Boundary-aware | High | Medium-high | Medium | High / medium | Medium | Best low-complexity controller candidate; strongest in mixture/scale cases. |
| Full triage | High | High | Medium | Medium / medium-high | Medium-high | Best registered flagship candidate only if S1 makes its actions exact and S2 exceeds the naive arms. |
| Stateful bandit | Very high | Very high | Medium | Low / high friction | High | Keep as an exploratory ceiling; adding it as a claim requires a new protocol. |

The simulation-informed sequencing is therefore unchanged from the frozen
protocol: (1) make action-level objective tests prove the intended boundary
behaviour on both stacks; (2) screen all six arms at matched tokens; (3) only
then let the expansion gate choose full triage versus a mechanism-only /
negative-controller route. A mechanism-plus-controller venue story is viable
only if the real seed-paired AUC, non-inferiority, and FLOPs-to-target gates
all pass. If the mechanism predicts but the controller ties the naive arms,
the honest route is the preregistered mechanism-only negative controller
paper—not a post-hoc bandit story.

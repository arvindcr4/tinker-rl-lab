# Hybrid experiment protocol v2: evidence and interpretation contract

This is a prospective repair protocol, not evidence that an experiment was run.
No training, remote evaluation, historical artifact modification, or paid job was
performed for this review. Historical results remain unaudited and are not
converted into new measurements by these changes.

## Evidence boundary

- Simulation entrypoints require explicit opt-in, produce `execution_kind=simulation`,
  `measured=false`, `claim_eligible=false`, and `artifact_type=illustrative_metrics`,
  and place emitted files under a `simulations/` subdirectory. They do not upload
  model repositories or write dummy checkpoint-shaped files
- Empirical aggregation requires completed, measured, claim-eligible receipts with
  independently identified `run_id` and `receipt_id`. Duplicate units are rejected
- A process-exit receipt is not a measurement receipt. Modal execution receipts
  deliberately carry `measured=false` and cannot pass the empirical gate by default
- Preserve actual evaluation IDs, checkpoint paths, missingness, sample counts,
  source records and interval definitions. Never substitute training rewards for
  held-out outcomes or treat trace steps as independently trained models
- The receipt validator checks declared provenance, not its historical authenticity.
  A run-to-artifact audit must establish which old outputs actually satisfy it

New empirical analysis:

```
python -m platform_hybrid.experiments.measured_analysis \
  receipts.json hypotheses.json --out measured_analysis_v2.json
```

Each hypothesis names `hypothesis_id`, `comparison_id`, `endpoint`, `arm_a`,
`arm_b`, and whether the design is paired. Each receipt also needs these matched
fields: `model_revision`, `initialization_id`, `adapter_config`, `dataset_hash`,
`prompt_hash`, `grader_version`, `evaluation_ids_hash`, `budget_kind`, and `budget`.
Paired designs require identical unique `pair_id` sets. The whole supplied,
prespecified hypothesis family receives one Bonferroni correction. Bootstrap
resampling is across independent runs (or matched run pairs). One run per arm
is descriptive evidence only and is refused for these inferential summaries.

No checked-in receipt is newly certified by this patch. Missing inputs are an
explicit eligibility failure. Historical aggregate means/SEs could support a
separate provenance-validated aggregate-statistics test, but no aggregate-only
claim is enabled without that provenance.

## D1: Difficulty strata are not a causal intervention

The existing dead/mixed/saturated arms alter prompt content, difficulty, reward
headroom and response length. Name the design difficulty-stratified and
observational with respect to the effect of ZVF. Record pilot counts and Wilson
intervals; service failures are unmeasured, never dead examples. Four failures
have probability `(1 - .25)^4 = .31640625` at true success probability .25. They
do not establish impossibility. A fresh G=8 group has contrast probability
`1 - .25^8 - .75^8 = .899871826171875` under independent draws.

A future direct grouping study must freeze each prompt's completion/reward
multiset, partition it into mixed versus homogeneous groups, use identical
initial model and optimizer copies, retain all gradient denominators and update
schedules, and compare immediate gradients before common disjoint evaluation.
This new experiment was not run.

## D2: Experimental unit, uncertainty, power and equivalence

Use independently trained, matched runs and a prespecified terminal endpoint for
algorithm comparisons. Retain prompt-level pairs during evaluation, resample
seed blocks, and account for prompt clustering where applicable. Temporal
within-run summaries are descriptive and conditional on that run. They cannot
supply independent seeds to a bootstrap, Welch test or Mann–Whitney test.

A registered design must state independent-run count, power assumptions and a
meaningful effect or equivalence/noninferiority margin. A nonsignificant test is
not proof of equivalence. Existing N=5/N=10 power illustrations are not a
registered protocol and must not determine a new claim's sample count.

## D3: Matched arms and honest objective names

The measured-analysis contract refuses mismatches in model revision,
initialization, adapters, dataset/prompt/grader/evaluation hashes and budget.
The legacy PPO-named TRL file actually uses RLOO and now emits `trl_rloo`.
It is not a matched GRPO arm: it still needs an approved resolved configuration
and runtime preflight. Classic MLP arithmetic policies and pretrained LLMs have
different representations/action spaces. DPO adds supervised preference
construction and data coverage; interpret that intervention explicitly.

The structural and notebook objectives are custom group-centered policy-gradient
variants; do not inherit PPO/GRPO guarantees from old filenames. The multiturn
path now scores exact stored states/actions, disables dropout in generation and
scoring, and samples untruncated temperature-one probabilities. Its tool reward
remains a custom format/chain diagnostic, not broad semantic tool-use success.

## D4: Budget questions and accounting

Separate fixed-update, fixed-attempt and fixed-cost studies. Log prompt tokens,
completion tokens, rejected samples, actual training tokens, evaluation tokens,
optimizer updates, elapsed time and measured billing separately. The repaired
completion-token runner caps each sampled group's maximum completion length
before sampling and bounds attempts; that is not a total-token or billing cap.
No reconstructed assumed sequence length is accepted as an actual token budget.

Keep denominators and optimizer schedules fixed in a filtering-only mechanism
study. Evaluate a compute-saving implementation separately because skipped Adam
updates and survivor averaging change the algorithm. The sweep's planned-dollar
sum is a preflight safeguard, explicitly not a live billing cap. A live
provider-enforced spending limit/receipt contract is still required.

## D5: Predictive validation

Use actual step IDs for lag joins, including gaps. Keep predictor and target
windows disjoint, compute prefix features online, and hold out entire runs or
configurations before feature engineering. The P8 classifier refuses too few
independent runs per class instead of leaking trace identity across folds.

Reward-only baselines must include nonlinear reward, trends, group size,
training stage, prompt/task heterogeneity and appropriate run/task effects.
Use held-out-run predictive performance and cluster-aware uncertainty. Ordinary
trajectory OLS p-values remain unsupported; the old inferential report entrypoint
fails closed. Calibration needs independent pilot and validation rollouts, not a
noisy plug-in success estimate compared on the same samples. No such experiment
or out-of-run performance claim was fabricated.

## D6: Operational finite-budget scope

Describe observed plateaus within tested budgets. Define task, objective,
budget, evaluation distribution, precision, stopping rule and uncertainty in the
manifest. MATH-500/HumanEval test items used for training are adaptation data;
they cannot also prove held-out generalization. Structural finalization now
records this limited scope and missing held-out measurements.

Retain failed/flat curve fits and all requested measured traces. The repaired
plot shows observed trajectories rather than selectively showing plausible
extrapolated asymptotes. Longer budgets, alternate learning-curve models and a
shared disjoint final-evaluation endpoint remain future experiments.

## D7: Layer concentration

Layer gradient norms depend on parameterization and reduction; LoRA factor
rescaling can change them without changing effective updates. A zero profile has
undefined ranking, not perfect overlap. Predictor and terminal windows must be
disjoint. Retain genuine EOS targets and exclude only later padding.

Select layers on independent data, compare hot/cold/random layers with matched
parameter counts, then measure effective weight changes, held-out outcomes,
runtime, memory and actual work separately. Trainable-parameter fractions are
not FLOP measurements. No such measurements or noninferiority conclusions were
created by this code repair.

## D8: Semantic tool evaluation

Freeze disjoint templates, entities and tool families before training, with
identical definitions and executors for all arms. Report schema validity, tool
choice, argument values, execution success and final-answer grounding separately.
Fixtures bind exact names and canonical arguments; unrelated calls receive an
error, never the next correct scripted observation. Unknown tools do not return
made-up successful results. Exact-answer fixture matching is a conservative
custom endpoint, not broad semantic equivalence.

Legacy four-scenario notebook evaluation is ineligible until frozen semantic
fixtures are supplied. The Tesla overlap is not retroactively cured by altering
past splits. New stopping behavior is unassisted and must be versioned separately
from historical wrap-up-assisted results. BFCL dry runs are labeled custom
simulations; the real integration is explicitly unavailable.

## D9: Execution/restart contract and preflight

`research_contracts.validate_manifest` requires protocol version, code/package/
model revisions, objective, seed and RNG states, data/prompt hashes, grader
version, actual budgets and checkpoint paths. `validate_resume` rejects contract
changes. Structural legacy resume is disabled because it lacks compatible local
RNG/baseline/diagnostic state. Causal training applies the recorded seed and
validates versioned bin identity; old bins are not silently reused.

Pin and inspect the intended TRL/Tinker APIs before resource allocation. The
Modal launcher uses supported CLI configuration instead of generated source
patches, with source compilation before scheduling. This is syntax preflight,
not certification of all external runtime API generations. The review's offline
venv does not supply a live Tinker/TRL/Modal deployment, credentials, real model
revisions or trusted generated-code sandbox.

HumanEval candidate construction includes prompt, completion, tests and
`check(entry_point)`, but execution is deliberately unavailable without trusted
isolation and a deadline. A subprocess, restricted globals dictionary, or
thread timeout is not that isolation. No generated candidate was executed in
this repair or its tests.


### Public loader provenance and prompt failures

Legacy report entrypoints now enforce the v2 receipt boundary directly. Container
simulation/ineligibility/failure flags are restrictive and cannot be overridden by
a nested row. Held-out checkpoint summaries need the evaluation's own completed
empirical receipt and matching saved checkpoint ID; training receipts cannot fill
missing evaluation provenance. An unknown held-out denominator stays unknown.

Paired prompt reports require eligible control and treatment evaluation receipts.
Only explicit correct/incorrect/wrong grader statuses are binary observations.
Timeout, missing, service error, unknown status and explicitly unmeasured outcomes
remain missing. One common measured prompt intersection is applied to every arm,
with original per-arm statuses and exclusion reasons in `measurement_exclusions`.
The resulting prompt-conditional estimates do not establish independent-run
causal effects. A shared scalar baseline is only descriptive in the legacy
base/instruct table; matched independent-run inference uses `measured_analysis`.

When a historical lagged record lacks a run ID, canonical record content supplies
a conservative deduplication key, independent of source file and traversal path.
Explicit seed/config/attempt differences remain distinct. Identical unidentified
independent runs cannot be certified as separate observations; durable run IDs
are required for empirical inference, whose legacy entrypoint remains disabled.


## Bounded student-trajectory distillation follow-up (C18)

The separate `student-trajectory-kl-v1` contract is documented in
[`STUDENT_DISTILLATION_V1.md`](../../platform_hybrid/experiments/implementations/STUDENT_DISTILLATION_V1.md).
It generates fresh local student continuations and scores token-mean forward
KL(teacher || student) on their valid response positions, with detached teacher
scores and stop-gradient sampling. This is a semi-gradient objective, not a
policy-gradient estimator of expected trajectory KL. It neither relabels the
fixed-history objective nor changes historical results. Synthetic CPU and tiny
random-model generation/backward checks use no optimizer steps; full Trainer
initialization remains skipped here because `accelerate>=1.1.0` is absent.
GPU execution, real-checkpoint compatibility and empirical training/performance
remain unverified; the contract is restricted to local single-device execution.

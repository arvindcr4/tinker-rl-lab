# Modal/OpenRLHF candidate protocol v2

Status: **candidate, unvalidated on the real GPU stack**. No training or paid jobs
were run while implementing this protocol. Historical artifacts are unchanged.
This is not a claim that the independent
`platform_hybrid/experiments/modal/modal_grpo_openrlhf.py` launcher (OpenRLHF 0.8.4)
is repaired, that any historical run used 0.8.0, or that frameworks are matched.

## Execution boundary

- `platform_modal.openrlhf.run(config)` dispatches real execution; missing backend,
  unsupported treatments, missing authentication, missing pinned inputs, failed
  processes and incomplete histories raise. There is no simulation fallback.
- `run(config, dry_run=True)` returns a separate `synthetic_fixture` schema with
  uncalibrated seeded random fixture values, without empirical reward-summary keys.
  `require_empirical_result` rejects this schema. Repaired JSON-trajectory consumers
  reject explicit synthetic/failed inputs. Legacy data lacking provenance is not
  retroactively authenticated by this change. The direct framework-comparison
  aggregator is read-only, has no training/simulation fallback, rejects untyped
  legacy artifacts until independently validated, and writes only a versioned
  future output rather than overwriting the historical comparison.
- The candidate driver requires the distribution `openrlhf==0.8.0`; the candidate
  requirements file is **not** a validated container/dependency lock. Before a paid
  campaign, assemble the exact image and run full CLI-parse, distributed reward,
  tokenizer and loss-parity smoke tests. The implemented help-option preflight and
  CPU callback fixture are necessary but insufficient integration checks.
- GRPO is the only supported algorithm. Unsupported nondefault settings, held-out
  evaluation and quantization are rejected. Model revisions must be pinned or an
  explicit immutable local snapshot supplied. The local path's immutability remains
  the operator's responsibility. Dataset revision is required for materialization;
  custom JSONL is content-hashed. Native tokenizer chat templates are used for GSM8K;
  supplied JSONL is explicitly labeled as supplied formatting.

## Work and objective contract

`max_steps` means optimizer updates; `epochs` means optimizer epochs per rollout.
A rollout comprises `rollout_batch_size` prompts and `sample_num` completions per
prompt. `train_batch_size` counts completions. Exact divisibility is required:

- completions per rollout = rollout batch × sample count
- updates per rollout = completions per rollout / train batch × epochs
- rollout rounds = max steps / updates per rollout
- retained prompt budget = rollout rounds × rollout batch

The v2 defaults are 8 prompts, 16 completions/prompt, a global training batch of
128 completions, one epoch, and 30 intended updates. They intentionally differ
from the defective legacy command and cannot be compared as the same treatment.
The legacy 8-prompt, 8-completion, batch-8 example is 30 rollouts / 240 intended
updates. Preprocessing that changes the retained prompt budget is rejected.
Configured budgets are labeled **configured, not observed**. Reward observations
are not reported as measured optimizer updates or token counts. Exact worker
counters, worker seed derivation and GPU-worker allocation telemetry are pending.

The executable CPU objective specification declares sample-SD group normalization
(correction=1; epsilon=1e-8), response-wise masked token means followed by a batch
mean, clipped surrogate bounds, frozen initial reference, and k3 KL in the actor
loss only. Zero reward contrast does not imply that KL gradients are zero. These
are candidate contract fixtures, **not evidence that upstream optimizer execution
matches them**. No double-KL defect is alleged.

## Attempt identity, metrics and credentials

Each launch receives a fresh private directory, UUID, W&B environment run ID and
atomic attempt manifest. No parent W&B run is created. The credential is inherited
in the environment and never appears in argv, logged configuration, command text,
or exception messages written to manifests. The child gets a nonsecret W&B enable
value. Historical logs were not accessed; operators should inspect and rotate any
credential exposed by the old launcher.

The fallback metric source is an exact W&B run ID with state `finished`, using
`scan_history` and the pinned `train/reward` key. Reward steps must be unique,
contiguous, finite and match the expected rollout count. Validated observations
are mirrored to a local, attempt-bound metrics file. This is **not** worker-side
local source-of-truth instrumentation; remote consistency or schema differences
fail closed as `incomplete`, rather than being converted to zero. Local worker
metrics remain an integration requirement.

States are `running`, `failed`, `incomplete`, and `completed`; unavailable summaries
remain null. Nonzero exits/timeouts never create an accepted result. Only completed
validated runs get `result.json`. Software package versions, available repository
commit, runtime platform/GPU information, supplied image digest, input/template and
reward-parser hashes, seeds, objective and effective command are recorded. Missing
image/backend commit/worker metadata remains explicitly unavailable.

## Resource and recovery policy

Colocated actor GPU count must match vLLM engines × tensor-parallel size. The driver
forwards memory utilization and uses explicit checkpoint intervals. The current
policy is fresh attempts only: no automatic resume or checkpoint upload, no model
reload for export, and no parent allocator metrics mislabeled as worker metrics.
A future resume implementation must verify attempt ownership and restore model,
optimizer, scheduler, consumed-data and RNG state; it must not reuse an old export
just because a directory exists. No CodeCarbon tracker starts at import.

## Scientific producer protocol

All repaired producer outputs go under
`platform_hybrid/experiments/results/protocols/pro-review-01-v2/<producer>/`.
Imports do not read/write results or create output directories. Required inputs
are checked before output creation. Each generated artifact is covered by an
`artifact_manifest.json` recording its evidence type, actual method and input
hashes. Historical files are not overwritten; the P6 producer never edits registry
conclusions. No producer was rerun against measured data during this repair.

- P6 preserves task-specific reported intervals. The equally weighted mean is
  descriptive only, with no pooled CI, bootstrap metadata, significance or
  directional proxy verdict. Raw paired samples are needed for a fixed-task
  uncertainty calculation. Population-of-task claims are not supported here.
- StateFlow labels are retrospective and use future terminal reward. Missing states
  and low-advantage-variance events remain censored. Stopped-policy retention is
  unavailable without common held-out checkpoint evaluations. Retrospective
  optimal three-segment fits, state means/order, group-size schedule associations
  and observed low-variance lags remain available with censored-event counts. A low advantage
  variance is not called gradient death. There is no stopping-effectiveness claim.
- NIAH preserves unrecovered events as null, reports recovery coverage, and compares
  H2 predictors only with nonoverlapping later targets. This is exploratory
  association, not independent forecasting or equivalence validation. The position
  curves, pooled position/width recovery grid, and ramp-vs-U-shape descriptive
  residual comparisons are retained without confirmatory verdicts.
- DPO local pair-gradient scaling is beta/2 at a frozen equal reference, with
  sigmoid-dependent weighting away from it. A tied differentiable reference gives
  zero. IRPO optima and feasibility remain unmeasured; ratios are not pair counts.
- Paired comparison uses one unique matched-seed dataset for mean difference,
  effect size, permutation and equivalence. Five pairs enumerate all 32 sign
  assignments. Paired t-equivalence uses a 90% interval at alpha=.05 and a candidate
  .005 margin; it is explicitly exploratory and not preregistered. Degenerate
  samples are inconclusive. Terminal observations are sorted and checked for
  duplicates/gaps; a known final-step budget is required to verify completion.
- AlphaStar/CFR mappings are untested analogies and descriptive thresholds.
  Constant ranks and missing cells cannot pass. The CFR simulation is an iid
  Bernoulli formula check, not a CFR update comparison. ZVF=1 means all rewards
  agree. A deviation from an iid baseline does not identify herding.
- HippoRAG uses fixed-personalization weighted PPR with normalized transitions and
  dangling redistribution. Total probability mass has no correlational signal.
  BM25@25 actually evaluates 25 candidates; a cutoff comparison is not a cost
  measurement. Random ordering uses a recorded stable digest seed.
- The former MIPRO demo is named an anchor-UCB **synthetic heuristic**, not a GP or
  actual MIPRO. Landscapes vary by seed, strategies have equal bounded budgets,
  observations retain candidate identities, and top quartiles use quantiles. The
  legacy global-eight-evaluations / nine-of-twelve-anchors criterion is impossible
  and is reported as an infeasible design, never replaced with exhaustive maxima.
  Observed score variance is not posterior uncertainty.

## Sweep planning

The sweep generator requires `--backend tinker-atropos`, the sole implemented
route. Its default is a bounded three-configuration pilot, with explicit seeds and
shell-safe argument/display quoting. It generates plans only. A separate runner
records config-hash/seed attempts, skips an identical successful process, and
records failures/interruption. Process success does not validate empirical output.
Held-out screening and independent confirmation seeds require an explicit next
campaign decision; no automatic 108-run campaign is launched.

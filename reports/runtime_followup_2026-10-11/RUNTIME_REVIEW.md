# Second-pass runtime review — 11 October 2026

Scope: reviewed local commit `d47e2b9c` against the upstream versioned source,
its caller, and the frozen thesis statements. This is a fresh source/contract
review by the same assistant, not a separate human or independent final-thesis
review. No GPU or provider execution, scientific experiment, push or publication
was authorized or performed in this continuation.

## Findings addressed

1. The previous command named OpenRLHF 0.8.4 but accepted any installed version.
   Native preflight now rejects a missing or different distribution version
   before data preparation. This checks package metadata, not code authenticity.
2. Reusing an output directory without `result.json` could mix stale checkpoints
   or logs with a new run. The wrapper now reserves a wholly new directory and
   preserves every existing output directory, including interrupted runs.
3. Runtime checks reject partial prompt batches, impossible distributed
   micro-batches, oversized colocated tensor parallelism and invalid logging
   paths. These are necessary checks, not proof that a topology fits in VRAM.
4. Local native reward metrics are now recoverable without W&B display-name
   lookup. TensorBoard is the default; W&B requires explicit opt-in and suppresses
   local metrics under this backend. The reader accepts exactly one event file,
   the exact `train/reward` tag, finite scalar values at consecutive steps from
   one, and exactly the planned number of rollout records. It disables scalar
   subsampling/orphan purging and binds the parsed file's stable hash. Missing,
   extra, changed, ambiguous or malformed evidence produces no reward trace.
   These checks bind local bytes; they do not authenticate training execution.
5. The old zero/empty metric fallback remains removed. `steps` stays null.
   `logged_rollout_steps` is explicitly distinct from optimizer updates and
   held-out evaluation. No summaries are borrowed from historical runs.

## Why the optimizer-step budget remains blocked

The [v0.8.4 CLI](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/cli/train_ppo_ray.py#L278-L302)
exposes sample, rollout, minibatch and epoch controls, not an optimizer-update
stop option. The wrapper therefore continues to reject `max_steps`.

In the [controller](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ppo_trainer.py#L276-L321),
`max_steps` is calculated for scheduling from dataset size and batch settings.
The [execution loop](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ppo_trainer.py#L392-L487)
iterates over the prompt loader and episodes; its logged step advances once per
rollout. Each actor fit can perform several minibatches and inner epochs.
The [actor loop](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ray/ppo_actor.py#L132-L181)
and [DeepSpeed strategy](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/utils/deepspeed/deepspeed.py#L90-L144)
then apply gradient accumulation. Setting a dataset size or counting reward-log
rows is not an observed optimizer-update stop condition.

A restricted arithmetic planner could estimate an update count under divisibility
and execution assumptions. It would not add a stop/checkpoint contract at the
actor boundary or verify effective optimizer updates. A supported adapter needs
versioned actor-side enforcement, explicit treatment of accumulation/replay and
resume, and completed-step evidence. Patching Ray actor internals or substituting
a scheduler count on CPU alone would not establish that contract. This phase
therefore implements honest native metrics and rejection, not a new training
engine. A GPU/backend execution test remains necessary before declaring parity.

The unified launcher also has canonical gradient-accumulation and LoRA-alpha
settings beyond this native command's supported contract. Its rejected path
must not be described as a complete mapping of that canonical experiment.
The separate historical Modal script has not been changed or validated here.

## Reproduction and manuscript scope

See [MANUSCRIPT_ERRATA.md](MANUSCRIPT_ERRATA.md) for exact affected passages and
[README.md](README.md) for the unchanged campaign SD reconciliation. Signed
records and strict gates were not altered. Frozen publication hashes are not
refreshed to hide any change. New validation results are recorded below after
this continuation's checks; the earlier 1,280-test run remains separately dated.

## New checks for this continuation

- Fresh complete CPU/native-document suite: **1,306 passed, one intentional
  non-CPU-device skip, three warnings; 84.53% configured coverage**, 91.63 seconds.
  This is a new run, adding 26 tests to the prior 1,280-test baseline.
- Focused native command, metrics and launcher contracts: **56 passed**.
  The 26 added cases cover version/dependency preflight, stale output,
  batch/topology/logging constraints, explicit W&B opt-in, full event retention,
  missing/extra/nonfinite/duplicate/unordered observations and file changes.
- Separate synthetic binary TensorBoard round trip: passed with the installed
  `SummaryWriter` and real `EventAccumulator`, mocked subprocess and dataset,
  no model or provider execution. Two planted rewards were recovered, while
  optimizer `steps` remained null. This is a file-format test, not a real run.
- Repository Ruff/policy/format checks: passed. New command/metrics modules
  passed additional explicit Ruff checks. Configured mypy: 20 files, passed.
- Nine aggregate audits passed; public stdlib checks: 237 passed.
- Exact frozen October 11 manifest byte/hash verification: passed. The full
  revision/review checker still fails at pending independent review, as intended.
- Campaign/sweep signed bytes and strict verifier bytes match `d47e2b9c`.
  Both strict provenance checks still exit 1 (95.7% and 82.6% respectively).
- Historical campaign source hash, seed sets, gain metric and both per-arm SD
  estimators were recomputed and matched the unsigned reconciliation receipt.

The coverage and mypy figures describe their configured repository surfaces,
not complete native-backend validation. This backend remains a source-checkout
integration; the existing wheel check does not certify native OpenRLHF support.
No new GitHub CI, GPU test, paid job, external upload or publication occurred.
The original developer checkout was left untouched, and the pre-existing
untracked `.coverage` was not used by the fresh suite.

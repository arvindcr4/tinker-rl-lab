# Native whole-batch budget fix — 11 October 2026

Continuation from `da00ceac2`. The previous wrapper checked global minibatch
size and prompt-loader tails separately, but not whether an entire rollout
contained whole global training batches. Thus 8 prompts × 8 completions with
`train_batch_size=1024` was accepted, although a one-epoch actor fit supplies
only 64 completions. Accumulation could cross actor-fit boundaries; smaller
uneven shards could also be dropped. Native preflight now rejects these layouts
before provider imports, data preparation, or output creation.

The explicit supported mode is **synchronous whole batches**: one dataset pass,
no resumption or dynamic filtering, an integer number of prompt rollouts,
whole distributed microbatches per global batch, and whole global batches per
rollout. Multiple inner epochs remain supported. This restriction avoids
partial accumulation between inner epochs or fits; it is not an optimizer-update stop hook.
Existing native configs using the oversized default global batch must choose
an explicit compatible batch size. Nothing silently rewrites their budget.

## What the new plan records

`plan_native_budget` returns a schema-tagged arithmetic plan, stored separately
as `budget_plan` in the execution receipt. For 32 prompts, rollout size 8,
group size 8, global batch 16 and three inner epochs, it reports:

- 4 planned rollouts;
- 12 planned global update boundaries per rollout;
- 48 planned global update boundaries in total.

These are configuration calculations, **not observed optimizer steps**.
Execution receipts continue to set both `steps` and `optimizer_steps` to null.
The native TensorBoard reader reports only `logged_rollout_steps` and training
rewards when its independent trace checks pass. A missing trace stays missing;
a plan never supplies measurements. The old 240-prompt, R=8, G=8, global-batch=8,
one-inner-epoch settings would imply 30 rollout iterations and 240 planned
update boundaries, not 30 optimizer updates.

A minimal configuration for this source-only dataset-budget path is:

```python
from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf.budget import plan_native_budget

config = OpenRLHFConfig(
    algorithm={"algorithm": "grpo", "sample_num": 8},
    remote_rm_url="http://127.0.0.1:8765/get_reward",
    max_steps=None,
    max_samples=240,
    rollout_batch_size=8,
    train_batch_size=64,
    micro_batch_size=1,
    epochs=1,
)
plan = plan_native_budget(config)  # arithmetic only; no provider execution
```

The reward URL must refer to a service actually provided by the operator; the
example does not create or validate one. Native execution also requires the
exact supported package version, TensorBoard, Ray/vLLM/DeepSpeed and an adequate
GPU environment, plus a new output directory. None was executed in this fix.

## Backend basis and exact unresolved dependency

The versioned [OpenRLHF actor loop](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ray/ppo_actor.py#L133-L181)
drops incomplete microbatches and iterates over the replay buffer for each inner
epoch. [Actor dispatch](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.4/openrlhf/trainer/ray/launcher.py#L283-L327)
partitions experience batches among actor workers. The new divisibility checks
ensure complete equal shards and accumulation boundaries for the wrapper's
fixed synchronous settings. CPU tests independently enumerate actor shards,
epochs and accumulation boundaries for several layouts; they do not simulate
DeepSpeed update success.

Actual optimizer-step enforcement still needs an actor-side contract that can
stop at a verified completed update, account for gradient accumulation and
resume/overflow behavior, and return completed-update evidence through Ray.
OpenRLHF 0.8.4's stock CLI has no such stop option; its controller's scheduler
estimate and per-rollout logs do not provide that contract. Therefore `max_steps`
remains rejected, including through the real unified-launcher-to-native path.
Implementing a scheduler-count substitution or killing a subprocess after a
log row would not resolve this dependency. A backend-aware adapter plus native
execution validation remains required before claiming canonical step parity.

The frozen thesis, campaign/sweep records, SD reconciliation and strict gates
are unchanged. This fix adds no scientific result. Existing [manuscript errata](MANUSCRIPT_ERRATA.md)
and [review limits](RUNTIME_REVIEW.md) remain applicable.

## New validation for this fix

- Fresh full suite: **1,318 passed, one intentional skip, three warnings,
  84.51% configured coverage**, 92.63 seconds. This adds 12 tests beyond the
  previous continuation's 1,306-test run; neither prior run is reused here.
- Focused native budget/command/metrics/launcher checks: **68 passed**.
- A before/after command-only reproduction confirmed that `da00ceac2` accepted
  the 64-versus-1024 layout and the new preflight rejects it. No training ran.
- Repository Ruff/policy/format, explicit new-module Ruff checks and configured
  20-file mypy passed. All nine aggregate audits and 237 public checks passed.
- All 87 frozen October 11 manifest byte counts/hashes match. Signed campaign,
  signed sweep, strict verifier and SD reconciliation bytes remain unchanged.
- Fresh campaign/sweep strict checks still exit 1 at 95.7% and 82.6%.
  The exact October 11 PDF review check still fails for missing independent
  passing review. No gate was weakened or receipt upgraded.

The integration checkout's pre-existing untracked `.coverage` was preserved;
this run used a separate coverage file. The original dirty developer checkout
was not changed. No GPU/model execution, paid work, credentials, external
transfer, push or institutional submission occurred. Backend step enforcement,
scientific evidence gaps and independent thesis review remain open.

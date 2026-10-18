# Validated engineering handoff — 2026-10-12

## Source and scope

Validated implementation checkpoint: `8ecfd8fcee8cb5278a2e034d4c9401e2ca71a39c`, published on `fix/provenance-openrlhf-20261011` in [PR 17](https://github.com/arvindcr4/tinker-rl-lab/pull/17). This document records an engineering snapshot; it does not revise frozen thesis artifacts or establish new benchmark results.

Start with [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) for supported commands and backend limits. With the locked dev environment installed, the short offline path is:

```sh
.venv/bin/python tools/run_offline_harness.py --out /tmp/tinker-offline-handoff-001.json
```

Choose a fresh output path. Expected result is five passing CPU checks, zero skips and a local evidence receipt. Random tiny models and in-memory service doubles exercise configuration inspection, training, independent GSPO updates, actor/Adam and critic resume, final-policy held-out scoring and completed-cache replay. The learning checks also recover from a deliberately lost optimizer response after an update was applied. These are correctness oracles, not estimates of real-model improvement.

## Verified results

- Standard locked dev environment: `make check submission-check` passed all stages, with **2,089 passed, 2 skipped**, **87.92%** supported-surface coverage
- Separate frozen-lock Python 3.12 environment with `dev` and the TensorBoard-only `cleanrl` extra: the same aggregate passed with **2,096 passed, 1 skipped**, **87.92%** coverage
- The sole skip in that second environment is the non-CPU critic-device check; this VM has no supported non-CPU device. The optional native TensorBoard module accounts for the other skip in the standard environment
- All **147 installed packages** in the separate environment passed dependency consistency checks; an offline frozen-lock dry run reported no changes. The main environment and project/lockfile bytes were preserved
- A fresh **31-file committed source subset** passed **54 workflow, comparability, cancellation, locking and real TensorBoard tests**. All **22 loaded project modules** resolved inside that subset, and source hashes stayed unchanged. Existing separately locked dependencies were reused; this is source-isolation evidence, not a hermetic-install claim
- All **15 engineering CI checks** passed at the implementation checkpoint. The separate scientific `minreport-audit` remains failed

Chronological commands, changes, failed prototypes and final validation counts are summarized in [VM_VALIDATION.md](VM_VALIDATION.md).

## Operational boundaries

Active local runs sharing one seed-checkpoint path are excluded before tracking/provider setup. Lock-marker existence is not ownership; do not delete the marker to bypass an active run. Completed replay uses a committed snapshot without rewriting its checkpoint. Ctrl+C attempts failed tracking finalization, releases local ownership and preserves the original interrupt, including when cleanup fails. Abrupt termination can still prevent cleanup.

Recovery tests concern committed process-level checkpoints. They do not certify power-loss filesystem durability, distributed locking, remote-job cancellation or a live provider's actor/optimizer persistence contract. A locally interrupted command is not evidence that its remote work stopped.

Canonical Tinker/SkyRL execution remains guarded where required semantics are not implemented. Direct baseline execution has a separate, explicitly limited contract. See [UNIFIED_EXECUTION_LIMITS.md](UNIFIED_EXECUTION_LIMITS.md).

## Scientific work still required

The current base/trained receipt audit checks recorded arithmetic and prompt/sampling metadata, but matching v1 receipts still return **UNKNOWN, exit 3** because target contents and scorer implementation are not bound. Matching configuration does not establish weight identity, decontamination, causal improvement, representativeness or significance.

Scorer-default/regrading policy, sampler-refresh cadence and the multi-call tool adapter remain unresolved decisions. Live-provider validation and adequately controlled multi-seed evaluation remain separate work. Existing scores and historical artifacts are unchanged; engineering passes do not clear the scientific audit or independent-review requirements.

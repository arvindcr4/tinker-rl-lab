# Flagship conformance pilot control plane

This directory turns `pilot_preregistration.json` into an exact 24-unit
screening matrix without allocating external compute.  It is intentionally
separate from every frozen E1 result, W&B run, Hugging Face repository, and
Colab session namespace.

Generate the audited dry-run manifest with:

```bash
PYTHONPATH=zvf-program/flagship python -m pilot.plan_screening --write
```

The current preregistration is `ready_to_run` with authorization limited to the
staged screening DAG. Generated plans permit only the frozen A100 smoke, six
immutable corpus jobs, and the 24 scientific units that become eligible after
their matching corpus is independently accepted. Confirmatory jobs remain
forbidden. The control plane fails if a bound parent/S1/theory hash changes,
the matrix is not exactly 24 units, screening and confirmation seeds overlap,
the accelerator is not A100, or any pilot identity overlaps the frozen E1
namespace.

Each plan also carries an explicit readiness report. The numeric execution
contract resolves the fixed-step/token-matching question with one shared,
immutable 100-group corpus per regime/seed and conservative identical charging
across its four conditions. The remote trainer, token/FLOP ledger, checkpoint
recovery, verifier, and go/kill analysis passed the recorded offline gate before
GPU authorization changed.

## Current execution state (2026-07-21)

The non-scientific A100 smoke is independently accepted. No scientific unit
has run. All six corpus jobs are terminal `failed_infrastructure` after three
guarded launcher attempts, and all 24 units remain `pending` because no corpus
has an acceptance receipt. The latest post-reload allocation wave failed before
VM creation for every corpus with Colab's exact error:

```text
TooManyAssignmentsError: Failed to issue request POST ...
variant=GPU&accelerator=A100: Precondition Failed
```

At the same time, `colab sessions` and `colab status` both report no active
sessions, and no flagship supervisor, launcher, or remote-training process is
running locally. The result JSON files under `launch/results/` are preserved
launcher outputs, not corpus acceptance receipts; the authoritative state is
`launch/supervisor_state.json` plus `launch/acceptance/`, which contains only
the accepted smoke. Do not reset corpus status or allocate again until Colab's
assignment state changes. This provider failure is infrastructure evidence, not
a scientific observation and not permission to substitute an accelerator.

**Capacity-restoration snapshot (2026-07-21 11:20 IST).** After the user
confirmed renewed Colab access, only `corpus__balanced_equal_length__s11` was
reset from infrastructure-terminal state. Its first guarded attempt obtained
an A100, passed the exact Python/package/accelerator check, and started W&B run
`ujryg527`. The first profiled group completed with 4,096 charged tokens, and
the unprofiled groups began advancing. The other five corpus jobs remain
terminal, all 24 scientific units remain dependency-gated, and no corpus is
accepted until its finished W&B run, private HF dataset commit, immutable
manifest, token/FLOP ledger, and local independent verification all pass.

`replay.py` freezes the first causal control: each regime/seed produces one
content-addressed replay corpus consumed in the same order by all four
conditions.  The balanced control keeps all eight rows and makes optimization
lengths equal by active EOS padding.  The variable-length regime chooses the
lexicographically first six-row subset with maximal population length CV and
fails closed unless that CV is at least 0.35.  Generated tokens from rejected
groups remain charged in the ledger.

# Remaining obligations and closure ledger

Date: 2026-07-19
Scope: 18 canonical manuscripts, their included source closure, companion
registry/audit tooling, and the new GRPO/PPO/SAO synthesis.

## Outcome of this pass

The local, non-fabricable obligations are closed:

- R05's enumerated proof gaps were discharged; the paper now states explicit
  assumptions, an exact Clopper--Pearson lower bound, and a rigorous proxy-
  optimum proof rather than proof sketches.
- R06--R08 no longer contain TODO result cells, author placeholders, or claims
  that planned adapters/runners are already released.
- The core audit has a frozen machine contract and a fail-closed aggregator;
  the M-GRPO agentic stratum has its own frozen contract.
- PPO/SAO PAM, GSR, EGM, gates, root aggregation, trace validation, tests, and
  a machine-readable confirmatory contract are implemented. No GLM-5.2 result
  is inferred from the arithmetic smoke-test implementation.
- Registry queries no longer crash on amendment sidecars or audit stale hard-
  coded checkouts. All 46 queryable records pass schema validation; delta-field
  drift is 0/33; measured sources resolve; strict validation reports zero
  errors and zero warnings (61 transparent information-level unknown blocks).
- The codebase graph service is configured and live. The indexed checkout has
  36,833 nodes and 63,906 edges and returns structural search results.
- All checked-in figure assets are now reached by the canonical papers. The
  rendered corpus has zero active figure fallbacks.
- All 18 PDFs are current, with zero unresolved citations, missing TeX inputs,
  or duplicate active labels across 328 unique included source files.

## Colab execution update (through 2026-07-19)

The Colab CLI was repaired and a T4-backed, resumable pilot campaign was run.
GRPO, Dr.GRPO, DAPO, and Adaptive-G seed 11 each completed with step-level W&B
logging and a private Hugging Face final checkpoint. Local completion is gated
on querying both remote records. Exact run links, HF commits, logs, manifests,
and the evidence boundary are recorded in
`zvf-program/audit/COLAB_EXECUTION_STATUS.md`.

These runs are explicitly `pilot-not-confirmatory`: Qwen2.5-0.5B synthetic
addition is not the frozen Qwen3-8B/GSM8K five-arm audit. E2--E7 below remain
open independently of E1; E1 is evidence-complete after its final
integrity-only evaluation repair.

After Colab A100 credits became available, all forty frozen E1 units completed:
all eight GRPO seeds (11, 23, 37, 53, 71, 89, 107, and 131), all eight DAPO
seeds, all eight GSPO seeds, all eight Dr.GRPO seeds, and all eight AERO seeds
on Qwen3-8B/GSM8K. Each used 30 optimizer steps
and the locked 500-example held-out set; the GRPO and GSPO units sampled 480
training completions each, while DAPO's frozen dynamic-sampling treatment
realized 1,824, 2,112, 1,472, 1,664, 1,648, 1,584, 1,840, and 1,728 rollouts. Their W&B runs are
finished, and each private HF repository contains full Trainer
checkpoints at steps 5, 10, 15, 20, 25, and 30 plus the final adapter and
manifest. A pre-hardening aggregate accepted all forty records, but the final
hash-integrity audit reopened six legacy evaluation records. Exact
checkpoint-30 evaluation-only repair has now closed all six. The hardened
campaign verifier accepts 40/40 with zero errors, and the frozen aggregate
reports `COMPLETE`: DAPO `DISAPPEARS`; GSPO, Dr.GRPO, and AERO are
`INCONCLUSIVE`.

The exact DAPO, GSPO, Dr.GRPO, and AERO treatment paths are now implemented and
locally tested. Each passed a one-step A100 preflight with a finished W&B run
and private HF checkpoint. The persistent campaign supervisor supports a
fail-closed concurrency limit of three and accepts only independently verified
local, W&B, HF-checkpoint, stack, treatment, and held-out evidence. The
authenticated Hugging Face account API independently confirmed the user's PRO
upgrade before the final campaign wave. AERO seeds 11, 23, and 37 completed and
passed strict local/W&B/HF reconciliation. GSPO seed 23 also completed and
passed strict reconciliation. GSPO seed 37 subsequently completed too; GSPO
seed 53 subsequently completed the GSPO arm. Dr.GRPO seed 37 subsequently
completed and passed strict reconciliation. AERO seed 71 subsequently passed
strict reconciliation; AERO seeds 53 and 89 subsequently passed strict
reconciliation. AERO seed 131, DAPO seed 37, and Dr.GRPO seed 23 then
completed and passed every final gate. The supervisor exited cleanly and all
Colab sessions were released.

The prior three A100 sessions were reclaimed simultaneously. Their durable HF
state is preserved (seed 89: checkpoint 30 and 288 evaluation rows; seed 107:
checkpoint 30 and 256 rows; seed 131: checkpoint 25), but Colab then rejected
replacement A100 assignments for insufficient quota or entitlement. The
hardened supervisor was paused under a global assignment cooldown and was
configured for evaluation-only recovery of seeds 89/107 and exact-source resume
of seed 131 when A100 allocation becomes available again.

The first guarded retry at 2026-07-15 15:45 UTC selected all three correct
recovery modes but Colab rejected each A100 assignment with HTTP 400 before VM
creation. Live checks confirmed the current OAuth2 identity
(`arvindcr4@gmail.com`), latest Colab CLI version (0.6.0), and a working
assignment-list API, ruling out stale authentication or a local runner defect.
The frozen A100 amendment and exact-source resume contract prohibit silently
substituting a lower-tier accelerator, so E1 could not advance until Colab
restored A100 quota/entitlement for this account.

A second guarded attempt at 2026-07-15 16:00 UTC confirmed the same external
A100 rejection and revealed a local retry-classification edge case. Failed
evaluation wrappers now resolve recursively to their immutable confirmatory
source requests, failures before Colab VM allocation are refunded instead of
exhausting a scientific unit's retry budget, and final-attempt children remain
active through retry-limit evaluation. The hardened 39-test suite and lint pass;
the supervisor was restarted with all three quota-only attempts at zero and the
preserved HF recovery state unchanged.

A live 2026-07-15 16:15 UTC retry verified the repair end to end: the two
checkpoint-30 units retained evaluation-recovery classification, seed 131
retained exact-source resume, and the repeated A100 assignment failures were
credited back so every attempt counter remains zero under the next cooldown.

A100 allocation resumed on 2026-07-16. GRPO seed 89 completed its exact
evaluation recovery at 310/500 (`0.6200`) and passed an independent audit of
its finished W&B run, private HF final commit, six complete checkpoint trees,
frozen fingerprints, final adapter, and contiguous 500-row trace. The aggregate
therefore advanced to 6/40 with 34 missing units and zero errors, while still
refusing all arm-level verdicts. The supervisor immediately backfilled the
released A100 with GRPO seed 107 evaluation recovery; DAPO seed 11 continues on
the other available A100 and has independently verified step-5 and step-10
private-HF checkpoints.

GRPO seed 107 subsequently completed its exact evaluation recovery at 317/500
(`0.6340`). A fresh independent audit matched its 500-row local and remote
manifests, frozen fingerprints, finished W&B summary, private HF final commit,
final adapter, and six complete checkpoint trees. The frozen aggregate advanced
to 7/40 with 33 missing units and zero errors. The supervisor released that
A100 and immediately backfilled it with GRPO seed 131 through exact-source
checkpoint resume. DAPO seed 11 reached step 21/30; its step-15 and step-20
private-HF checkpoints also independently contain all mandatory resume files.

When a third A100 slot became available, DAPO seeds 23 and 37 encountered
setup-only Colab WebSocket losses before training or W&B initialization. Their
idle sessions were released and both attempts were credited back to zero. The
launcher now treats the exact transport error as provider-transient and wraps
Colab's own execution timeout with a local watchdog, so a dead CLI child cannot
hold campaign capacity indefinitely. The focused 33-test suite, Ruff,
byte-compilation, and diff checks pass. DAPO seed 53 then launched cleanly into
the recovered slot while GRPO seed 131 began evaluation from an independently
verified checkpoint 30. DAPO seed 11 advanced to step 26/30 and its private
step-25 checkpoint independently contains all six mandatory resume files. DAPO
seed 53 reached step 6/30 and its first private step-5 checkpoint independently
contains the same complete resume set; GRPO seed 131 reached 216/500 held-out
rows with its evaluation progress persisted to HF.

GRPO seed 131 subsequently completed exact-source checkpoint recovery and the
full held-out evaluation at 311/500 (`0.6220`). Independent checks matched its
500-row local and remote manifests byte for byte, verified a contiguous trace,
the frozen stack and unit fingerprints, finished W&B run `6dd77cf1`, exact
private HF commit `7786d5e9452729bed8d029bc2e69cfe0f89e1d06`, final adapter,
and all six complete checkpoint trees. The aggregate advanced to 8/40 with 32
missing units and zero errors. Its idle Colab session was released after a
post-result CLI cleanup anomaly, and the supervisor immediately backfilled the
third A100 with DAPO seed 23. DAPO seed 11 is evaluating from verified
checkpoint 30; DAPO seed 53 continues from verified checkpoints 5 and 10.

DAPO seed 11 subsequently completed its full held-out evaluation at 324/500
(`0.6480`). Independent checks matched its 500-row local and remote manifests
byte for byte, verified contiguous indices, the frozen stack and DAPO treatment
fingerprints, finished W&B run `e629158e`, exact private HF commit
`33f158dfabf2ab0056cc213eb9c43755f5941eef`, final adapter, and all six
complete 11-file checkpoint trees. The local wrapper completed with return code
zero and no failed step. The aggregate advanced to 9/40 with 31 missing units,
zero errors, and no verdict. Its idle A100 was explicitly released, and the
supervisor backfilled the slot with DAPO seed 37. That setup lost its kernel
WebSocket before training or W&B initialization; no VM or scientific state
remains, the watchdog terminated the stale transport, and the supervisor
credited the attempt back to zero under a dependency cooldown. DAPO seed 53
continues beyond verified checkpoint 25 and DAPO seed 23 continues beyond
verified checkpoint 10 on the two healthy A100 sessions.

When DAPO seed 37 retried after cooldown, a fresh A100 again lost its kernel
WebSocket during setup before W&B or training initialization. This third
transport-only failure left no VM or scientific state and was credited back to
zero. The supervisor now rotates eligible untouched units ahead of credited
retries, then selects the least-recently attempted retry, so one provider-flaky
unit cannot indefinitely pin campaign capacity. All 20 focused supervisor
tests, Ruff, byte-compilation, and diff checks pass; the supervisor was
gracefully reloaded without interrupting the healthy DAPO seed-53 and seed-23
children. DAPO seed 53 completed step 30, its full private-HF resume checkpoint
was independently verified, and it entered held-out evaluation while DAPO seed
23 continued beyond verified checkpoint 10.

After the cooldown, the reloaded supervisor exercised the rotation policy live
and selected untouched DAPO seed 71 rather than retrying DAPO seed 37 a fourth
consecutive time. Its A100 passed package installation and the frozen environment
check, initialized running W&B run `98b3f510`, and created its private HF
repository. It then reached independently verified private checkpoint 10 at exact
HF commit `ff2d368842192d5da2c35d9657f4e7d71982e798`, with all six resume-critical
artifacts and trainer global step 10. DAPO seed 53's held-out evaluation advanced through independently
verified private commit `bf33bcb7d418c6b2855e602d2799690bf5d449a9`
at 480/500 rows, with 301 correct (62.7%). DAPO seed 23 reached independently
verified private checkpoint 20 at exact HF commit
`c56ec2adeafc6fe62b2a197fa77ebcf92a348603`, with all six resume-critical
artifacts and trainer global step 20.

DAPO seed 53 then completed its held-out evaluation at 318/500 (`0.6360`).
Independent checks matched the remote and local manifests byte for byte,
verified 500 contiguous indices, the frozen stack and DAPO treatment
fingerprints, finished W&B run `a4c73e0e`, exact private HF commit
`194c9d2ea0d72cb93c751a314bbec9cbdcf7b5c0`, final adapter, and all six
complete 11-file checkpoint trees. The wrapper recorded return code zero and
no failed step. The aggregate advanced to 10/40 with 30 missing units, zero
errors, and no verdict. The post-completion CLI standard-stream error did not
alter any accepted artifact; the idle A100 was explicitly released so the
supervisor can backfill the third slot while DAPO seeds 23 and 71 continue.

Colab then reclaimed the seed-23 and seed-71 VMs while their local six-hour
CLI transports remained alive. Their durable private-HF checkpoints at steps
20 and 10 were independently preserved. Once Colab capacity was restored, the
stale transports were terminated and exact-source recovery allocated two fresh
A100s, passed the frozen environment check, and confirmed resume from those
precise checkpoints. DAPO seed 89 simultaneously obtained the third A100 and
passed the same environment check. The supervisor now reaps an active CLI
transport only when its exact named session is absent from three consecutive
successful server probes and no unnamed assignment is still resolving; 24
focused tests, byte-compilation, Ruff, and diff checks pass. The campaign is
again at the three-session limit, and its heartbeat explicitly notifies this
thread whenever Colab rejects an allocation for exhausted credits, quota, or
accelerator entitlement.
The recovered seed-23 run then completed all 30 steps and the 500-row held-out
evaluation with 312 correct answers (`0.6240`). Independent reconciliation
confirmed finished W&B run `c0d53921`, exact private HF commit
`08d3a34da45050b1d4787de45e5d493555fb464c`, all six complete checkpoint trees,
the final adapter, frozen fingerprints, and a contiguous remote manifest
byte-identical to the accepted local manifest. A stale post-result Colab stream
was closed only after a strict remote-only reconciliation mode rebuilt the
local record from those immutable artifacts without a new GPU allocation. The
idle A100 was released, the aggregate advanced to 11/40 with zero errors, and
the supervisor backfilled the slot with DAPO seed 107. Seed 71 reached
independently verified private checkpoint 30 at exact HF commit
`e6b10e441d945f82b2c36c20977242e6f0e520a2`; all six 11-file checkpoint
trees contain the resume-critical artifacts and report their expected trainer
global steps. Its held-out evaluation reached independently verified private
progress commit `e3463574664ba7af2b045f9e6a6483da3cfdba24` at 464/500 rows,
with 464 contiguous indices, 288 correct, checkpoint step 30, held-out size 500,
and the frozen unit fingerprint. Fresh seed 89 reached independently verified
private checkpoint 30 and progressed through 480/500 evaluation rows at exact HF commit
`d1c3a9d74c669f8a4d0fd072cd7fcd1a356bd155`; all six checkpoint trees contain
11 files and every resume-critical artifact, its evaluation indices are
contiguous with 300 correct, and the frozen unit and stack fingerprints remain
intact. Seed 71 then completed all 500 held-out rows with 317 correct (`0.6340`).
Independent reconciliation verified finished W&B run `98b3f510`, exact private
HF commit `59406ac609a1c7539b68891f1bb639f8e8893dc9`, all six complete 11-file
checkpoint trees and trainer steps, the final adapter, the identical contiguous
500-row trace, and both frozen fingerprints. Its A100 was released, the
aggregate advanced to 12/40 with 28 missing units and zero errors, and the
supervisor backfilled DAPO seed 131 while seeds 89 and 107 continued. The focused
suite now passes 42 tests. DAPO seed 107 subsequently reached exact private HF
commit `cefd62d418d685b468441c99b7301df3c3963925`; its 11-file checkpoint-5,
checkpoint-10, and checkpoint-15 trees contain the adapter, optimizer,
scheduler, RNG state, trainer states at their exact global steps, and every
other resume-critical artifact, while W&B run `f56135ec` remains live beyond
step 15. DAPO seed 131 likewise reached exact private HF commit
`06d3e0951e1e37423ab0035a083e96de85159672`; its 11-file checkpoint-5 and
checkpoint-10 trees contain the same resume-critical state, its latest trainer
state reports exactly global step 10, and W&B run `95076dd5` remains live.
Byte-compilation, Ruff, and diff checks pass. Seed 89 then completed all 500 held-out rows with 318 correct
(`0.6360`). Independent reconciliation verified finished W&B run `c0e809d5`,
exact private HF commit `c290319779e45c6ca690d17fd666faa7a8e73813`, all six
complete 11-file checkpoint trees and exact trainer steps, the final adapter,
identical contiguous progress and manifest traces, and both frozen
fingerprints. The idle A100 and stale transport were released, the aggregate
advanced to 13/40 with 27 missing units and zero errors, and the supervisor
backfilled GSPO seed 11 to restore three active A100s. GSPO seed 11 subsequently
reached exact private-HF commit `d44a159ea7510c97887b5905a62e1da8eaa18ddd`;
its checkpoint-5 and checkpoint-10 trees each contain all 11 required resume
files, and its latest trainer state reports exactly global step 10.

At 2026-07-17 04:17 UTC, host inspection established that the Mac had rebooted
at 04:14:42 UTC, terminating the local tmux supervisor and runner transports.
This was a host reboot rather than a reported Colab credit or quota failure.
W&B marked DAPO seeds 107 and 131 and GSPO seed 11 crashed at global steps 19,
14, and 19, respectively. Their last complete private-HF recovery points remain
DAPO-107 checkpoint 15 at `cefd62d418d685b468441c99b7301df3c3963925`,
DAPO-131 checkpoint 10 at `06d3e0951e1e37423ab0035a083e96de85159672`,
and GSPO-11 checkpoint 15 at
`7ba66f44a5060e8f125caab76656c5c2cbcaf33d`; every tree contains all 11
resume files and the latest trainer states report their exact steps. Three
exact-source resume sessions were launched, all three received A100s and passed
the frozen environment check, and the persistent supervisor was restored with
three active and three remote assignments. The process scanner was also
hardened to ignore a tmux server process title that embeds its first window's
runner command, preventing a phantom fourth slot after recovery; the focused
suite remains 42/42 with Ruff and diff checks clean. GSPO seed 11 then completed
training at exact private-HF commit
`29364775fc310e75ad821a2ed8745b90f0aa4793`; all six checkpoint trees contain
the 11 required resume files, the checkpoint-30 trainer state reports exactly
global step 30, W&B remains live at step 30, and the 500-row held-out evaluation
reached 104/500 rows; exact private-HF progress commit
`279201f3eabe201f3ecca7189117f7eed63279f6` independently contains 96
contiguous traces covering indices 0--95 with 59 correct. DAPO seed 131
reached exact private-HF commit `4f5e8ca6c78cfcb758b58b16915359ebd8d31c8e`;
its checkpoint-15 tree contains all 11 resume files and its trainer state reports
exactly global step 15. DAPO seed 107 reached exact private-HF commit
`ef001e4caed28e3dd01cf1462cbb62e66f46eb20`; its checkpoint-20 tree contains
all 11 resume files, its trainer state reports exactly global step 20, and W&B
confirms step 20.

At 2026-07-17 05:18 UTC, Colab simultaneously removed all three remote
assignments while their local CLI transports remained alive. The supervisor's
three-probe guard confirmed the exact session names were absent before closing
only those stale transports. Three replacement probes at 05:20 UTC then failed
before VM creation with Colab's explicit A100 quota-or-entitlement rejection;
all were credited back and no scientific attempt was consumed. Private-HF
recovery remains independently verified at DAPO-107 checkpoint 20, DAPO-131
checkpoint 15, and GSPO-11 checkpoint 30 with exact evaluation progress through
row 112 at commit `19b2887e44b5fa4e38c75db539165274f7dc3863`. The hardened
selector now gives Hub-proven evaluation recovery first priority, followed by
partial-checkpoint recovery, untouched units, and ordinary retries. A live
selection check against the current Hub state therefore chooses GSPO-11,
DAPO-107, and DAPO-131 for the next three A100s. The 05:35 UTC guarded retry
confirmed this ordering live: evaluation recovery plus both exact-source
checkpoint resumes were selected, all three A100 allocations received the same
pre-VM quota-or-entitlement rejection, and all three attempts were refunded
before the next cooldown. The focused suite passes 44/44, with
byte-compilation, Ruff, and diff checks clean. The active 15-minute heartbeat
notifies this task on the present quota/entitlement condition and when
allocation returns.

At 2026-07-17 07:28 UTC, reloaded Colab credits restored A100 allocation. An
exact GSPO-11 recovery probe bypassed only the stale cooldown, obtained an A100,
and was joined by DAPO-107 and DAPO-131 on the other two slots. Every session
passed the frozen NVIDIA A100-SXM4-40GB environment check and exact package
pins. Private-HF state restored exactly at GSPO-11 checkpoint 30 and row 112,
DAPO-107 checkpoint 20, and DAPO-131 checkpoint 15. The persistent supervisor
independently counts three local runners, three named remote sessions, and
three occupied slots, so automatic backfill remains fail-closed while these
recoveries run.

At 2026-07-17 08:51 UTC, GSPO seed 11 completed exact evaluation recovery with
320/500 correct (`0.6400`). Independent checks confirmed contiguous indices,
matching frozen fingerprints, finished W&B run `5f4fc0d3`, exact private HF
commit `37a2793c2138940f1ece2950dd56df3e1cdf7ccc`, the final adapter, and six
complete 11-file checkpoint trees whose trainer states report exact steps
5/10/15/20/25/30. The accepted remote and local 500-row manifests are
byte-identical. The aggregate advanced to 14/40 with 26 missing units, zero
errors, and no verdict; the idle A100 was released and immediately backfilled
with GSPO seed 71 while DAPO seeds 107 and 131 continue.

At 2026-07-17 10:22 UTC, DAPO seed 107 completed exact-source recovery with
317/500 correct (`0.6340`). Independent checks confirmed contiguous indices,
matching frozen unit/stack/treatment fingerprints, finished W&B run `f56135ec`,
exact private HF commit `6e60484cfd40dd1d6b17ee92b18af93e12f7a011`,
the final adapter, and six complete 11-file checkpoint trees at exact global
steps 5/10/15/20/25/30. The accepted local and remote manifests are byte-identical.
The aggregate advanced to 15/40 with 25 missing units, zero errors, and no
verdict; its released A100 was immediately backfilled with GSPO seed 89.
DAPO seed 131 and GSPO seed 71 were moved safely onto evaluation-recovery
sessions after stale Colab transports, resuming from verified HF prefixes of
304 and 80 rows respectively.

At 2026-07-17 11:05 UTC, DAPO seed 131 completed evaluation recovery with
316/500 correct (`0.6320`). Independent checks verified finished W&B run
`95076dd5`, exact private HF provenance, six complete 11-file checkpoint trees
at steps 5/10/15/20/25/30, the final adapter, a contiguous 500-row trace, and
the frozen unit and stack fingerprints. Post-acceptance review found that the
evaluation-only finalizer hardcoded 480 rollouts rather than reading DAPO's
checkpoint telemetry. The finalizer now derives dynamic-arm rollouts
fail-closed and preserves the frozen treatment configuration. The corrected
local/remote manifest is byte-identical at private HF commit
`f34a67a1348a1f556e0f1ba78c2812cadf06e5ed`, records the telemetry-backed 1,728
rollouts and an explicit correction receipt, and matches the corrected finished
W&B summary. The aggregate advanced to 16/40 with 24 missing units and zero
errors; the released A100 was immediately backfilled with GSPO seed 107.

At 2026-07-17 11:44 UTC, GSPO seed 71 completed evaluation recovery with
319/500 correct (`0.6380`). Independent checks verified finished W&B run
`10ef44ab`, private HF commit `d3b74ccd365ba40f467c24e68b64bc597eb6746c`,
six complete 11-file checkpoint trees at exact trainer steps
5/10/15/20/25/30, the final adapter, a contiguous 500-row trace, and matching
frozen stack, unit, and treatment fingerprints. The older recovery finalizer
had omitted the treatment fields from `run_config`; a provenance-only
correction restored those fields and added an explicit receipt without changing
training, rollouts, or held-out evidence. The corrected local and remote
manifests are byte-identical and the finished W&B summary points to the corrected
commit. The aggregate advanced to 17/40 with 23 missing units and zero errors;
the released A100 was immediately backfilled with GSPO seed 131, restoring
three active sessions alongside GSPO seeds 89 and 107.

At 2026-07-17 13:18 UTC, GSPO seed 89 completed its full frozen run with
323/500 correct (`0.6460`). Independent checks confirmed finished W&B run
`3ec7b7f9`, exact private HF commit
`40695b92925a2a11cf2a63fe6bfa3b65715004c6`, six complete 11-file checkpoint
trees at trainer steps 5/10/15/20/25/30, the final adapter, matching frozen
stack, unit, and GSPO treatment fingerprints, and a contiguous 500-row trace
with unique completion hashes. The accepted local and remote manifests are
byte-identical. The aggregate advanced to 18/40 with 22 missing units and zero
errors; the released A100 was immediately backfilled with Dr.GRPO seed 11,
restoring three active sessions alongside GSPO seeds 107 and 131.

At 2026-07-17 14:02 UTC, GSPO seed 107 completed its full frozen run with
320/500 correct (`0.6400`). Independent reconciliation confirmed finished W&B
run `879174ca`, exact private HF commit
`2b521a728689e87afc7ac027108a4e3725468820`, six complete 11-file checkpoint
trees at trainer steps 5/10/15/20/25/30, the final adapter, matching frozen
stack, unit, and treatment fingerprints, and a contiguous 500-row trace with
unique completion hashes. Local and remote manifests are byte-identical. The
aggregate advanced to 19/40 with 21 missing units and zero errors. The next
third-slot allocation for Dr.GRPO seed 23 was rejected before VM creation with
Colab's A100 quota-or-entitlement error, so it consumed no scientific attempt
and was credited back to attempt 0. GSPO seed 131 and Dr.GRPO seed 11 remain on
their allocated A100s with all durable checkpoints preserved while the
supervisor applies guarded allocation backoff.

At 2026-07-17 14:14 UTC, Colab reclaimed both remaining A100 sessions. The
supervisor required three missing-session polls before reaping either wrapper.
GSPO seed 131 preserved all six checkpoints and a verified 368-row evaluation
prefix at exact private HF commit
`b3d67a9710a3e031cadc2a97744851fae3edf21c` (221 correct, contiguous indices,
and 368 unique completion hashes). Dr.GRPO seed 11 preserved complete
checkpoints through exact global step 25 at private HF commit
`ce0166fdc14a3d3d555ebe4a10c2afce40fb5f53`. The 14:18 UTC guarded retry chose
evaluation-only recovery for GSPO-131, exact-source checkpoint resume for
Dr.GRPO-11, and a fresh Dr.GRPO-37 unit, proving that recovery routing remained
correct after simultaneous reclamation. Colab rejected all three A100
allocations before VM creation with its quota-or-entitlement error. Each
infrastructure failure was credited back, no scientific attempt was consumed,
and the supervisor remains live under allocation backoff through 14:33:47 UTC.

At 2026-07-17 14:34 UTC, reloaded Colab credits restored three A100 allocations
on the first normal probe after cooldown. The supervisor launched GSPO seed 131
in evaluation-only recovery, Dr.GRPO seed 11 in exact-source checkpoint resume,
and a fresh Dr.GRPO seed 53 unit. All three sessions passed the frozen A100,
CUDA 12.8, Torch 2.11.0+cu128, and six-package environment check. GSPO-131
restored checkpoint 30 plus its exact 368-row prefix, Dr.GRPO-11 restored
checkpoint 25, and Dr.GRPO-53 entered the fresh training path. Direct Colab and
campaign state both report three named remote sessions and three occupied
slots, re-establishing the proven concurrency ceiling without relaxing any
scientific invariant.

At 2026-07-17 14:42 UTC, independent GSPO-131 evidence inspection found that
the standalone checkpoint evaluator omitted `completion_sha256` from its newly
generated recovery rows. No such row was accepted. The session was stopped,
the evaluator was fixed to hash every completion and fail closed on malformed
final traces, and a tested rewind helper now records and removes only an
unverifiable suffix before deterministic replay. Rows 368--399 from source
commit `793083965c5b89d6d59cd85829466c194eb4f873` were rewound. Exact private HF
commit `8e3b43df25f842a02ecfb27d33c938a59a2025fe` restores the last valid boundary:
368 contiguous rows, 368 valid and unique hashes, and 221 recomputed correct
answers, with an explicit repair receipt. The focused suite passes 42/42 and
the complete E1 audit suite passes 50/50; Ruff, byte-compilation, and diff
checks are clean. GSPO-131 is queued to replay
from row 368 after its per-unit cooldown; the supervisor used the available
slot for fresh Dr.GRPO seed 71 while preserving the three-session ceiling.

At 2026-07-17 16:36 UTC, Dr.GRPO seed 11 completed exact-source checkpoint
resume and its full held-out evaluation with 321/500 correct (`0.6420`).
Independent reconciliation verified finished W&B run `52417d14`, exact private
HF commit `dbcd0740bc95d0500713dfbcd19e667cdd8555a3`, the final adapter, six
complete 11-file checkpoints at exact trainer steps 5/10/15/20/25/30, matching
frozen stack/unit/treatment fingerprints, and 500 contiguous rows with valid
unique completion hashes. The remote progress trace is identical to the final
manifest. The aggregate advanced to 20/40 with 20 missing units and zero
errors; the released A100 was immediately backfilled with GSPO seed 131
evaluation recovery from the repaired 368-row boundary, retaining three live
sessions alongside Dr.GRPO seeds 53 and 71.

At 2026-07-17 17:07 UTC, GSPO seed 131 completed deterministic evaluation
recovery from the repaired 368-row boundary with 314/500 correct (`0.6280`).
Independent reconciliation verified finished W&B run `726607de`, exact private
HF commit `5753775e1976d3441f8643fff2dc852a53e16da3`, six complete 11-file
checkpoints at exact trainer steps 5/10/15/20/25/30, the final adapter, matching
frozen stack/unit/treatment fingerprints, and a 500-row trace with contiguous
indices and valid unique hashes. The original 368-row prefix is byte-identical
and the local and remote manifests have the same SHA-256 digest. The aggregate
advanced to 21/40 with 19 missing units and zero errors; the released A100 was
immediately backfilled with fresh Dr.GRPO seed 89, retaining three live sessions
alongside Dr.GRPO seeds 53 and 71.

At 2026-07-17 17:26 UTC, Dr.GRPO seed 53 completed its fresh frozen run with
318/500 held-out answers correct (`0.6360`). Independent reconciliation
verified finished W&B run `1215e9cb`, exact private HF commit
`5bef04e01ba9345493e014b5ea301f295122d378`, the final adapter, six complete
11-file checkpoints at exact trainer steps 5/10/15/20/25/30, and matching
frozen stack, treatment-spec, and unit fingerprints. Its 500 indices are
contiguous, all completion hashes are valid and unique, and the local and
remote manifests have identical SHA-256 digests. The aggregate advanced to
22/40 with 18 missing units and zero errors; the released A100 was immediately
backfilled with fresh Dr.GRPO seed 107, retaining three live sessions alongside
Dr.GRPO seed 89 and seed 71 evaluation recovery.

At 2026-07-17 17:44 UTC, Dr.GRPO seed 71 completed evaluation recovery from its
exact checkpoint-30 source and original 240-row prefix with 310/500 held-out
answers correct (`0.6200`). Independent reconciliation verified finished W&B
run `918b9a8a`, exact private HF commit
`a8cd65ea26f03909ab84b2f5f3601b64ba0ae0c5`, the final adapter, six complete
11-file checkpoints at exact trainer steps 5/10/15/20/25/30, and matching
frozen stack, treatment-spec, source-unit, and recovery fingerprints. Its 500
indices are contiguous, all completion hashes are valid and unique, the
original 240-row trace prefix is byte-identical, and the local and remote
manifests have identical SHA-256 digests. The aggregate advanced to 23/40 with
17 missing units and zero errors; the released A100 was immediately backfilled
with fresh Dr.GRPO seed 131, retaining three live sessions alongside Dr.GRPO
seeds 89 and 107.

Between 2026-07-17 19:58 and 20:35 UTC, fresh Dr.GRPO seeds 89, 107, and 131
completed with 314/500 (`0.6280`), 314/500 (`0.6280`), and 323/500 (`0.6460`)
held-out answers correct. Independent reconciliation verified their finished
W&B runs, exact private HF final revisions, final adapters, six complete
11-file checkpoints per run at exact trainer steps 5/10/15/20/25/30, matching
frozen stack/treatment-spec/unit fingerprints, and contiguous 500-row traces
with valid unique completion hashes. Every local manifest is byte-identical to
its remote counterpart. The aggregate advanced to 26/40 with 14 missing units
and zero errors.

Fresh AERO seeds 11, 23, and 37 then each reached private checkpoint 5 before
their wrappers stopped. Their W&B runs are marked `crashed`; no held-out rows
exist. Exact-source recovery is currently blocked by Colab's explicit error:
`Backend rejected accelerator 'A100'. You may not have quota or entitlement for
this accelerator on your account.` These allocation failures are not counted
as scientific attempts. The missing persistent supervisor was restored at
2026-07-18 03:32 UTC, reproduced the same error, and returned to guarded global
backoff while preserving all three checkpoint-5 revisions.

Reloaded Colab credits later admitted exact-source A100 resumes for all three
AERO seeds. Each reached optimizer step 20, and private checkpoint 15 was
independently verified at commits `259f061364acb8586fbd2d2efb928bab0808b68f`,
`d8cdbef89af1726f2215653ff2ddfd5bb8aa99b6`, and
`3e8a87345d03cc643143ee0a3969ee4b581b42e8`. Hugging Face then rejected every
checkpoint-20 commit with `Private repository storage limit reached, please
upgrade your plan to increase your private storage limit`. No checkpoint-20
remote artifact exists; checkpoint 15 remains the recovery point. The
supervisor was paused and every Colab session released to avoid wasting compute.
An inventory measured 89,870,440,367 bytes across 37 private E1 repositories;
finishing the 14 missing private-HF units requires an HF storage-plan upgrade or
equivalent additional private storage capacity. The measured full-unit and
checkpoint-15 footprints project another 39,606,696,138 bytes (39.61 GB,
36.89 GiB) for the three partial and eleven fresh repositories, before safety
margin; deleting the 2.74 GB of older preflight artifacts would not suffice.

At 2026-07-18 08:10 UTC, the authenticated Hugging Face API reported
`isPro=true`. The persistent supervisor restarted all three exact-source AERO
checkpoint-15 recoveries. Seeds 11 and 23 received A100s, passed the frozen
environment check, and restored their private checkpoints. Seed 37 lost its
Colab connection during installation; this infrastructure failure was credited
back, did not consume a scientific attempt, and was scheduled for guarded retry
after 2026-07-18 08:27:32 UTC. The campaign remains 26/40 until strict full-unit
validation succeeds.

At 2026-07-18 08:27:49 UTC, seed 37 received a fresh A100, passed the frozen
environment check, and began exact-source checkpoint-15 restoration. The
supervisor now proves three local runners, three named remote A100 sessions,
and three occupied slots. Seeds 11 and 23 have advanced to steps 18 and 17.

At 2026-07-18 08:55 UTC, AERO seeds 11 and 23 crossed the first post-upgrade
private-storage gate. Their exact checkpoint-20 Hub commits are
`6d407834b53939792e00a9a7389e665854023805` and
`f8a44fa5b71d0360d7e5c6fc353dd18adb0cb42a`. Independent downloads verified
all 11 files in each tree and every resume-critical artifact; both
`trainer_state.json` files report `global_step=20` and `max_steps=30`.
Distinct adapter hashes confirm seed-specific artifacts. Seed 37 then committed
its complete checkpoint-20 tree at exact revision
`6f504b1e797263485243f4423c873e6110f9ed22`; an independent download verified
all six resume-critical files, `global_step=20`, `max_steps=30`, and a distinct
adapter SHA-256. All three AERO resumes therefore have durable private
checkpoint-20 recovery points and live W&B runs. This proves the HF PRO
capacity is usable, but does not count any unit complete before step 30, 500
held-out rows, W&B finalization, and strict remote reconciliation.

At 2026-07-18 09:41 UTC, AERO seed 11 privately committed a complete
checkpoint-25 tree at exact revision
`9c9192a194f55af9c41c0794d856e05f39e7d637`. The private repository now has
the five expected checkpoint boundaries through step 25 and 56 files total.
A fresh independent download verified all six resume-critical artifacts;
`trainer_state.json` proves `global_step=25` and `max_steps=30`, and the adapter
SHA-256 is
`adf36e6145fb45ba18e407c63ea455f1aa7919c4393c737257e909e25da5bc2f`.
Its W&B run remains live. Seeds 23 and 37 have advanced to completed steps 24
and 22 with live W&B telemetry. The frozen aggregate correctly remains 26/40
until full training, 500 held-out rows, W&B finalization, and strict remote
reconciliation finish.

At 2026-07-18 09:47 UTC, AERO seed 23 also privately committed checkpoint 25
at exact revision `588a35a656323bcbf7494e0807156ef0e6445907`. The private
repository has 56 files across checkpoints 5/10/15/20/25. A fresh independent
download verified all six resume-critical artifacts; trainer state is exactly
step 25 of 30 and the adapter SHA-256 is
`1ef6c121468aaffaef69f7c9ba860d345ec0da35fdf38466629836b805e24541`.
Its W&B run remains live, seed 11 has advanced to completed step 26, and seed
37 continues beyond its verified checkpoint-20 source. No partial unit enters
the aggregate, which remains 26/40.

At 2026-07-18 10:00 UTC, AERO seed 37 also reached an independently verified
private checkpoint 25 at exact revision
`1ec05bd995447e56d93594b65af76a0576f9599f`. The repository has 56 files
across checkpoints 5/10/15/20/25. A fresh download verified all six
resume-critical artifacts; trainer state reports step 25 of 30 and the adapter
SHA-256 is
`46da25bd0605cc314a9b858c048801891be3945fafa57a4d9a352d1d16d8274d`.
All three active AERO seeds now have independently audited checkpoint-25
recovery points, and their W&B runs remain live. Seed 11 has advanced to
completed step 28. The aggregate remains 26/40 pending checkpoint 30,
evaluation, W&B finalization, and strict reconciliation.

At 2026-07-18 10:24 UTC, AERO seed 23 completed training and committed its
required checkpoint-30 tree at exact private revision
`7c3f1493880c96b35d9a0577893daab28ccaafc7`. The repository now has all six
checkpoint boundaries 5/10/15/20/25/30 and 67 files. A fresh independent
download verified the six resume-critical artifacts; trainer state is exactly
step 30 of 30 and the adapter SHA-256 is
`75c062f1bd029f4f40da4180e2f4641173507db8068711d0a15f610f76f24839`.
Held-out evaluation has begun and committed durable progress 16/500 at
revision `c3aa51017639cbe198781720cc317aa18ef6c48c`. W&B is still live. Seed 23
therefore remains an open obligation until the 500-row record, final adapter
and manifest, finished W&B run, fingerprints, and strict reconciliation all
validate; the aggregate remains 26/40.

At 2026-07-18 10:28 UTC, AERO seed 11 also completed training and committed
checkpoint 30 at exact private revision
`ca0d9556d1d56b98edf51d6dd9dcc6f6f9143298`. An independent download verified
all six resume-critical artifacts, trainer state 30/30, and adapter SHA-256
`d34a2608a49cce40cdeb1c427823357ffcada6cd57a9dc38984c0f4a9f2123d6`.
Its held-out pass durably reached 16/500 at revision
`96cc983d98dfa88ff3fd59670a06b2cedd1e92ce`; seed 23 concurrently advanced to
32/500 at revision `04516d5770325de7afccb73ba4ed87c88345e327`. Both remain open
obligations until 500 rows and all final provenance gates validate. Seed 37 is
live at completed step 27, so the aggregate remains 26/40.

At 2026-07-18 10:52 UTC, AERO seed 37 also completed training and committed
checkpoint 30 at exact private revision
`12ac8fa85a62ba2febda7651ed1e278300c624c8`. A fresh independent download
verified all six resume-critical artifacts, trainer state 30/30, and adapter
SHA-256
`3e7a7c06e5d9bfdf5a67053aad616e80707a7a503bd21aa8de3e76b1fff757c1`.
Its evaluation durably reached 16/500 at revision
`d6f13ad86cfae44c1bd3337b272c999c2044ef30`; seeds 11 and 23 had concurrently
advanced to 112/500 and 128/500. All three active AERO obligations are now in
held-out evaluation, but remain open until 500 rows and every final provenance
gate validate. The aggregate remains 26/40.

At 2026-07-18 12:11--12:19 UTC, AERO seeds 11 and 23 completed their 500-row
held-out passes and final private manifests. Fresh audits verified checkpoints
5/10/15/20/25/30, exact stack and treatment fingerprints, contiguous traces,
finished W&B runs `0d4c16f9` and `3573a421`, and immutable Hub commits
`2affa816a79574e2c6e9b38915b2f58daf5fd9cf` and
`92123d35e401ff9c6c42494a383116cfbd91479e`. Seed 23's completed remote
manifest was reconciled into the required local record after its Colab
transport hung post-completion. The frozen aggregate now validates 28/40 with
12 missing units, zero errors, and no verdict. AERO seed 37 then reached a
durable 448/500 trace at private commit
`e6b584d4bcfbdada58c19b46efa0b063604fc280` before Colab reclaimed its
session. A fresh download verified trainer state 30/30, 448 contiguous rows,
and 280 correct predictions under the frozen unit fingerprint. An AERO seed-53
backfill was rejected before VM assignment with `Backend rejected
accelerator 'A100'. You may not have quota or entitlement for this accelerator
on your account.` The supervisor refunded the attempt and entered guarded
provider backoff with no active session and no lost checkpoint or evaluation
evidence. At the 2026-07-18 18:10 IST retry, the supervisor selected seed 37's
evaluation recovery and fresh AERO seeds 71 and 89. Colab rejected all three
A100 assignments before VM creation with the same quota-or-entitlement error;
all attempts were refunded. At 18:26 IST, a private-repository create, upload,
read-back, and cleanup probe independently proved that the Hugging Face upgrade
had restored private storage. The supervisor then launched seed 37 in exact
evaluation recovery and fresh AERO seeds 107 and 131, but Colab again rejected
all three A100 assignments before VM creation with the same error. The attempts
were refunded, seed 37 remains recoverable from checkpoint 30 plus 448/500
rows, and the next guarded retry was 18:41:13 IST. That retry selected seed 37
evaluation recovery plus fresh DAPO seed 37 and GSPO seed 23, but Colab again
rejected every A100 request before VM creation with the same error. All three
attempts were refunded, no remote session was created, and the next guarded
retry was 18:56:44 IST. That retry selected seed 37 evaluation recovery plus
fresh GSPO seeds 37 and 53, but Colab again rejected all three A100 assignments
before VM creation with the same error. Every attempt was refunded, and the
next guarded retry is 19:12:17 IST. A non-allocating OAuth diagnostic confirms
the verified CLI identity `arvindcr4@gmail.com` and a successful A100
eligibility GET, isolating the failure to Colab's provider-side POST allocation
gate.
At 19:12 IST, the next guarded retry selected seed 37 evaluation recovery plus
fresh Dr.GRPO seeds 23 and 37. Colab rejected all three A100 POST allocations
before VM creation with the same error; every attempt was refunded and the next
guarded retry is 19:27:28 IST.
At 19:27 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seeds 53 and 71. Colab again rejected all three POST allocations before
VM creation; every attempt was refunded and the next guarded retry is 19:43:09
IST.
At 19:43 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seeds 89 and 107. Colab again rejected all three A100 POST allocations
before VM creation with the same quota-or-entitlement error. No Colab session
was created, every pre-assignment attempt was refunded, and the next guarded
retry is 2026-07-18 19:58:50 IST.
At 19:58 IST, the next retry selected seed 37 evaluation recovery plus fresh
AERO seed 131 and DAPO seed 37. Colab again rejected all three A100 POST
allocations before VM creation with the same quota-or-entitlement error. No
Colab session was created, every pre-assignment attempt was refunded, and the
next guarded retry is 2026-07-18 20:14:21 IST.
At 20:14 IST, A100 capacity returned. Colab admitted all three requested
sessions: AERO seed 37 exact-source evaluation recovery, plus fresh GSPO seeds
23 and 37. The authoritative session list reports an A100 for each session;
all three completed environment verification on CUDA 12.8 with the frozen
training stack and entered their recovery or training paths. The supervisor is
again at the proven three-session ceiling and will backfill released capacity.
The resumed AERO seed-37 evaluator completed 500/500 rows, after which a
post-evaluation accounting check incorrectly required the fixed-arm 480-rollout
minimum. AERO's frozen treatment generates 12--16 real rollouts per step, and
the immutable checkpoint telemetry reported a valid 436 over 30 steps. The
arm-specific 360--480 bound and append-only campaign-log classifier were fixed
under 40 focused tests. A second finalization pass reused the complete remote
trace without regenerating examples. Independent verification accepted W&B run
`7547ba19`, private HF commit
`613d3f70cc0fb89eae5775a398d6f66136d14f49`, all six checkpoint trees, the
byte-identical local/remote manifest, frozen fingerprints, and 500 contiguous
hashed rows with 318 correct. The aggregate now validates 29/40 with 11 missing
units and zero errors; GSPO seed 53 immediately backfilled the released A100.

At 2026-07-18 17:36 UTC, GSPO seed 23 completed all 500 held-out examples and
passed an independent local/W&B/HF audit. Finished W&B run `1c16ac07` and
private HF commit `d1a6f1879bdee3eb41f65cdfc1ce34606641d828` agree on the
0.638 exact-match score, six checkpoint steps, frozen stack and GSPO treatment
fingerprints, and 480 training rollouts. The downloaded remote manifest is
byte-identical to the accepted local manifest and contains 500 contiguous
completion hashes with 319 correct predictions; its SHA-256 is
`2eff8815f458788ba09895d40012b6c9a0ff9877be8791f6649e8f216d88cea2`.
The aggregate advanced to 30/40 with 10 missing units, zero errors, and no
verdict.

At 2026-07-18 17:43 UTC, GSPO seed 37 also completed all 500 held-out examples
and passed independent reconciliation. Finished W&B run `60e5b791`, private HF
commit `a9c9ee88756d4c10e67925964f7b4a3662e354ad`, all six checkpoint trainer
states, and the frozen stack and GSPO treatment fingerprints agree. The
downloaded remote manifest is byte-identical to the local manifest and contains
500 contiguous completion hashes with 321 correct predictions; its SHA-256 is
`73ff2239a2ac7fe3f7618afca53e937be018222014162a91a6d4b5d2395204d9`.
The aggregate advanced to 31/40 with 9 missing units and zero errors. The
released A100 was backfilled with Dr.GRPO seed 37, restoring the three-session
ceiling alongside GSPO seed 53 and Dr.GRPO seed 23.

At 2026-07-18 17:58 UTC, GSPO seed 53 completed and passed independent
reconciliation, closing the full eight-seed GSPO arm. Finished W&B run
`ada5a2c2`, private HF commit
`6b58a60fb73d0ac1f34e77e52aac9083877fb7b7`, all six checkpoint trainer
states, and the frozen fingerprints agree. The byte-identical remote/local
manifest contains 500 contiguous completion hashes with 316 correct
predictions and SHA-256
`ad6de89244fde05c7b69975bf81ae65459c0580e8980674e82ec3b4b7729133f`.
The aggregate advanced to 32/40 with 8 missing units and zero errors.

At 2026-07-18 20:40 UTC, Dr.GRPO seed 37 completed and passed independent
reconciliation. Finished W&B run `5610e4f2`, private HF commit
`26e36e5e915f05922d9792014e2afecd18f6f364`, all six exact checkpoint
trainer states, and the frozen stack and treatment fingerprints agree. The
remote progress trace and remote/local manifests contain the same 500
contiguous completion hashes with 311 correct predictions (`0.6220`) and
canonical trace SHA-256
`cf82a9e27c1cfd322e6fe4dbd901d59cd2ac4b0d5d6bb4858fa1b3bffe7cca54`.
The aggregate advanced to 33/40 with 7 missing units and zero errors, and the
released A100 was immediately backfilled with AERO seed 89.

At 2026-07-18 21:44 UTC, Colab reclaimed the three active AERO sessions after
they disappeared from the authoritative session list for three consecutive
polls. The supervisor terminated only the stale transports and preserved exact
private recovery points: AERO-53 checkpoint 20 at
`7f044084d43b5632204032b39ed5d1856b930a2e`, AERO-71 checkpoint 25 at
`ce3c740552eb17980e3cdd504239407c80598ff4`, and AERO-89 checkpoint 5 at
`c45d78465df340f1ca910411b3e06bff4ecc964e`. Two guarded exact-source resume
waves were rejected before allocation with `Backend rejected accelerator
'A100'. You may not have quota or entitlement for this accelerator on your
account.` Pre-assignment retry credit was restored, no scientific attempt was
consumed, and no Colab session remains allocated. E1 stays at 33/40 until A100
capacity returns and a full unit passes the normal W&B/HF/held-out audit.

At 2026-07-19 04:21 UTC, reloaded Colab credits restored A100 allocation. A
successful capacity probe was released, then the three-slot supervisor restarted
in terminal session `7348`. AERO-53/71/89 all received A100-SXM4-40GB sessions,
passed the frozen stack check, downloaded their exact private checkpoints
20/25/5, and reconnected the original W&B run IDs. Reconstruction completed and
the resumed runs reached optimizer steps 21/26/6, proving exact-source
continuation; no partial run enters the aggregate before the normal
checkpoint-30 and 500-row held-out gates.

At 2026-07-19 04:53 UTC, AERO seed 53 durably advanced its private recovery
point to checkpoint 25 at exact HF commit
`50a57e24a8db17f93cc1f00b5ccbe4d040e3a210`. Independent verification proved
that the repository is private, the revision resolves exactly, and the
downloaded trainer state has `global_step=25`, `max_steps=30`, and intact AERO
metrics. Seeds 53, 71, and 89 are now computing steps 26, 29, and 9 from
authoritative private checkpoints 25/25/5; the aggregate correctly remains
33/40 until a complete unit passes every final evidence gate.

At 2026-07-19 05:06 UTC, AERO seed 71 completed training and independently
verified private checkpoint 30 at commit
`39deaf8e358edc7d19cf9548276d0253e5a77571`; its trainer state records
`global_step=30`, `max_steps=30`, intact AERO metrics, and held-out evaluation
has reached 48/500. Its exact private 48/500 snapshot at commit
`97c037dc58b55c33714c566e72c7a6a7b27feded` independently verifies contiguous
indices 0--47, valid completion hashes, 29/48 correct, and the frozen unit
fingerprint. AERO seed 89 independently verified private checkpoint 10
at commit `cf36173bcfa9373dcae7b7342757766132b861f8`, with
`global_step=10`, `max_steps=30`, and intact AERO metrics. The authoritative
private recovery points are therefore 25/30/10 for seeds 53/71/89. Seed 71
remains unaccepted until its 500-row evidence, final adapter/manifest, finished
W&B run, and all frozen fingerprints pass the normal audit.

At 2026-07-19 06:50 UTC, AERO seed 71 passed every final evidence gate. Its
finished W&B run `c73138a3` and exact private HF commit
`5168da9f98c37cd432e60b14443436daaaed6951` contain all six checkpoints, the
final adapter and manifest, and a contiguous, hash-valid 500-row trace with
316/500 correct (0.6320). Exact downloads verified every checkpoint's
`global_step`, `max_steps=30`, the frozen stack and treatment fingerprints, and
byte-identical local/remote manifests. The fail-closed aggregate therefore
advanced to 34/40 with zero errors and six units missing. AERO seed 53 has
completed training and its current exact private commit
`6a2d3e706d38d7dea596b67f88bce5d9de4da932` independently verifies 320/500
contiguous held-out rows, valid hashes, and 193 correct (0.6031). AERO seed 89
remains recoverable from verified checkpoint 20 at
`b001dd2357bde9d4927fdf11f48871330a42631a`. The released seed-71 A100 was
backfilled with AERO seed 107 under W&B `853facf3` and private repository
`arvindcr4/tinker-rl-lab-e1-aero-s107-853facf3`. The supervisor and all three
A100 sessions remain healthy; no partial unit is credited to the aggregate.

At 2026-07-19 07:36 UTC, AERO seed 53 passed every final evidence gate.
Finished W&B run `bd8d23f9` and exact private HF commit
`448707a7a1dce3f705204d90d06598a971cfbed2` contain all six checkpoints, the
final adapter and manifest, and 500 contiguous, hash-valid held-out rows with
316/500 correct (`0.6320`). Every trainer state has the expected
`global_step` and `max_steps=30`; the local and remote manifests are
byte-identical with SHA-256
`c011fab69d3277d9bdfd5073ce1247ebef73261ec1879498cf0340b20c9c7d87`.
The fail-closed aggregate advanced to 35/40 with five missing units, zero
errors, and no verdict. Its released A100 was backfilled with fresh AERO seed
131, so AERO seeds 89, 107, and 131 now occupy the three verified A100 slots.

At 2026-07-19 10:08 UTC, AERO seed 89 passed every final evidence gate.
Finished W&B run `201279c6` and exact private HF commit
`43421d47afa9a82453e883508a634f55ff770f14` contain all six checkpoints, the
final adapter and manifest, and 500 contiguous, hash-valid held-out rows with
315/500 correct (`0.6300`). All trainer states have the required global step,
`max_steps=30`, and contiguous histories; the local and remote manifests are
byte-identical with SHA-256
`1ecc0af8f374cf430375ae5db3ae6422678fd413b01c59143c6e44a6d59713d9`.
The fail-closed aggregate advanced to 36/40 with four missing units, zero
errors, and no verdict. The released A100 was backfilled with DAPO seed 37.
AERO seed 107 has independently verified private checkpoint 20 at
`f412a1e4d81ca7473a5036411d9a7fb58268c557`, and AERO seed 131 has independently
verified private checkpoint 15 at
`c4a68af97c99679cc37a37fcba533873d6ecd25c`.

At 2026-07-19 13:23 UTC, AERO seed 107 completed all 500 held-out rows with
315 correct (`0.6300`) and passed every final evidence gate. Finished W&B run
`853facf3`, exact private HF commit
`42980267a519c028dd0e5a0592c0a106933fb0b7`, all six complete checkpoint trees,
the final adapter, frozen fingerprints, and 500 contiguous unique completion
hashes reconcile with the accepted local record. The local and remote manifests
are byte-identical with SHA-256
`8da9d6a49ae2961bcc291ad2c12164b0c83a69ff0bed7564431d04f4347a7632`.
The fail-closed aggregate advanced to 37/40 with three missing units, zero
errors, and no verdict. Its released A100 was immediately backfilled with
fresh Dr.GRPO seed 23.

AERO seed 131 subsequently completed with 320/500 exact matches (`0.6400`).
Finished W&B run `bdb898f5`, exact private HF commit
`ec9931f7fe85e6b180dbb85d6c20c3e1646f5272`, all six complete checkpoint
trees, the final adapter, frozen fingerprints, and 500 contiguous unique
completion hashes reconcile with the accepted local record. The local and
remote manifests are byte-identical with SHA-256
`7f8f939913d2518af19194576ac18651d06f1ac26f120c653fba120446561697`.
The fail-closed aggregate advanced to 38/40 with two missing units, zero errors,
and no verdict.

DAPO seed 37 subsequently completed with 314/500 exact matches (`0.6280`).
Finished W&B run `605e7589`, exact private HF commit
`bb7fb354be9f02461e94f69d226b3566e8417368`, all six complete checkpoint
trees, the final adapter, frozen fingerprints, and 500 contiguous unique
completion hashes reconcile with the accepted local record. The local and
remote manifests are byte-identical with SHA-256
`15448dfcde4a095d782b82fa259bb2fcc318a66e780389a0fa987e10d5b1038e`.
The fail-closed aggregate advanced to 39/40 with only Dr.GRPO seed 23 missing,
zero errors, and no verdict.

Dr.GRPO seed 23 subsequently completed with 313/500 exact matches (`0.6260`).
Finished W&B run `63941052`, exact private HF commit
`4318ffd9026b64749561c033ab7c0f3e3841cb84`, all six complete checkpoint
trees, the final adapter, frozen fingerprints, and 500 contiguous unique
completion hashes reconcile with the accepted local record. The local and
remote manifests are byte-identical with SHA-256
`3057541e3364213fa2ee36dfdfde995cf78b59c49fecfa50665739dcdb8a07de`.
The aggregate initially accepted all 40 units and emitted paired verdicts.
The subsequent campaign-wide hash-integrity audit found legacy unhashed
prefixes in GRPO seeds 11, 89, and 107; DAPO seed 131; and GSPO seeds 11 and
71. Those records are preserved with checksums under
`zvf-program/audit/results/full/legacy-unhashed-2026-07-19`, but the hardened
campaign and aggregate rejected them. DAPO seed 131 has since passed exact
checkpoint-30 replay and independent local/W&B/private-Hub reconciliation at
commit `39f916902470b3b800af5c8d60d398a164cd2b95`; GRPO seed 89 also
passed at commit `f88da35f8939dc7ff74ed0b37a004fa8c78379a8`, and GRPO seed 107
passed at commit `ffcbfffd322a181e82c0d3a552ec611432dce471` with 317/500 correct
and manifest SHA-256
`792e11fff9154c6d421322d065ef1c3be296054426fe5ed3c4a212cb97103dc0`.
GRPO seed 11 has now passed too, with 325/500 correct at exact private commit
`b39028ea32b042247e7ecd3ee228b8b302c55226` and manifest SHA-256
`b08c10604b0222b5296fb4345931401d9e5ac207cfc2260108786d3904a3f343`.
GSPO seed 11 has also passed, with 320/500 correct at exact private commit
`f064274b7c79372fe3fa1501737ae8a5398bec07` and manifest SHA-256
`bf40c2437315d306ff7d22737d3d750e7950969b9c8ff9612d7f5add4acdc592`.
GSPO seed 71 subsequently resumed from exact checkpoint 30 and the verified
416-row private ledger. It completed with 319/500 exact matches (`0.6380`) at
private commit `c0c0b968a61a9ade251b5b7e6ece3119197dc1b1`. Finished W&B run
`10ef44ab`, checkpoints 5/10/15/20/25/30, the final adapter, frozen
fingerprints, 500 contiguous valid unique completion hashes, and byte-identical
manifest SHA-256
`0243e256a3cdae62d30cad889f8bdf19bfe6d7cc2862edc60edffc23586ac3ed`
all reconcile. The full verifier now reports 40/40 with zero errors; the frozen
aggregate reports `COMPLETE` and emits DAPO `DISAPPEARS`, with GSPO, Dr.GRPO,
and AERO `INCONCLUSIVE`.

## Remaining empirical obligations

These require accelerator time, an external environment/data source, or both.
They are deliberately not replaced with toy numbers.

| ID | Obligation | Prepared artifact | Acceptance condition | Blocker / owner |
|---|---|---|---|---|
| E1 | **Closed:** repair all six legacy held-out traces from their exact private checkpoint-30 trees | Hardened campaign/aggregate validators, preserved legacy receipts, evaluation rewind path | All six repaired manifests contain indices 0--499, 500 unique valid completion hashes, recomputed scores, matching private-Hub commits; verifier and aggregate return `COMPLETE` | Closed 2026-07-20; final verdicts emitted |
| E2 | PPO/SAO cause-aware routing evaluation: 5 arms × 5 seeds in the GRPO/PPO cell and the SAO agentic cell (50 arm/seed jobs) | `platform_hybrid/experiments/signal_starvation/preregistration.json` and instrumentation package | H1--H4 scored at matched action-token/environment budgets with prompt-clustered intervals | LLM PPO/SAO training stack, SWE-Bench environment, accelerator budget |
| E3 | M-GRPO agentic audit: 3 arms × 5 seeds (15 jobs) | `zvf-program/audit/preregistration_mgrpo.json` | Full planner/sub-agent manifests, matched budgets, all seeds, paired held-out intervals | Multi-agent training implementation, tools sandbox, GPU budget |
| E4 | P07 controller bakeoff | Existing fixed-token controller protocol and EGM extensions | Static G16, naive symmetric retry, failure-only, boundary-aware, and full triage compared at equal tokens | Same compute campaign as E2 |
| E5 | Direct group-size and scaling confirmations | P01/P03 protocols and current reconstructed/direct provenance labels | Multi-seed token-matched direct G sweep including G=32 and matched cross-scale cells | GPU budget |
| E6 | Length-bias external validity | P04 capped null-test protocol | Uncapped or long-horizon multi-seed mediation study | GPU budget and longer generation policy |
| E7 | Fraud side-study external validity | P08 parked scope and honest synthetic/noisy-sensor analysis | Real, cross-institution or temporally held-out fraud data under approval | Data access, privacy/ethics approval |

## Remaining release and ecosystem obligations

| ID | Obligation | Work completed here | Acceptance condition | Authority needed |
|---|---|---|---|---|
| R1 | TRL, verl, and OpenRLHF adoption | Submission-ready issue text and acceptance criteria in `zvf-program/position/ADOPTION_PACK.md` | Public issue/PR URLs linked from the manuscript | User/project maintainer approval for external writes |
| R2 | Default trainer emitters | Canonical schema plus stdlib generator/verifier are executable | Tested opt-in adapters upstreamed for the three trainers | Framework-specific implementation and review |
| R3 | Literature-scale PDF Auditor and full arm launcher | Released-vs-planned claims corrected; contracts and refusing aggregator exist | Versioned CLI/package with integration tests and immutable examples | Engineering work; no claim of completion |
| R4 | Stable manuscript identifiers | Working-paper citations are explicit | arXiv/DOI/venue IDs replace internal working-paper keys | Submission and publication authority |
| R5 | Registry unknown-field backfill | Health audit now reports the real 28-stack/18-delta state | Source-backed values replace the 61 fully-unknown MIN-REPORT item blocks, or providers attest that they are opaque | Original authors/runtime providers; never infer hidden settings |
| R6 | Final authorship and venue routing | Companion drafts use the known Arvind C R / PES University metadata; venue scopes are explicit | Every coauthor approves authorship/order, blind/non-blind variant, and submission venue | Human authors |

## Verification commands

```bash
python3 -m unittest platform_hybrid/experiments/signal_starvation/test_metrics.py
python3 zvf-program/audit/test_aggregate_audit.py
python3 zvf-program/audit/aggregate_audit.py \
  --input-dir zvf-program/audit/results/full \
  --output zvf-program/audit/results/audit.json
python3 zvf-program/audit/verify_colab_e1_campaign.py \
  --output zvf-program/audit/results/campaign-verification.json

python3 platform_hybrid/registry/query.py validate
python3 platform_hybrid/registry/query.py drift
python3 platform_hybrid/registry/query.py health
python3 platform_hybrid/registry/query.py validate-strict

python3 autoresearch/improve-260714-1806/inventory_papers.py
python3 autoresearch/improve-260714-1806/self_review_corpus.py
```

Only after all eight hash-complete seed records for every arm are present may
the audit aggregator print `COMPLETE` and emit frozen paired verdicts. It
currently refuses the one remaining legacy record, as required by the passing
safety tests.

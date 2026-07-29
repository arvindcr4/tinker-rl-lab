# Flagship conformance pilot control plane

This directory turns `pilot_preregistration.json` into an exact 24-unit
screening matrix without allocating external compute.  It is intentionally
separate from every frozen E1 result, W&B run, Hugging Face repository, and
Colab session namespace.

Generate the amended version-2 dry-run manifest with:

```bash
PYTHONPATH=zvf-program/flagship python -m pilot.plan_screening --write
```

The current preregistration is version 2, implementation revision 6, and
`ready_to_run` under user-authorized correction `A1-R4`. A fresh source-bound
A100 smoke gates the screening DAG. The six corpus identities remain frozen to
revision-4 generator provenance, while the 24 unit identities bind the new
revision-6 training source. Balanced seed 11 reuses its accepted final commit,
balanced seed 23 resumes its verified group-20 prefix, and incomplete corpora
run only from the frozen revision-4 archive. Only one corpus session may run at
a time, each incomplete corpus has at most three VM attempts, and confirmatory
jobs and accelerator substitution remain forbidden.

Each plan also carries an explicit readiness report. The numeric execution
contract resolves the fixed-step/token-matching question with one shared,
immutable 100-group corpus per regime/seed and conservative identical charging
across its four conditions. The remote trainer, token/FLOP ledger, checkpoint
recovery, verifier, and go/kill analysis passed the recorded offline gate before
GPU authorization changed.

## Current execution state and amendment boundary (2026-07-22)

Version 1 is closed as infrastructure evidence. The balanced-equal-length
seed-11 corpus was attempted three times after capacity returned. W&B runs
`ujryg527`, `lwjtk9dk`, and `hge0xhav` were all reclaimed after roughly
2h20m; the latter two ended identically after group 99 and 393,714 charged
tokens. Because version 1 uploaded only after all 100 groups, each private Hub
repository remained a zero-payload skeleton and no prefix could be resumed.
No corpus or scientific unit was accepted.

The user authorized amendment `A1-corpus-intermediate-persistence` on
2026-07-22. Version 2 keeps every scientific field unchanged and changes only
infrastructure persistence, retry, and provenance semantics. It uses new
protocol/source-scoped identities and a separate control surface. The first
version-2 launch found that `runtime_install.py` was bound by the source
manifest but omitted from the uploaded archive; it stopped before W&B or group
generation. Implementation revision 2 records that correction as
`A1-R1-complete-source-bundle-and-preserve-attempt-logs`, uses
`plans-v2-corpus-resume-r1/` and `launch-v2-corpus-resume-r1/`, archives every
attempt log/result before retry, and retries automatically only for recognized
provider-infrastructure failures. Version-1 and superseded version-2 state are
preserved.

The revision-2 local gate passes 106/106 exact pinned tests plus the 55/55
focused amendment suite. Its fresh source-bound A100 smoke is independently
accepted. Balanced-equal-length seed 11 completed all 100 groups on an A100 as
W&B run `b8eoqd09`. Its first immutable group-20 prefix is independently
verified at private-Hub commit
`46030fba999dccbabc40567ab8f605589aa6a50a`, with all 20 group artifacts,
source bindings, and token/FLOP prefix intact. Its replacement group-40 prefix
is independently verified at exact commit
`55091520f883bec456fe3f3334edf68dbc770013`, with all 40 group artifacts and
the 160,423-token prefix ledger intact. The group-60 replacement is likewise
independently verified at exact commit
`4776e185ee8a91e924672179062380fb9423bddb`, with all 60 group artifacts and
the 236,615-token prefix ledger intact. The final resumable group-80 prefix is
independently verified at exact commit
`2faf00b02c5c81fcdcd2c4ed9e97e5fa8b721101`, with all 80 group artifacts and
the 317,482-token prefix ledger intact. The final verifier then re-hashed the
complete corpus, its source manifest, and that exact group-80 checkpoint;
balanced-equal-length seed 11 is accepted at private-Hub commit
`91ec135ce5ffd562d991e535a16cae28c6552389`, corpus fingerprint
`8b24a0520a97f0d5101c2662a1e3e369e8342c1759c9963a0ccb909b01525589`,
with 396,672 charged generated tokens and zero resumes. The local launcher had
exited while the remote execution survived, so
`launch-v2-corpus-resume-r1/recovery/corpus__balanced_equal_length__s11__attempt-1.json`
records its verified no-duplicate adoption. A launchd-owned supervisor now
continues the remaining DAG. No scientific unit is accepted yet.

That r1 supervisor exposed two independent implementation defects before any
scientific step was accepted. First, the initial intended/native units failed
before optimizer step 1 because gradient-checkpoint recomputation escaped the
deterministic SDPA context. Revision r2 fixed the backward context, but its
fresh non-scientific A100 smoke then emitted mathematically invalid cosines
`1.00221848487854` and `1.0022610425949097`; the previous verifier checked only
finiteness. Both r1 units, the r1 corpus, and all r2 state are preserved but
excluded.

Implementation revision 4 (`A1-R3-bound-cosine-diagnostics-and-verifiers`)
computes receipt diagnostics in float64 and enforces cosine `[-1,1]`,
non-negative relative L2, and positive gradient norms in the producer,
preflight verifier, and every full-record receipt. Its exact pinned local gate
passes 109/109, the focused gate passes 55/55, and Ruff is clean. Fresh r3
plans and launches live under `plans-v2-corpus-resume-r3/` and
`launch-v2-corpus-resume-r3/`, bound to protocol SHA-256
`04d20f712f652f80754fa4c8c0a3f48d4d2f1c5d716b3981746322c938b21970` and
launch fingerprint
`f01ad8e3991365fcf36160386b32dfdc69c034d1697773f8868b9dd5682d7de3`.
LaunchAgent `ai.openai.codex.flagship-pilot-v2-r3` is active; only its fresh
non-scientific A100 smoke was initially runnable. Attempt 1 is now independently
accepted with valid cosines `0.999957795529626` and
`0.9999999999997982`, positive norms/FLOPs, exact pins, and exact source hashes.
The smoke session was deleted and balanced-equal-length seed 11 corpus attempt
1 is the only running downstream job. No r3 corpus or scientific unit is
accepted yet.

That r3 corpus now has an independently verified group-20 prefix at exact
private-Hub commit `7c6d13ee7b22ef1a9ca83f2a550a43fbcff8a7e9`, fingerprint
`a054b9c6f1ce9a69424677f201c46c242c805bb22674e8744fedb381e3fe556b`.
The verifier downloaded and re-hashed the source manifest and all 20 group
artifacts, reconciling 80,081 charged tokens, profiler steps 1/20, exact
A100/runtime bindings, one W&B attempt (`3jpcepfy`), and zero resumes. The
same run continues toward group 40; the prefix is resumable infrastructure and
does not count as an accepted corpus or scientific unit.

The replacement group-40 prefix is likewise independently verified at exact
private-Hub commit `b23d1da97dc5dadd3da6d133ba3ffb048d055af0`, fingerprint
`5c1a6cf763737d63efa116e1bac67a5061e06f34dbd360ae6e1fefd7b42dda3b`.
All 40 group files and the source manifest re-hash exactly, reconciling
160,423 charged tokens, profiler steps 1/20/40, exact pins, one W&B attempt
(`3jpcepfy`), and zero resumes. The same A100 run continues toward group 60;
the prefix remains resumable infrastructure rather than an accepted corpus.

The group-60 replacement is independently verified at exact private-Hub commit
`a0c83171731c497ce13ae1dcc14b48b045c72956`, fingerprint
`dd7caf181a7463196d86d404ea21ff2fe5b88e8878f388757b70b8a268ff5790`.
The verifier downloaded and re-hashed all 60 group files, the checkpoint
manifest, and all 14 source entries, reconciling 236,615 charged tokens,
profiler steps 1/20/40/60, 15,644 profiled generated tokens, exact pins, one
W&B attempt (`3jpcepfy`), and zero resumes. W&B exposes the identical
commit/fingerprint and the same A100 run has continued through group 61 toward
group 80. This prefix remains resumable infrastructure rather than an accepted
corpus or scientific unit.

The final resumable group-80 replacement is independently verified at exact
private-Hub commit `ba2a67680eee15e956f406fd9caebc83326967cf`, fingerprint
`c50c78dda0978525d7bf32247087850436e844b43825234d572dc5a2ed3e4b12`.
The verifier downloaded and re-hashed all 80 group files, the checkpoint
manifest, and all 14 source entries, reconciling 317,482 charged tokens,
profiler steps 1/20/40/60/80, 19,740 profiled generated tokens, exact pins, one
W&B attempt (`3jpcepfy`), and zero resumes. W&B exposes the identical
commit/fingerprint and token ledger. The same A100 attempt continues toward
the 100-group final record; this prefix is resumable infrastructure, not an
accepted corpus or scientific unit.

Balanced-equal-length seed 11 is now the first independently accepted r3
corpus. The full verifier downloaded all 185 remote files and re-hashed the 100
group artifacts, corpus manifest, and all 14 source entries at exact
private-Hub commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`, corpus
fingerprint
`b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.
It reconciled 396,672 charged tokens, profiler steps 1/20/40/60/80/100, 22,698
profiled generated tokens, exact group-80 checkpoint lineage, exact pins, one
finished W&B run (`3jpcepfy`), one attempt, and zero resumes. The supervisor's
acceptance receipt is durable and the Colab session was released. The current
eligible count is one r3 corpus and zero scientific units. Balanced seed 23 is
the only running corpus builder; the intended/native balanced-seed-11 units are
also running on separate A100s.

The first r3 scientific wave exposed a deterministic protocol contradiction
before optimizer step 1. Intended W&B run `22107a6b` and native run
`07c23895` both emitted the same step-0 evaluation (`accuracy=0.15625`, 64,038
generated tokens) and then failed. Native's archived traceback terminates at
`TrainingContractError: intended gradient norm is non-positive or non-finite`.
Exact replay group 1 has eight zero rewards, so all intended/native/selected
advantages and gradients are jointly zero. The complete accepted corpus has 59
all-zero groups, 3 all-one groups, and 38 mixed groups. Consequently the r3
rule requiring positive norms and numeric cosines at every step cannot emit 62
of the required 100 receipts or satisfy the 95/100 balanced-equivalence gate.
Neither failed unit performed optimizer step 1 or produced an eligible artifact.

The supervisor failed closed and its crash-only LaunchAgent was unloaded before
replacement work could launch. Both unit sessions were released. The detached
balanced-seed-23 corpus was allowed to reach its first resumable boundary and
then stopped; its group-20 prefix independently verifies at private-Hub commit
`b1d897a968470898848ddb85ba24a334c3d59237`, fingerprint
`67d51945e773e9e6aa50a88f8d72a182230c2452bd0285caf00be554b1aa1764`,
with 80,988 charged tokens, profiler steps 1/20, one W&B attempt (`ge121gt6`),
and zero resumes. The stop propagated after W&B row 22 (86,052 charged tokens),
but the Hub head remains the exact group-20 commit and the orphaned run is stale
`running`; rows 21--22 are not recoverable. There are no active Colab sessions
or local controllers. Replacement execution is forbidden until an explicit
joint-zero/one-sided-zero receipt and scoring amendment—and a corpus reuse
versus rebuild decision—is authorized.

At each checkpoint the `resume/` Hub prefix contains the complete accepted
group prefix, a content-addressed manifest, source hashes, the cumulative
token/FLOP prefix ledger, and the W&B attempt ledger. Hub commits are atomic:
an interrupted upload leaves the preceding valid commit addressable. The final
corpus verifier independently re-reads and hashes both the 100-group corpus and
the group-80 checkpoint commit before releasing any downstream unit.

`replay.py` freezes the first causal control: each regime/seed produces one
content-addressed replay corpus consumed in the same order by all four
conditions.  The balanced control keeps all eight rows and makes optimization
lengths equal by active EOS padding.  The variable-length regime chooses the
lexicographically first six-row subset with maximal population length CV and
fails closed unless that CV is at least 0.35.  Generated tokens from rejected
groups remain charged in the ledger.

### Authorized A1-R4 continuation

The prior amendment blocker is resolved. The user authorized A1-R4 with corpus
reuse on 2026-07-22. Revision 5 represents `nonzero`, `joint_zero`, and named
one-sided-zero gradient relations explicitly; zero-vector cosine and relative
L2 values are null. Joint-zero is equivalence/zero effect, one-sided-zero is
maximal divergence, and nonzero thresholds are unchanged. Selected zero
gradients skip `optimizer.step()` but advance the scheduler once.

Frozen corpus provenance is recorded in
`provenance/r3-corpus-bindings.json`, with deterministic archives
`r3-corpus-source.tar.gz` and `r3-control-source.tar.gz`. The accepted balanced
seed-11 final and balanced seed-23 group-20 prefix both pass live revision-6
verification. The exact local gate passes 103/103 pilot tests plus 12/12
preregistration tests, the focused gate passes 69/69, and Ruff check/format is
clean. Current generated control surfaces are
`plans-v2-corpus-resume-r4-1/` and `launch-v2-corpus-resume-r4-1/`;
confirmatory
execution remains forbidden until the screening verdict is GO.

The first revision-5 smoke installed the pinned stack but stopped during remote
environment validation because the uploaded unit bundle omitted the frozen
archive files checked by the protocol. `A1-R4.1` records this pre-model,
pre-W&B packaging failure. Revision 6 includes both content-addressed archives
in the upload and runtime source manifest and uses fresh r4-1 identities.

The fresh revision-6 A100 smoke is now independently accepted with exact pins,
positive phase FLOPs and gradient norms, valid nonzero relations, intended vs
native cosine `0.999957795529626`, and selected vs intended cosine
`0.9999999999997982`. Its session was stopped. Balanced seed 11 was then
reaccepted at the exact frozen final commit, releasing the balanced seed-23
resume and the new revision-6 intended/native seed-11 units.

### A1-R4.2 revision-7 continuation

Revision-6 intended W&B `a0a67b52` proved the joint-zero no-op path, then
failed only because a separately reduced cosine for byte-identical
selected/intended vectors was `1.000000000002599`. Native W&B `87ba3535`
corroborated the frozen joint-zero/nonzero sequence. `A1-R4.2` preserves every
scientific threshold and maps exact vector equality directly to cosine `1.0`,
relative L2 `0.0` under implementation revision 7.

The exact local gate now passes 104/104 pilot and 12/12 preregistration tests,
70/70 focused tests, byte-identical manifest regeneration, and Ruff across all
22 changed Python files. The r4-2 protocol and unit source hashes are
`87d929d0a3af789d3ba3ee10a1f4c3e83572ecec7cc4efa28ca032008f88fbc4`
and `005d3f8242b992cf70af2944c2b3f63351f5d3e00e95cdc5caeb40d1261b0918`.
Use `plans-v2-corpus-resume-r4-2/` and
`launch-v2-corpus-resume-r4-2/` for current execution.

The fresh revision-7 A100 smoke is independently accepted with exact pins,
positive FLOPs/norms, intended/native cosine `0.999957795529626`, relative L2
`0.009205099545490102`, exact selected/intended cosine `1.0`, and an applied
optimizer update. Balanced seed 11 is reaccepted at its frozen final commit.
Surviving seed-23 W&B `ncpafe25` independently passed group 40 at commit
`b45dc64a59a8cd7fb068d0f2182c507c34db8aec` / fingerprint
`1d7e72efb8df8e22beb15a9756d8255aa6b44f4f4a9f4af3d53b547143138c37`
and is live beyond that boundary without duplication. Fresh revision-7
intended/native seed-11 sessions are live under suffix `87d9005d`. The hard
three-A100 and one-corpus ceilings remain enforced; confirmatory execution is
forbidden unless screening returns GO.


### Resume adoption after interrupted controller (2026-07-22 evening)

Thread `019f880f-c98e-7a93-a4c7-83352e8eff8a` stopped mid-monitor with both
revision-7 seed-11 units already allocated. Live adoption found:

- intended W&B `8170fe50` and native W&B `1724b02f` still running on A100 sessions
  `fpilot-*-s11-87d9005d`, with immutable Hub checkpoints through step 40 and live
  mechanism steps through 59;
- seed-23 W&B `ncpafe25` crashed after group 72, but its group-60 checkpoint is
  independently verified at private-Hub commit
  `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba` / fingerprint
  `a6e170736a463412b3067460f524e4e10e06ee3b0d03402861a4f106953a3308`
  (239,047 charged tokens; receipt under
  `launch-v2-corpus-resume-r4-2/recovery/`);
- superseded r3/r4/r4-1 LaunchAgents are contained; only the two non-`RunAtLoad`
  r4-2 unit controllers remain loaded.

Durable state now lives in
`launch-v2-corpus-resume-r4-2/supervisor_state.json` and
`launch-v2-corpus-resume-r4-2/execution-notes.md`. Scientific-unit count remains
zero accepted; confirmatory execution remains forbidden.

### Final-attempt unit loss and next wave

After the interrupted-controller adoption, both revision-7 intended/native balanced
seed-11 final attempts progressed through independently verified step-40 checkpoints
and live W&B steps through 59, then lost their A100 assignments (`keep-alive` 404 /
pruned). Attempt-3 logs and recovery receipts are archived under
`launch-v2-corpus-resume-r4-2/`; those two units are `failed_infrastructure` with
attempts exhausted. Partial step-40 progress is not unit acceptance.

The next capacity-legal wave uses free A100s for balanced seed-23 corpus resume from
independently verified group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
plus the remaining seed-11 units `epsilon_only` and `reduction_only`, still under the
one-corpus / three-A100 ceilings.

### Wave-3 acceptances and wave-4 launch (2026-07-23)

Wave 3 completed and is fully reconciled. Balanced seed-23 corpus is accepted
complete: 100/100 groups at private-Hub commit
`664b9189dec25ded62bd74166a8dab0bf5727589`, fingerprint
`bf54deaf4a62b0cffc69e40452d8133eba206163bcff44c1b6c1c6c83918e225`, 400,448
charged tokens, resume_count 2, W&B `k1uazlrj` finished (complete independent
verify receipt and corpus acceptance under `launch-v2-corpus-resume-r4-2/`).
The remaining balanced seed-11 units are the first two accepted scientific
units of the campaign, each after the full fail-closed
`verify_unit_remote` pass (all checkpoint manifests/files, final adapter,
held-out evidence, finished W&B run, exact plan/corpus config):

- `epsilon_only` balanced s11: W&B `1cf851c0`, final held-out accuracy
  0.1328125, HF artifact commit `803546b3bd0a172e4bc6b28598b7e075eac77476`;
- `reduction_only` balanced s11: W&B `b2bd75df`, final held-out accuracy
  0.1484375, HF artifact commit `d887b1844fdc40d016323766e3079748fa91bd53`.

All three wave-3 A100 sessions were stopped cleanly by their launchers before
the 2026-07-23 03:43 local host reboot, so the reboot killed only finished
controllers; `colab sessions` verified empty on return.

Wave 4 is live under launchd LaunchAgents (`RunAtLoad=false`,
`KeepAlive=false`), one attempt each consumed by a tool-teardown kill plus a
pre-allocation credential miss that was fixed durably via the standard HF CLI
token file:

| Job | Session | W&B | Attempt |
|---|---|---|---|
| `corpus__balanced_equal_length__s37` | `fpcorp-bala-s37-10e4` | `60bhrt9q` | 2/3 |
| `fpilot__intended_full__balanced_equal_length__s23` | `fpilot-inte-bala-s23-87d9005d` | `29173df6` | 2/3 |
| `fpilot__native_trl__balanced_equal_length__s23` | `fpilot-nati-bala-s23-87d9005d` | `f1b5321a` | 2/3 |

Both unit runs are config-reconciled against the accepted seed-23 corpus
(commit `664b9189…` / fingerprint `bf54deaf…`) on A100. Accepted totals: 2
corpora (balanced s11, balanced s23), 2 scientific units (epsilon/reduction
balanced s11). intended/native balanced s11 remain terminal
`failed_infrastructure`; do not relaunch. One-corpus and three-A100 ceilings
remain enforced; confirmatory execution remains forbidden unless screening
returns GO.

### Wave-4 acceptances and wave-5 queue (2026-07-23)

Wave 4 completed and is fully reconciled; every session was stopped cleanly by
its launcher. Balanced seed-37 corpus is accepted complete: 100/100 groups at
private-Hub commit `b2cb4ca32e52cf61b9388d86a49701aa34df52f8`, fingerprint
`673d3f27a08c650e900069ca05db23fe45407299be13b726ae81c36c96578a20`, 398,761
charged tokens, resume_count 0, W&B `60bhrt9q` finished. Both balanced seed-23
units passed the full fail-closed `verify_unit_remote` acceptance path:

- `intended_full` balanced s23: W&B `29173df6`, final held-out accuracy
  0.15625, HF artifact commit `bec743e68e369c19df461b29ff6082595bb42def`;
- `native_trl` balanced s23: W&B `f1b5321a`, final held-out accuracy 0.1328125,
  HF artifact commit `36424ea8dcdb309daebc1e9e2c4f2133bb8fa116`.

Accepted totals: 3 corpora (balanced s11/s23/s37), 4 scientific units
(epsilon/reduction s11; intended/native s23). intended/native balanced s11
remain terminal `failed_infrastructure`; do not relaunch. Next capacity-legal
wave: `corpus__filtered_variable_length__s11` plus
`fpilot__epsilon_only__balanced_equal_length__s23` and
`fpilot__reduction_only__balanced_equal_length__s23`, under the same one-corpus
/ three-A100 ceilings and the confirmatory-execution ban.

### Filtered seed-11 corpus validation failure (2026-07-23)

Wave 5 launched 2026-07-23T02:05Z: the two balanced seed-23 ablation units are
healthy in training, but `corpus__filtered_variable_length__s11` died during
remote generation with
`ReplayContractError: filtered pool maximum selected-row length CV 0.000000 is
below 0.350000` — the deterministic length-variation contract gate
(`replay.py FILTERED_MIN_LENGTH_CV`) rejecting the generated pool before any
group was produced. Classified `failed_validation` (attempt 1/3; no
infrastructure signature; W&B `usgmq1en` crashed with zero groups). Automatic
retry is forbidden because frozen revision-7 source would deterministically
reproduce the same gate failure, and new wave launches are halted while the
validation failure stands, per the fail-closed scheduler doctrine. The two
in-flight balanced units are independent of the filtered regime and continue
to completion. Unblocking the filtered corpora requires investigation of
filtered pool generation or a frozen-source amendment — surfaced for user
decision; no gate was weakened.

### Wave-5 ablation acceptances (2026-07-23)

Both balanced seed-23 ablation units finished cleanly and passed the full
fail-closed `verify_unit_remote` acceptance path:

- `epsilon_only` balanced s23: W&B `c3eec6d0`, final held-out accuracy
  0.1328125, HF artifact commit `5d769e889a503658e8785dbf2a79dceb4a439a04`;
- `reduction_only` balanced s23: W&B `920fb29b`, final held-out accuracy
  0.15625, HF artifact commit `db5d04372a5ec298b78730c26ce2779052e6366c`.

Accepted totals: 3 corpora (balanced s11/s23/s37), 6 scientific units
(epsilon/reduction s11; intended/native s23; epsilon/reduction s23). The
filtered seed-11 corpus `failed_validation` remains open; new wave launches
stay halted per the fail-closed scheduler doctrine. Dependency-ready if the
user authorizes continuing balanced work: all 4 balanced-s37 units. The 12
filtered-regime units and 2 remaining filtered corpora stay gated on the
filtered-corpus decision.

Root-cause follow-up (read-only, no source or gate changed): the filtered
regime is structurally unpassable under the frozen contract. Direct
measurement on the accepted balanced s23 corpus shows essentially every
completion filling the 512-token cap (group-000 raw lengths `[512×8]`;
group-001 `[512×7, 479]`; group-050 `[512×8]`) — Qwen3-1.7B almost never emits
EOS inside 512 tokens, so completion lengths cannot vary. The balanced regime
is unaffected by design (`selected_cv=0.0`; right-padding is charged as
active), while the filtered regime's CV ≥ 0.35 gate requires early-EOS
variation this model/cap combination effectively never produces; MATH-500
makes truncation even more certain. Filtered s23/s37 would fail identically.
Unblocking needs a user-authorized amendment (completion length, CV gate, or
pool semantics), a filtered-regime descope, or acceptance of the pilot as
balanced-complete with the filtered regime recorded as contract-infeasible
evidence.

### Wave-6 continuation state (2026-07-27)

The filtered regime is now recorded as contract-infeasible and descoped. Four
balanced seed-37 scientific units remain `pending_quota_reset`; intended-full
and reduction-only have independently verified step-60 checkpoints. A fresh
named A100 probe (`fprobe-r4-2-capacity-20260727`) was rejected with the same
quota/entitlement error, so no unit was relaunched, no attempt was consumed,
and no accelerator was substituted. The obsolete KeepAlive quota monitor was
unloaded after it was found crash-looping on a missing `/tmp` script.

Metadata correction (2026-07-27): fourteen descoped descendant status summaries
had accidentally named `Qwen2.5-0.5B`. The preregistration and executed runtime
identify `Qwen/Qwen3-1.7B`; the summaries are now model-neutral. This changes no
gate, error observation, job status, or scientific artifact.

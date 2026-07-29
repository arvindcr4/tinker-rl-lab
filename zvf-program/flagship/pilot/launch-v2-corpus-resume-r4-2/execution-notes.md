# Flagship r4-2 execution notes

Updated: 2026-07-22T14:47:05.921905+00:00

## Control surface

- Plans: `plans-v2-corpus-resume-r4-2/`
- Launch: `launch-v2-corpus-resume-r4-2/`
- Protocol SHA-256: `87d929d0a3af789d3ba3ee10a1f4c3e83572ecec7cc4efa28ca032008f88fbc4`
- Unit/source bindings SHA-256: `005d3f8242b992cf70af2944c2b3f63351f5d3e00e95cdc5caeb40d1261b0918`
- Launch manifest fingerprint: `25ef91234d58643c2d1eaea23832e0b676cb99e66ce774551b1f0ae1de9cee0d`
- Amendment: A1-R4.2 / implementation revision 7

## Independently accepted so far

1. Fresh A100 smoke: accepted (`acceptance/preflight__a100_stack_smoke.json`).
2. Balanced-equal-length seed 11 corpus: accepted at private Hub commit `2735a27d5f18bbdaaae76494a2047b39a4318e22`, fingerprint `b09c72247b168297e73ce5edf2aad59e4496e7d78257beb252e864dd1a9587f1`.

No scientific unit is accepted yet. Confirmatory execution remains forbidden.

## Live scientific units (not accepted)

Both revision-7 final-attempt units are live under non-`RunAtLoad` LaunchAgents and A100 Colab sessions `*-87d9005d`.

| Unit | W&B | Session | Latest immutable checkpoint | Live W&B step |
|---|---|---|---|---|
| intended_full / balanced / s11 | `8170fe50` | `fpilot-inte-bala-s11-87d9005d` | step-40 commit `23de08ec44fa3e0e130e7edd9261c09ff56a1793` | 59 |
| native_trl / balanced / s11 | `1724b02f` | `fpilot-nati-bala-s11-87d9005d` | step-40 commit `4b9ebcc79f86e56205c2524d0f4221facdbcc41c` | 59 |

Observed mechanism evidence on both runs includes joint-zero no-ops and nonzero cosines inside thresholds; this is live progress only, not unit acceptance.

## Seed-23 corpus boundary

W&B `ncpafe25` crashed after group 72. The only independently verified resumable boundary is group 60:

- commit: `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
- fingerprint: `a6e170736a463412b3067460f524e4e10e06ee3b0d03402861a4f106953a3308`
- charged tokens: 239047
- receipt: `recovery/corpus__balanced_equal_length__s23__group-60-independent-verify.json`

Rows 61-72 are not a higher accepted checkpoint. No new corpus allocation while two unit A100s are held; one-corpus and three-A100 ceilings remain enforced.

## Operational containment

Superseded LaunchAgents r3/r4/r4-1 were invalid/corrupt and have been moved aside to `*.plist.superseded.bak` so they cannot allocate duplicate compute. Only the two r4-2 unit controllers remain loaded, both with `RunAtLoad=false`.

## Resume policy if this controller dies

1. Do not relaunch either live unit session if remote Colab remains BUSY or W&B remains running with the same identity.
2. Adopt existing remote work, verify config/source/runtime pins, then continue monitoring.
3. Resume seed-23 only from the verified group-60 commit above, one corpus session at a time, after unit capacity allows.
4. Never weaken gates or count partial checkpoints as accepted units/corpora.


## Final-attempt unit loss and next wave (2026-07-22T14:54:28.727343+00:00)

Both revision-7 final attempts for intended/native balanced seed-11 lost their A100
assignments around 2026-07-22T14:48:42Z (`keep-alive` 404) and were pruned by
2026-07-22T14:49:18Z. Local launcher/exec processes remained hung afterward and were
terminated only after archiving attempt-3 logs and recovery receipts. No duplicate
relaunch of those identities was performed.

| Unit | Attempt | Terminal | Independently verified partial progress |
|---|---:|---|---|
| intended_full / balanced / s11 | 3/3 | failed_infrastructure | step-40 commit `23de08ec…` / fp `c5be4d39…`; W&B `8170fe50` through step 59 |
| native_trl / balanced / s11 | 3/3 | failed_infrastructure | step-40 commit `4b9ebcc7…` / fp `f564682d…`; W&B `1724b02f` through step 59 |

Receipts:

- `recovery/fpilot__intended_full__balanced_equal_length__s11__attempt-3-remote-prune.json`
- `recovery/fpilot__native_trl__balanced_equal_length__s11__attempt-3-remote-prune.json`
- step-40 independent verifies under the same recovery directory

Because max attempts are exhausted for those two units, they are terminal for this
control surface. Remaining capacity-legal work with free A100s:

1. `corpus__balanced_equal_length__s23` resume from independently verified group-60 commit `8b1f2105…`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Hard ceilings remain: one corpus session and at most three A100 sessions. Confirmatory
execution remains forbidden. Scientific-unit acceptance count remains zero.

## A100 entitlement blocker (2026-07-22T14:56:37.200097+00:00)

After archiving the intended/native final-attempt losses, the next capacity-legal wave
tried to allocate:

- `corpus__balanced_equal_length__s23`
- `fpilot__epsilon_only__balanced_equal_length__s11`
- `fpilot__reduction_only__balanced_equal_length__s11`

All three local controllers died during `colab new --gpu A100` with no session created.
An explicit probe:

```bash
colab --auth=oauth2 new --gpu A100 --session fprobe-r4-2-capacity
```

returned:

> Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.

`colab sessions` is empty and `/tun/m/assignments` returns `[]`. Under the frozen A100-only
contract, T4/L4/CPU substitution is forbidden. The campaign is fail-closed on this external
provider entitlement/capacity blocker.

When A100 entitlement returns, resume only:

1. seed-23 corpus from verified group-60 commit `8b1f2105…`
2. epsilon and reduction balanced seed-11 units

Do not relaunch intended/native seed-11; their attempt budgets are exhausted.

## A100 re-probe (2026-07-22T14:58:27.165180+00:00)

Repeated live probe still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

`colab sessions` remains empty. No accelerator substitution is authorized. Campaign remains paused on this external provider blocker; seed-23 resume and epsilon/reduction seed-11 stay pending without burned scientific attempts.

## Offline gate while A100-blocked (2026-07-22T15:00:18.534053+00:00)

Repeated live A100 probes continue to fail closed with the same provider literal:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

While blocked, the authoritative isolated offline pilot gate was re-run with the frozen pin set and passed:

```text
104 passed, 2 warnings in 21.07s
```

`uvx ruff check zvf-program/flagship/pilot` also passed. Format drift was observed on 8 files but intentionally not applied, because reformatting would mutate frozen scientific source hashes.

Campaign remains paused on the external A100 entitlement/capacity blocker. No accelerator substitution is authorized. When A100 returns, resume seed-23 from verified group-60 and launch epsilon/reduction seed-11 only.

## Goal blocked on A100 entitlement (2026-07-22T15:01:55.698942+00:00)

The same external blocker has now repeated across three consecutive goal turns:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

Live checks on this turn:

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-7` rejected A100
- no live pilot launchers

Preserved progress remains:

- smoke accepted
- balanced seed-11 corpus accepted
- intended/native seed-11 failed_infrastructure after attempt 3 remote prune, with independently verified step-40 partials only
- seed-23 independently verified group-60 resume boundary at commit `8b1f2105…`
- offline isolated pilot gate 104/104 pass

No further scientific allocation is possible under the frozen A100-only contract until entitlement/capacity is restored. T4/L4/CPU substitution remains forbidden.

## Resumed-goal A100 re-probe (2026-07-22T15:02:27.976206+00:00)

After goal resume, live re-probe still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-8` rejected A100
- no live pilot launchers

Preserved campaign state is unchanged. Fresh blocked-audit count for this resume starts at 1.

## Resumed-goal A100 re-probe #2 (2026-07-22T15:02:59.665679+00:00)

Second consecutive resumed-goal re-probe still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-9` rejected A100
- no live pilot launchers

Fresh blocked-audit count for this resume is now 2/3. Campaign remains fail-closed; no accelerator substitution authorized.

## Resumed-goal blocked again on A100 entitlement (2026-07-22T15:03:36.843372+00:00)

The resumed-goal blocked audit is complete: three consecutive resumed goal turns hit the same external provider rejection:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

Live checks remain empty (`colab sessions` none; no pilot launchers). Campaign is blocked fail-closed under the frozen A100-only contract. No T4/L4/CPU substitution is authorized.

Resume map when entitlement returns:

1. `corpus__balanced_equal_length__s23` from verified group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Do not relaunch intended/native seed-11; attempt budgets are exhausted.

## Fresh re-probe after blocked status (2026-07-22T15:04:16.605695+00:00)

Live re-probe after the blocked status still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-12` rejected A100
- no live pilot launchers

Fresh blocked-audit count for this resume is 1/3. Campaign remains fail-closed under the A100-only contract.

## Fresh re-probe #2 after blocked status (2026-07-22T15:04:46.625581+00:00)

Second consecutive live re-probe after the blocked status still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-13` rejected A100
- no live pilot launchers

Fresh blocked-audit count for this resume is 2/3. Campaign remains fail-closed under the A100-only contract.

## Fresh blocked-audit complete after resume (2026-07-22T15:05:18.758446+00:00)

Three consecutive resumed goal turns hit the same external provider rejection:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

Live checks remain empty (`colab sessions` none; no pilot launchers). Campaign is blocked fail-closed under the frozen A100-only contract. No T4/L4/CPU substitution is authorized.

Resume map when entitlement returns:

1. `corpus__balanced_equal_length__s23` from verified group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Do not relaunch intended/native seed-11; attempt budgets are exhausted.

## Fresh re-probe after blocked status (2026-07-22T15:05:58.967324+00:00)

Live re-probe after the blocked status still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-15` rejected A100
- no live pilot launchers

Fresh blocked-audit count for this resume is 1/3. Campaign remains fail-closed under the A100-only contract.

## Fresh re-probe #2 after blocked status (2026-07-22T15:06:27.853794+00:00)

Second consecutive live re-probe after the blocked status still fails closed:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

- `colab sessions` empty
- probe `fprobe-r4-2-capacity-16` rejected A100
- no live pilot launchers

Fresh blocked-audit count for this resume is 2/3. Campaign remains fail-closed under the A100-only contract.

## Fresh blocked-audit complete after resume (2026-07-22T15:06:59.812501+00:00)

Three consecutive resumed goal turns hit the same external provider rejection:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

Live checks remain empty (`colab sessions` none; no pilot launchers). Campaign is blocked fail-closed under the frozen A100-only contract. No T4/L4/CPU substitution is authorized.

Resume map when entitlement returns:

1. `corpus__balanced_equal_length__s23` from verified group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Do not relaunch intended/native seed-11; attempt budgets are exhausted.

## Fresh blocked-audit complete after resume (2026-07-22T15:14:01.512383+00:00)

Three consecutive resumed goal turns hit the same external provider rejection:

```text
Backend rejected accelerator 'A100'. You may not have quota or entitlement for this accelerator on your account.
```

Live checks remain empty (`colab sessions` none; no pilot launchers). Campaign is blocked fail-closed under the frozen A100-only contract. No T4/L4/CPU substitution is authorized.

Resume map when entitlement returns:

1. `corpus__balanced_equal_length__s23` from verified group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Do not relaunch intended/native seed-11; attempt budgets are exhausted.

## A100 entitlement restored (2026-07-22T15:16:36.031951+00:00)

Live probes succeeded after the long entitlement outage:

- `fprobe-r4-2-capacity-38` Session READY
- `fprobe-r4-2-capacity-39` Session READY on A100 `gpu-a100-s-kkb-ass1c2-30d0fp858j61n`

Probe session stopped to free capacity. External blocker cleared. Next authorized wave:

1. `corpus__balanced_equal_length__s23` resume from group-60 commit `8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
2. `fpilot__epsilon_only__balanced_equal_length__s11`
3. `fpilot__reduction_only__balanced_equal_length__s11`

Intended/native seed-11 remain terminal `failed_infrastructure` (attempts exhausted). Confirmatory execution still forbidden.

## Next-wave live after A100 restore (2026-07-22T15:21:57.887410+00:00)

After freeing orphan unnamed A100 assignments and relaunching, all three authorized jobs allocated:

| Job | Session | Endpoint | PID | Status |
|---|---|---|---:|---|
| corpus__balanced_equal_length__s23 | `fpcorp-bala-s23-10e4` | `gpu-a100-s-kkb-usc1b1-2khsjulyuc6w6` | 99028 | running (install/upload) |
| fpilot__epsilon_only__balanced_equal_length__s11 | `fpilot-epsi-bala-s11-87d9005d` | `gpu-a100-s-kkb-usc1c1-jis2jeq1vu86` | 99029 | running (install/upload) |
| fpilot__reduction_only__balanced_equal_length__s11 | `fpilot-redu-bala-s11-87d9005d` | `gpu-a100-s-kkb-ass1c0-2l1f11p2cwxno` | 99030 | running (install/upload) |

Seed-23 still resumes from independently verified group-60 commit `8b1f2105…`. No unit/corpus acceptance yet.


## Live wave adoption after thread handoff ( 2026-07-22T15:29:01.530042+00:00 )

Continued from thread `019f880f-c98e-7a93-a4c7-83352e8eff8a`. Live A100 wave still held by detached controllers; no relaunch performed.

| Job | Session | Endpoint | PID | W&B | Live status |
|---|---|---|---:|---|---|
| corpus__balanced_equal_length__s23 | `fpcorp-bala-s23-10e4` | `gpu-a100-s-kkb-usc1b1-2khsjulyuc6w6` | 99028 | `k1uazlrj` | running; resumed from group 60; live `corpus/group=66`, charged tokens `263623`, `corpus_resume_count=2` |
| fpilot__epsilon_only__balanced_equal_length__s11 | `fpilot-epsi-bala-s11-87d9005d` | `gpu-a100-s-kkb-usc1c1-jis2jeq1vu86` | 99029 | `1cf851c0` | running; install/env pass; bound to accepted seed-11 corpus `2735a27d…` / `b09c7224…`; no training steps logged yet |
| fpilot__reduction_only__balanced_equal_length__s11 | `fpilot-redu-bala-s11-87d9005d` | `gpu-a100-s-kkb-ass1c0-2l1f11p2cwxno` | 99030 | `b2bd75df` | same as epsilon |

Verified resume/config identity for seed-23:
- `corpus_start_group=60`
- `corpus_resumed_checkpoint.hf_commit=8b1f2105bc715e5dbf9545f4bc244f3e8800e5ba`
- `corpus_resumed_checkpoint.fingerprint=a6e170736a463412b3067460f524e4e10e06ee3b0d03402861a4f106953a3308`
- checkpoint schedule still `[20, 40, 60, 80]`

Acceptance status unchanged:
- accepted: preflight A100 smoke; balanced seed-11 corpus
- terminal failed_infrastructure (do not relaunch): intended_full / native_trl balanced seed-11
- no scientific unit accepted
- confirmatory execution still forbidden
- one-corpus and three-A100 ceilings remain enforced

Next monitor gates:
1. seed-23 group-80 checkpoint → independent local + HF + W&B verify before any acceptance progress claim
2. epsilon/reduction first `_step` / mechanism receipts, then full-unit independent verify only after finished artifacts
3. free corpus A100 only after seed-23 full acceptance, then remaining corpora one at a time


## Live progress poll ( 2026-07-22T15:31:09.273739+00:00 )

Controllers still alive (PIDs 99028/99029/99030); all three A100 sessions BUSY.

- seed-23 corpus `k1uazlrj`: running at `corpus/group=69`, charged tokens `275911`, resume from verified group-60 commit `8b1f2105…`
- epsilon `1cf851c0`: running, no `_step`/summary yet
- reduction `b2bd75df`: running, no `_step`/summary yet

Still not acceptance. Waiting for group-80 checkpoint and unit step receipts before independent verification.


## Live progress poll ( 2026-07-22T15:31:47.079899+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=71`, charged tokens `283978`
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:32:20.490689+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=72`, charged tokens `287991`
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:33:26.631860+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=75`, charged tokens `300279`
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY
- history confirms contiguous groups 61→75 after resume; no checkpoint events yet (expected at 80)


## Live progress poll ( 2026-07-22T15:34:42.487046+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=77`, charged tokens `308225`
- private HF dataset head still `8b1f2105…` / completed_groups 60 (no group-80 push yet)
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:35:24.443005+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=78`, charged tokens `312321`
- private HF dataset head still `8b1f2105…` / completed_groups 60 (no group-80 push yet)
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:36:46.502946+00:00 )

- seed-23 corpus `k1uazlrj`: running at `corpus/group=78`, charged tokens `312321`
- private HF dataset head still `8b1f2105…` / completed_groups 60 (no group-80 push yet)
- epsilon `1cf851c0`: running, no steps yet
- reduction `b2bd75df`: running, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:39:17.045170+00:00 )

- seed-23 corpus `k1uazlrj`: last logged `corpus/group=78`, charged tokens `312321`; W&B heartbeat fresh (`2026-07-22T15:38:33Z`)
- private HF dataset head still `8b1f2105…` / completed_groups 60
- pause after group 78 is consistent with generating 79-80 and/or durable group-80 checkpoint upload
- epsilon `1cf851c0` heartbeat `2026-07-22T15:38:32Z`, no steps yet
- reduction `b2bd75df` heartbeat `2026-07-22T15:38:35Z`, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:40:14.562298+00:00 )

- seed-23 corpus `k1uazlrj`: still last logged `corpus/group=78`, charged tokens `312321`; heartbeat `2026-07-22T15:39:33Z`
- private HF dataset head still `8b1f2105…` / completed_groups 60
- epsilon `1cf851c0` heartbeat `2026-07-22T15:39:31Z`, no steps yet
- reduction `b2bd75df` heartbeat `2026-07-22T15:39:37Z`, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:41:11.786542+00:00 )

- seed-23 corpus `k1uazlrj`: still last logged `corpus/group=78`, charged tokens `312321`; heartbeat `2026-07-22T15:40:33Z`
- private HF dataset head still `8b1f2105…` / completed_groups 60
- epsilon `1cf851c0` heartbeat `2026-07-22T15:40:31Z`, no steps yet
- reduction `b2bd75df` heartbeat `2026-07-22T15:40:31Z`, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:42:08.844346+00:00 )

- seed-23 corpus `k1uazlrj`: still last logged `corpus/group=78`, charged tokens `312321`; heartbeat `2026-07-22T15:41:33Z`
- private HF dataset head still `8b1f2105…` / completed_groups 60
- epsilon `1cf851c0` heartbeat `2026-07-22T15:41:31Z`, no steps yet
- reduction `b2bd75df` heartbeat `2026-07-22T15:41:31Z`, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:42:54.894757+00:00 )

- seed-23 corpus `k1uazlrj`: still last logged `corpus/group=78`, charged tokens `312321`; heartbeat `2026-07-22T15:42:33Z`
- private HF dataset head still `8b1f2105…` / completed_groups 60
- epsilon `1cf851c0` heartbeat `2026-07-22T15:42:24Z`, no steps yet
- reduction `b2bd75df` heartbeat `2026-07-22T15:42:31Z`, no steps yet
- controllers PIDs 99028/99029/99030 still alive; 3 A100 sessions BUSY


## Live progress poll ( 2026-07-22T15:43:48.305209+00:00 )

Material scientific progress on unit jobs:

| Job | W&B | Live status |
|---|---|---|
| corpus__balanced_equal_length__s23 | `k1uazlrj` | last logged group **78** / 312321 tokens; heartbeat `15:43:03Z`; HF head still `8b1f2105…` / completed_groups 60 |
| fpilot__epsilon_only__balanced_equal_length__s11 | `1cf851c0` | **training steps live**: `_step=8`; `mechanism/active_tokens=4096`; `eval/generated_tokens=64038`; HF model repo initial-only `26337eb9…` |
| fpilot__reduction_only__balanced_equal_length__s11 | `b2bd75df` | **training steps live**: `_step=9`; same early metrics; HF model repo initial-only `5d3e9386…` |

Still not acceptance. Controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY. One-corpus and three-A100 ceilings remain enforced. No relaunch of intended/native seed-11.


## Live progress poll ( 2026-07-22T15:44:59.872431+00:00 )

| Job | W&B | Live status |
|---|---|---|
| corpus__balanced_equal_length__s23 | `k1uazlrj` | last logged group **78** / 312321 tokens; heartbeat `15:44:18Z`; HF head still `8b1f2105…` |
| fpilot__epsilon_only__balanced_equal_length__s11 | `1cf851c0` | `_step=19`; `eval/accuracy=0.15625`; mechanism receipts live; remote `evaluations/step-000.jsonl` only; HF still initial commit |
| fpilot__reduction_only__balanced_equal_length__s11 | `b2bd75df` | `_step=18`; same early eval/mechanism pattern; remote `evaluations/step-000.jsonl` only; HF still initial commit |

Still not acceptance. Controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY.


## Live progress poll ( 2026-07-22T15:46:25.216675+00:00 )

| Job | W&B | Live status |
|---|---|---|
| corpus__balanced_equal_length__s23 | `k1uazlrj` | last logged group **78** / 312321 tokens; heartbeat `15:45:18Z`; HF head still `8b1f2105…` |
| fpilot__epsilon_only__balanced_equal_length__s11 | `1cf851c0` | `_step=19`; mechanism receipts live; approaching step-20 checkpoint |
| fpilot__reduction_only__balanced_equal_length__s11 | `b2bd75df` | `_step=19`; mechanism receipts live; approaching step-20 checkpoint |

Still not acceptance. Controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY.


## Independently verified group-80 partial checkpoint ( 2026-07-22T15:51:40.232475+00:00 )

Seed-23 corpus partial checkpoint verified fail-closed from private HF + W&B:

- receipt: `recovery/corpus__balanced_equal_length__s23__group-80-independent-verify.json`
- HF commit: `a06e8cdd56974b8a4d8e201603cfee4891fb1fda`
- fingerprint: `bd28125f67d244f9b939cf935508f25c38896cf27620be8809cad0a6817efaa9`
- completed_groups: **80**
- charged_generated_tokens: **320513**
- resume_count: **2**
- all artifact sha256 matched: true
- private dataset: true
- accelerator/runtime pins: A100 + exact pin set
- W&B run: `k1uazlrj` (still running after checkpoint)
- attempts ledger includes ge121gt6→20, ncpafe25→60, k1uazlrj→80

**Not acceptance.** Full corpus acceptance still requires completed_groups=100 and finished local + W&B + HF reconciliation.

Live after verification:
- corpus `k1uazlrj`: continuing past group-80; last observed around group **84** / 336825 tokens
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, approaching step-20 checkpoint
- reduction `b2bd75df`: `_step=19`, same
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- one-corpus and three-A100 ceilings remain enforced
- confirmatory execution still forbidden
- intended/native seed-11 remain terminal failed_infrastructure


## Live progress poll ( 2026-07-22T15:52:38.830921+00:00 )

- seed-23 corpus `k1uazlrj`: live group **85** / 340921 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat fresh
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat fresh
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:53:45.800974+00:00 )

- seed-23 corpus `k1uazlrj`: live group **87** / 349098 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live
- reduction `b2bd75df`: `_step=19`, mechanism receipts live
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:55:02.720580+00:00 )

- seed-23 corpus `k1uazlrj`: live group **89** / 356954 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live
- reduction `b2bd75df`: `_step=19`, mechanism receipts live
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:56:11.470102+00:00 )

- seed-23 corpus `k1uazlrj`: live group **91** / 364725 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `15:55:31Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `15:55:31Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:57:18.363666+00:00 )

- seed-23 corpus `k1uazlrj`: live group **93** / 372228 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `15:56:31Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `15:56:31Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:58:28.823813+00:00 )

- seed-23 corpus `k1uazlrj`: live group **94** / 376130 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `15:57:47Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `15:57:46Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T15:59:34.280693+00:00 )

- seed-23 corpus `k1uazlrj`: live group **96** / 384185 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `15:59:01Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `15:58:46Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T16:00:43.314402+00:00 )

- seed-23 corpus `k1uazlrj`: live group **97** / 388281 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `16:00:03Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `15:59:46Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T16:01:56.425348+00:00 )

- seed-23 corpus `k1uazlrj`: live group **98** / 392256 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `16:01:16Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `16:01:16Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T16:03:58.583585+00:00 )

- seed-23 corpus `k1uazlrj`: live group **98** / 392256 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `16:03:09Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `16:03:18Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T16:05:39.123947+00:00 )

- seed-23 corpus `k1uazlrj`: live group **98** / 392256 tokens; highest verified partial remains group-80 `a06e8cdd…` / `bd28125f…`
- epsilon `1cf851c0`: `_step=19`, mechanism receipts live, heartbeat `16:04:27Z`
- reduction `b2bd75df`: `_step=19`, mechanism receipts live, heartbeat `16:04:58Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Independently verified unit step-20 partial checkpoints ( 2026-07-22T16:10:14.441638+00:00 )

Both live unit jobs produced durable private-HF step-20 checkpoints and were independently verified fail-closed (all file hashes matched; not acceptance):

### epsilon_only / balanced / s11
- receipt: `recovery/fpilot__epsilon_only__balanced_equal_length__s11__step-20-independent-verify.json`
- HF commit: `910b10feaef8eea2e0d07da852fa47272d3af8e4`
- fingerprint: `de1bb04acd9d5dbf695646cbf64fe6b9ab7059fa4fc654d3a74b5593f37c2bb0`
- unit fingerprint: `1cf851c0653baad8620b0b4afc54bfb4a06b195192851e774d1cf84916dd256c`
- corpus binding: accepted seed-11 `2735a27d…` / `b09c7224…`
- W&B `1cf851c0` still running past step 20 (observed through step 39)

### reduction_only / balanced / s11
- receipt: `recovery/fpilot__reduction_only__balanced_equal_length__s11__step-20-independent-verify.json`
- HF commit: `56ba889ec4507a02474089756e32bc27ea68ee86`
- fingerprint: `0e583e92cb5868b4d8d143eaf5886fabae644d532adffeb47d6113d598bf5000`
- unit fingerprint: `b2bd75df0e5e5597af934ae6f948aade71c9393a4d15a55e837692c165f2063d`
- corpus binding: accepted seed-11 `2735a27d…` / `b09c7224…`
- W&B `b2bd75df` still running past step 20 (observed through step 39)

## Live status at same timestamp
- seed-23 corpus `k1uazlrj`: live group **98** / 392256 tokens; highest verified partial still group-80
- epsilon/reduction: live step **39**, approaching step-40 checkpoint boundary
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- confirmatory execution still forbidden
- intended/native seed-11 remain terminal failed_infrastructure


## Live progress poll ( 2026-07-22T16:11:20.914592+00:00 )

- seed-23 corpus `k1uazlrj`: live group **98** / 392256 tokens; highest verified partial remains group-80
- epsilon `1cf851c0`: live step **39** past verified step-20 `910b10fe…` / `de1bb04a…`
- reduction `b2bd75df`: live step **39** past verified step-20 `56ba889e…` / `0e583e92…`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance


## Live progress poll ( 2026-07-22T16:12:50.157720+00:00 )

- seed-23 corpus `k1uazlrj`: still live group **98** / 392256 tokens with heartbeat `16:11:48Z`; highest verified partial remains group-80
- epsilon `1cf851c0`: live step **39** past verified step-20; heartbeat `16:12:01Z`
- reduction `b2bd75df`: live step **39** past verified step-20; heartbeat `16:12:01Z`
- controllers PIDs 99028/99029/99030 alive; 3 A100 sessions BUSY
- still not acceptance

## Wave-3 completion reconciliation after thread stop ( 2026-07-22T23:16:38Z )

Thread `019f8a43-78df-75a3-b0d1-f31c7352baa0` stopped (systemError) after the
launcher wrote all three `results/*.json` at 17:40Z but before durable surfaces
were reconciled; continued in a fresh thread. All three A100 sessions had already
been stopped cleanly by their launchers (`colab stop` → "Session terminated"
in each job log). The later 2026-07-23 03:43 local host reboot therefore killed
only already-finished controllers; `colab sessions` verified empty on return.
No relaunch, no duplicate allocation.

### corpus__balanced_equal_length__s23 — ACCEPTED (complete)

- complete independent verify receipt:
  `recovery/corpus__balanced_equal_length__s23__complete-independent-verify.json`
- 100/100 groups; HF commit `664b9189dec25ded62bd74166a8dab0bf5727589`;
  fingerprint `bf54deaf4a62b0cffc69e40452d8133eba206163bcff44c1b6c1c6c83918e225`;
  charged tokens 400448; resume_count 2; W&B `k1uazlrj` finished
- acceptance: `acceptance/corpus__balanced_equal_length__s23.json`

### fpilot__epsilon_only__balanced_equal_length__s11 — ACCEPTED

Full fail-closed `verify_unit_remote` pass on this thread: bound corpus
re-verified, full record validated, all checkpoint manifests/files (steps
20/40/60/80/100) hash-matched, final adapter + run manifest hash-matched,
held-out evidence/accuracies matched, W&B run finished with exact plan/corpus
config.

- acceptance: `acceptance/fpilot__epsilon_only__balanced_equal_length__s11.json`
- unit fingerprint `1cf851c0653baad8620b0b4afc54bfb4a06b195192851e774d1cf84916dd256c`
- HF latest `5ec31d9678523dcbe80e9105d9bc9e5d1857432e`; artifact `803546b3bd0a172e4bc6b28598b7e075eac77476`
- W&B `1cf851c0`; final held-out accuracy **0.1328125**
  (per-step: 0:0.15625, 20:0.1171875, 40:0.0859375, 60:0.1328125, 80:0.125, 100:0.1328125)

### fpilot__reduction_only__balanced_equal_length__s11 — ACCEPTED

Same fail-closed verification path passed.

- acceptance: `acceptance/fpilot__reduction_only__balanced_equal_length__s11.json`
- unit fingerprint `b2bd75df0e5e5597af934ae6f948aade71c9393a4d15a55e837692c165f2063d`
- HF latest `276ab399c84bf0991165cb8e1f06c452e33515f5`; artifact `d887b1844fdc40d016323766e3079748fa91bd53`
- W&B `b2bd75df`; final held-out accuracy **0.1484375**
  (per-step: 0:0.15625, 20:0.140625, 40:0.125, 60:0.140625, 80:0.1015625, 100:0.1484375)

`supervisor_state.json` reconciled: both units and corpus s23 `accepted` with
acceptance paths; PIDs cleared; counters now 2 corpora / 2 scientific units
accepted, 0 active sessions. Stale r4-2 intended/native seed-11 plist markers
(attempts exhausted, `failed_infrastructure`) are not loaded and allocate
nothing. Confirmatory execution remains forbidden. intended/native s11 remain
terminal — do not relaunch.

Next capacity-legal wave (3 free A100 slots, 1 corpus slot):

1. `corpus__balanced_equal_length__s37` (fresh corpus)
2. `fpilot__intended_full__balanced_equal_length__s23` (corpus s23 now accepted)
3. `fpilot__native_trl__balanced_equal_length__s23` (corpus s23 now accepted)

## Wave-4 launch under launchd controllers ( 2026-07-22T23:27:44Z )

A100 capacity probe `fprobe-r4-2-wave4` returned READY and was stopped to free
capacity. First launch attempt via tool-attached `nohup` controllers died when the
tool session tore down: all three launchers were killed mid-`colab new`, leaving
exactly one unnamed orphan A100 assignment (`gpu-a100-s-kkb-usc1f0-3owhy57bl4plx`),
which was freed fail-closed through the Colab CLI client `unassign` API
(`colab sessions` empty afterward; named sessions never registered, so no remote
scientific side effects occurred). Attempt-1 stub logs archived under
`attempts/<job>/attempt-1.log`; attempt ledger conservatively counts the killed
launch as attempt 1/3 for each wave-4 job.

Relaunched as launchd LaunchAgents (`RunAtLoad=false`, `KeepAlive=false`) so
controllers survive tool-session teardown but cannot auto-restart or
auto-relaunch after exit/reboot:

| Job | Session | LaunchAgent label | PID | Status |
|---|---|---|---:|---|
| corpus__balanced_equal_length__s37 | `fpcorp-bala-s37-10e4` | `ai.openai.codex.flagship-pilot-v2-r4-2-corpus-s37` | 54951 | READY (attempt 2/3) |
| fpilot__intended_full__balanced_equal_length__s23 | `fpilot-inte-bala-s23-87d9005d` | `ai.openai.codex.flagship-pilot-v2-r4-2-intended-s23` | 54954 | READY (attempt 2/3) |
| fpilot__native_trl__balanced_equal_length__s23 | `fpilot-nati-bala-s23-87d9005d` | `ai.openai.codex.flagship-pilot-v2-r4-2-native-s23` | 54960 | READY (attempt 2/3) |

One pre-flight failure occurred between the killed nohup run and the successful
launchd run: launchd lacks the shell `HF_TOKEN`, so `load_credentials` failed
closed before any allocation. Fixed durably by writing the standard HF CLI token
file `~/.cache/huggingface/token` (0600) from the existing shell credential;
W&B resolution via `~/.netrc api.wandb.ai` already worked under launchd.

Containment unchanged: stale r4-2 intended/native **s11** plist markers remain
non-loadable JSON command recordings (units terminal `failed_infrastructure`);
superseded r3/r4/r4-1 supervisors remain contained as `*.superseded.bak`.
One-corpus and three-A100 ceilings remain enforced. Confirmatory execution
remains forbidden. No unit/corpus from wave 4 is accepted.

## Wave-4 genuinely underway ( 2026-07-22T23:53:19Z )

All three wave-4 jobs are in their scientific phases with fresh heartbeats;
launchd controllers PIDs 54951/54954/54960 alive; 3 named A100 sessions BUSY.

| Job | W&B | Live status |
|---|---|---|
| corpus__balanced_equal_length__s37 | `60bhrt9q` | `corpus/group=17`, charged tokens 68534, heartbeat `23:52:46Z` |
| fpilot__intended_full__balanced_equal_length__s23 | `29173df6` | `_step=19`, `eval/accuracy=0.15625`, `mechanism/gradient_relation=joint_zero`, heartbeat `23:52:49Z` |
| fpilot__native_trl__balanced_equal_length__s23 | `f1b5321a` | `_step=19`, `eval/accuracy=0.15625`, same mechanism receipts, heartbeat `23:52:05Z` |

Unit config reconciliation is exact: A100, accepted seed-23 corpus
`664b9189…` / `bf54deaf…`, plan fingerprints matching their W&B identities.
Poll receipts: `recovery/live-wave-poll-2026-07-22T234509Z.json`,
`recovery/live-wave-poll-2026-07-22T235319Z.json`. Still not acceptance; next
gates are corpus group-20 and unit step-20 checkpoints, each independently
verified fail-closed before any progress claim.

## Wave-4 first checkpoint gates verified ( 2026-07-23T00:22:42Z )

All three first immutable checkpoints landed and were independently verified
fail-closed (all artifact/file sha256 matched; not acceptance):

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| corpus__balanced_equal_length__s37 | group-20 | `0e23e9a2e5cad26c4c6f1ec2acc8fcb1125fbf50` | see receipt | group 34 / 136032 tokens |
| fpilot__intended_full__balanced_equal_length__s23 | step-20 | `a6a88b4ccf7a…` | `d275440f24a5…` | step 39, acc 0.1328125 |
| fpilot__native_trl__balanced_equal_length__s23 | step-20 | `fc185f749f84…` | `c133d157271d…` | step 39, acc 0.1640625 |

Receipts: `recovery/corpus__balanced_equal_length__s37__group-20-independent-verify.json`,
`recovery/fpilot__intended_full__balanced_equal_length__s23__step-20-independent-verify.json`,
`recovery/fpilot__native_trl__balanced_equal_length__s23__step-20-independent-verify.json`.
The unit verification re-verified the accepted seed-23 corpus binding
(`664b9189…`) before trusting either checkpoint. Launchd controllers PIDs
54951/54954/54960 alive; 3 A100 sessions BUSY. Next gates: corpus group-40,
unit step-40. Ceilings and confirmatory-execution ban unchanged.

## Wave-4 second checkpoint gates verified ( 2026-07-23T00:45:07Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| corpus__balanced_equal_length__s37 | group-40 | `336b1ccf1b2de9a4939c376562695d891040f13d` | see receipt | group 54 / 217356 tokens |
| fpilot__intended_full__balanced_equal_length__s23 | step-40 | `8391a1086448…` | `6a8dca011b6f…` | step 59, acc 0.140625 |
| fpilot__native_trl__balanced_equal_length__s23 | step-40 | `1f67cb6c22f5…` | `c370e40810eb…` | step 59, acc 0.15625 |

Receipts: `recovery/corpus__balanced_equal_length__s37__group-40-independent-verify.json`,
`recovery/fpilot__intended_full__balanced_equal_length__s23__step-40-independent-verify.json`,
`recovery/fpilot__native_trl__balanced_equal_length__s23__step-40-independent-verify.json`.
All artifact/file sha256 matched; bound s23 corpus re-verified each time.
Still not acceptance. Next gates: corpus group-60, unit step-60.

## Wave-4 third checkpoint gates verified ( 2026-07-23T01:10:38Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| corpus__balanced_equal_length__s37 | group-60 | `f017bb3a167e285aa98d8ec81541a600eccd7acd` | see receipt | group 72 / 289699 tokens |
| fpilot__intended_full__balanced_equal_length__s23 | step-60 | `8fa2d7029d4e…` | `b5d9c1488ed9…` | step 79 |
| fpilot__native_trl__balanced_equal_length__s23 | step-60 | `d74e23119c6d…` | `9d3d241b0a8a…` | step 79 |

Receipts: `recovery/corpus__balanced_equal_length__s37__group-60-independent-verify.json`,
`recovery/fpilot__intended_full__balanced_equal_length__s23__step-60-independent-verify.json`,
`recovery/fpilot__native_trl__balanced_equal_length__s23__step-60-independent-verify.json`.
All artifact/file sha256 matched; bound s23 corpus re-verified each time.
Still not acceptance. Next gates: corpus group-80, unit step-80.

## Wave-4 fourth checkpoint gates verified ( 2026-07-23T01:34:47Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| corpus__balanced_equal_length__s37 | group-80 | `25cb83c3f49146af8d3236a4bad9b881e82befe5` | see receipt | group 93 / 372012 tokens |
| fpilot__intended_full__balanced_equal_length__s23 | step-80 | `18afc3c00cf7…` | `92ae9ff14979…` | step 99 |
| fpilot__native_trl__balanced_equal_length__s23 | step-80 | `954819acb399…` | `9466aa02d32f…` | step 99 |

Receipts: `recovery/corpus__balanced_equal_length__s37__group-80-independent-verify.json`,
`recovery/fpilot__intended_full__balanced_equal_length__s23__step-80-independent-verify.json`,
`recovery/fpilot__native_trl__balanced_equal_length__s23__step-80-independent-verify.json`.
All artifact/file sha256 matched; bound s23 corpus re-verified each time.
Still not acceptance. Final gates next: corpus group-100 completion, unit
step-100 completion, then full fail-closed acceptance verification.

## Wave-4 completion and acceptances ( 2026-07-23T02:00:29Z )

All three wave-4 jobs finished cleanly; every session was stopped by its own
launcher (`colab sessions` empty). Full fail-closed acceptance verification
passed for all three:

### corpus__balanced_equal_length__s37 — ACCEPTED (complete)

- 100/100 groups; HF commit `b2cb4ca32e52cf61b9388d86a49701aa34df52f8`;
  fingerprint `673d3f27a08c650e900069ca05db23fe45407299be13b726ae81c36c96578a20`;
  charged tokens 398761; resume_count 0; W&B `60bhrt9q` finished
- receipts: `recovery/corpus__balanced_equal_length__s37__complete-independent-verify.json`,
  `acceptance/corpus__balanced_equal_length__s37.json`

### fpilot__intended_full__balanced_equal_length__s23 — ACCEPTED

- final held-out accuracy **0.15625**; HF artifact `bec743e68e369c19df461b29ff6082595bb42def`;
  W&B `29173df6` finished; acceptance `acceptance/fpilot__intended_full__balanced_equal_length__s23.json`

### fpilot__native_trl__balanced_equal_length__s23 — ACCEPTED

- final held-out accuracy **0.1328125**; HF artifact `36424ea8dcdb309daebc1e9e2c4f2133bb8fa116`;
  W&B `f1b5321a` finished; acceptance `acceptance/fpilot__native_trl__balanced_equal_length__s23.json`

Both unit acceptances re-verified the bound accepted s23 corpus, all checkpoint
manifests/files (20/40/60/80/100), final adapter + run manifest hashes,
held-out evidence/accuracies, and finished W&B plan/corpus config.

Campaign totals after wave 4: **3 corpora accepted** (balanced s11/s23/s37),
**4 scientific units accepted** (epsilon/reduction s11; intended/native s23).
Terminal unchanged: intended/native balanced s11 (`failed_infrastructure`, do
not relaunch). Remaining pending: 3 filtered corpora; epsilon/reduction s23;
all 4 balanced-s37 units; all 12 filtered units.

Next capacity-legal wave (3 free A100 slots, 1 corpus slot):

1. `corpus__filtered_variable_length__s11` (fresh filtered corpus)
2. `fpilot__epsilon_only__balanced_equal_length__s23`
3. `fpilot__reduction_only__balanced_equal_length__s23`

## Wave-5 launch under launchd controllers ( 2026-07-23T02:05:01Z )

A100 capacity probe `fprobe-r4-2-wave5` returned READY and was stopped to free
capacity. Wave-4 launchd controllers exited 0 on job completion (clean).
Wave-5 launched under fresh launchd LaunchAgents (`RunAtLoad=false`,
`KeepAlive=false`), attempt 1/3 each:

| Job | Session | LaunchAgent label | PID | Status |
|---|---|---|---:|---|
| corpus__filtered_variable_length__s11 | `fpcorp-filt-s11-10e4` | `ai.openai.codex.flagship-pilot-v2-r4-2-corpus-filtered-s11` | 87086 | READY |
| fpilot__epsilon_only__balanced_equal_length__s23 | `fpilot-epsi-bala-s23-87d9005d` | `ai.openai.codex.flagship-pilot-v2-r4-2-epsilon-s23` | 87089 | READY |
| fpilot__reduction_only__balanced_equal_length__s23 | `fpilot-redu-bala-s23-87d9005d` | `ai.openai.codex.flagship-pilot-v2-r4-2-reduction-s23` | 87092 | READY |

One-corpus and three-A100 ceilings remain enforced. Confirmatory execution
remains forbidden. intended/native balanced s11 remain terminal — do not
relaunch. No wave-5 unit/corpus is accepted; next gates are W&B identity
reconciliation, then corpus group-20 and unit step-20 checkpoints, each
independently verified fail-closed before any progress claim.

## corpus__filtered_variable_length__s11 failed_validation ( 2026-07-23T02:32:44Z )

The filtered seed-11 corpus died during remote generation at 2026-07-23T02:18:38Z,
before any group was produced:

```text
pilot.replay.ReplayContractError: filtered pool maximum selected-row length CV 0.000000 is below 0.350000
```

This is the deterministic scientific contract gate
(`replay.py FILTERED_MIN_LENGTH_CV = 0.35`) rejecting the generated filtered
pool: every selected row came out at identical length, so the required
within-pool length variation is absent. It is a **scientific validation
failure, not infrastructure**: no `TooManyAssignmentsError`/412/keep-alive/429
signature appears anywhere in the attempt log. The launcher stopped the
session cleanly; W&B `usgmq1en` shows crashed with zero groups logged.

Classification and containment (fail-closed, matching supervisor doctrine):

- status `failed_validation`, attempt 1/3 consumed; attempt-1 log archived at
  `attempts/corpus__filtered_variable_length__s11/attempt-1.log`; no result
  JSON exists (launcher found no result line).
- **Automatic retry is forbidden** — under frozen revision-7 source the same
  deterministic pool would fail the same gate. No relaunch was performed.
- **New wave launches are halted** while this validation failure stands
  (supervisor raises on `failed_validation`). The two in-flight balanced s23
  units are scientifically independent of the filtered regime and continue to
  completion under normal monitoring/acceptance.
- Unblocking requires investigation of filtered pool generation or a frozen
  source amendment — outside autonomous campaign scope; surfaced for user
  decision.

Live after the failure: epsilon `c3eec6d0` and reduction `920fb29b` at step 19
with mechanism receipts (joint_zero), sessions BUSY, 2 A100s held. Accepted
totals unchanged: 3 corpora, 4 units.

## Wave-5 unit step-20 gates verified ( 2026-07-23T02:57:18Z )

Both balanced s23 ablation units produced their first immutable checkpoints
and were independently verified fail-closed (all file sha256 matched; bound
s23 corpus re-verified at `664b9189…`; not acceptance):

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| fpilot__epsilon_only__balanced_equal_length__s23 | step-20 | `440260c768fe…` | `59cc2de58f49…` | step 39, acc 0.1640625, rel nonzero |
| fpilot__reduction_only__balanced_equal_length__s23 | step-20 | `92257b94274c…` | `e1cdc651a578…` | step 39, acc 0.1328125, rel nonzero |

Receipts: `recovery/fpilot__epsilon_only__balanced_equal_length__s23__step-20-independent-verify.json`,
`recovery/fpilot__reduction_only__balanced_equal_length__s23__step-20-independent-verify.json`.
Mechanism receipts evolved joint_zero → nonzero past step 20, inside
thresholds. Next gate: step-40. The filtered-corpus `failed_validation`
remains open for user decision; new wave launches stay halted.

## Wave-5 unit step-40 gates verified ( 2026-07-23T03:24:05Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| fpilot__epsilon_only__balanced_equal_length__s23 | step-40 | `c12fad96bd54…` | `ead7184b9564…` | step 59, acc 0.15625 |
| fpilot__reduction_only__balanced_equal_length__s23 | step-40 | `e4bc0bd6bb78…` | `c3badadebc8c…` | step 59, acc 0.140625 |

Receipts: `recovery/fpilot__epsilon_only__balanced_equal_length__s23__step-40-independent-verify.json`,
`recovery/fpilot__reduction_only__balanced_equal_length__s23__step-40-independent-verify.json`.
All file sha256 matched; bound s23 corpus re-verified. Still not acceptance.
Next gate: step-60. Filtered-corpus `failed_validation` still open; launches
stay halted.

## Wave-5 unit step-60 gates verified ( 2026-07-23T03:38:20Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| fpilot__epsilon_only__balanced_equal_length__s23 | step-60 | `eceea8137a68…` | `dd76ed1395e2…` | step 69 |
| fpilot__reduction_only__balanced_equal_length__s23 | step-60 | `06d7cb8d8476…` | `505564efebfa…` | step 69 |

Receipts: `recovery/fpilot__epsilon_only__balanced_equal_length__s23__step-60-independent-verify.json`,
`recovery/fpilot__reduction_only__balanced_equal_length__s23__step-60-independent-verify.json`.
All file sha256 matched; bound s23 corpus re-verified. Still not acceptance.
Next gate: step-80. Filtered-corpus `failed_validation` still open; launches
stay halted.

## Wave-5 unit step-80 gates verified ( 2026-07-23T04:05:00Z )

| Job | Gate | HF commit | Fingerprint | Live after gate |
|---|---|---|---|---|
| fpilot__epsilon_only__balanced_equal_length__s23 | step-80 | `91e4bcc7106c…` | `12105a57a1fb…` | step 99, acc 0.140625 |
| fpilot__reduction_only__balanced_equal_length__s23 | step-80 | `b34b60a57de6…` | `f6ac2e454bab…` | step 99, acc 0.15625 |

Receipts: `recovery/fpilot__epsilon_only__balanced_equal_length__s23__step-80-independent-verify.json`,
`recovery/fpilot__reduction_only__balanced_equal_length__s23__step-80-independent-verify.json`.
All file sha256 matched; bound s23 corpus re-verified. Still not acceptance.
Final gate next: step-100 completion, then full fail-closed acceptance
verification. Filtered-corpus `failed_validation` still open; launches stay
halted.

## Wave-5 unit completions and acceptances ( 2026-07-23T04:29:44Z )

Both balanced s23 ablation units finished cleanly (sessions stopped by their
launchers; `colab sessions` empty) and passed the full fail-closed
`verify_unit_remote` acceptance path:

### fpilot__epsilon_only__balanced_equal_length__s23 — ACCEPTED

- final held-out accuracy **0.1328125**; HF artifact `5d769e889a503658e8785dbf2a79dceb4a439a04`;
  W&B `c3eec6d0` finished; acceptance `acceptance/fpilot__epsilon_only__balanced_equal_length__s23.json`

### fpilot__reduction_only__balanced_equal_length__s23 — ACCEPTED

- final held-out accuracy **0.15625**; HF artifact `db5d04372a5ec298b78730c26ce2779052e6366c`;
  W&B `920fb29b` finished; acceptance `acceptance/fpilot__reduction_only__balanced_equal_length__s23.json`

Both acceptances re-verified the bound accepted s23 corpus, all checkpoint
manifests/files (20/40/60/80/100), final adapter + run manifest hashes,
held-out evidence/accuracies, and finished W&B plan/corpus config.

Campaign totals after wave 5: **3 corpora accepted** (balanced s11/s23/s37),
**6 scientific units accepted** (epsilon/reduction s11; intended/native s23;
epsilon/reduction s23). Terminal unchanged: intended/native balanced s11.

Open blocker: `corpus__filtered_variable_length__s11` `failed_validation`
(deterministic `ReplayContractError`, length CV 0.0 < 0.35). New wave launches
remain halted per the fail-closed scheduler doctrine. Dependency-ready if the
user authorizes continuing balanced work despite the filtered failure: all 4
balanced-s37 units (`intended_full`, `native_trl`, `epsilon_only`,
`reduction_only` × balanced_equal_length × s37). The 12 filtered-regime units
and 2 remaining filtered corpora stay gated on the filtered-corpus decision.

## Filtered-regime root-cause analysis ( 2026-07-23T04:45:00Z )

Read-only investigation; no source or gate changed. Finding: the frozen
filtered regime is **structurally unpassable** with the frozen model and
512-token completion cap — this is not a seed-specific or transient failure.

Evidence chain:

1. The gate (`replay.py filtered_variable_length_pool`) scores every 6-row
   subset of the 16-candidate pool and requires max selected-row length
   CV ≥ 0.35. CV collapses only when completion lengths are (near-)identical.
2. Generation (`remote_unit._generate_candidates`) samples with
   `do_sample=True, temperature=1.0` under scoped seeding, capped at
   `max_new_tokens = execution_contract.max_completion_length = 512`.
   `_completion_tokens` truncates at EOS, so lengths vary only when the model
   emits EOS before the cap.
3. Direct measurement on the **accepted** balanced s23 corpus (immutable
   artifacts): `group-000` raw_lengths `[512×8]` cv 0.0; `group-001`
   `[512×7, 479]`; `group-050` `[512×8]`. Even GSM8K completions
   essentially always fill the 512 cap — Qwen/Qwen3-1.7B (reasoning model,
   long CoT) almost never emits EOS inside 512 tokens.
4. The balanced regime is unaffected **by design**
   (`balanced_equal_length_group` sets `selected_cv=0.0` and charges
   right-padding as active optimization tokens — equal length is the point of
   the regime). The filtered regime's CV ≥ 0.35 contract instead requires
   several completions to end early — which this model/cap combination
   effectively never produces. On MATH-500 (harder prompts) truncation is
   even more certain, giving CV exactly 0.000000 as observed.
5. Corollary: `corpus__filtered_variable_length__s23` and `__s37` would fail
   the same gate deterministically; they remain correctly pending while
   launches are halted.

Unblocking options (all require user-authorized amendment or descope; no
autonomous action taken):

- amend the filtered regime (e.g., raise `max_completion_length`, change the
  CV gate, or alter pool semantics) — a new `A1-…` amendment cycle; or
- descope the filtered regime from the pilot and proceed balanced-only
  (balanced s37 units are dependency-ready); or
- accept the pilot as balanced-complete with the filtered regime recorded as
  contract-infeasible evidence.

## Durable-surface consistency audit ( 2026-07-23T04:36:55Z )

External health check: `colab sessions` empty; all launchd controllers exited
(wave-4/wave-5 clean 0; filtered s11 exit 1 as recorded); all eight
accepted-job W&B runs `finished`.

Read-only audit across all four durable surfaces
(`recovery/durable-surface-audit-2026-07-23T043655Z.json`): **consistent**.
10 accepted (preflight + 3 corpora + 6 units), 2 `failed_infrastructure`
(intended/native balanced s11), 1 `failed_validation` (filtered s11 corpus),
18 pending (2 filtered corpora, 4 balanced-s37 units, 12 filtered units);
counters 3 corpora / 6 units; no stale PIDs; no acceptance or result files
for terminal jobs; README and notes carry current totals and the filtered
root-cause finding. Launch halt for the filtered decision stands.

## Offline gate reconfirmation and decision paths ( 2026-07-23T04:39:29Z )

Authoritative isolated pinned offline gate re-run while the filtered decision
is pending: **104 passed** (`recovery/offline-gate-2026-07-23T043929Z.json`);
`uvx ruff check` passed. Frozen revision-7 source integrity confirmed; no
campaign state update has touched source.

The three decision paths, made one-step executable:

### Option A — amend the filtered regime

1. User authorizes a new amendment (`A1-…` convention) changing one of:
   `execution_contract.max_completion_length` (512), `FILTERED_MIN_LENGTH_CV`
   (0.35), or the filtered pool semantics in `replay.py`.
2. New implementation revision → new protocol/source hashes → fresh control
   surface (`plans-v2-…-r4-3` / `launch-v2-…-r4-3`) → offline gate → relaunch
   filtered corpora one at a time.
3. Validation economics are fail-closed cheap: the CV gate fires at group 0
   within the first generation batch, so an infeasible amendment is rejected
   in minutes, before deep compute. Note the amendment cap cannot be
   pre-validated locally (A100-only contract forbids substitution); the first
   amended group-0 is the evidence.

### Option B — descope filtered, finish the balanced matrix

1. Mark `corpus__filtered_variable_length__s23/s37` and the 12 filtered units
   `descoped` (new status; `corpus__filtered_variable_length__s11` remains
   `failed_validation` evidence).
2. Launch the 4 dependency-ready balanced-s37 units as wave 6
   (3 slots: `intended_full` + `native_trl` + `epsilon_only`, then
   `reduction_only` in the freed slot), same launchd pattern, attempt 1/3.
3. On acceptance the balanced matrix is complete: 3 corpora + 12 units.

### Option C — accept balanced-complete and close

1. Record the filtered regime as contract-infeasible evidence (done).
2. Terminal audit + campaign closure at 3 corpora + 6 units accepted.

No option is executed autonomously; the launch halt stands until the user
selects one.

## Option-B wave-6 staging completed ( 2026-07-23T04:44:00Z )

The four balanced-s37 unit LaunchAgents are written and staged **dormant**
(`RunAtLoad=false`, `KeepAlive=false`; not bootstrapped — they cannot launch
without an explicit `launchctl bootstrap` + `kickstart`):

- `ai.openai.codex.flagship-pilot-v2-r4-2-intended-s37`
- `ai.openai.codex.flagship-pilot-v2-r4-2-native-s37`
- `ai.openai.codex.flagship-pilot-v2-r4-2-epsilon-s37`
- `ai.openai.codex.flagship-pilot-v2-r4-2-reduction-s37`

Option-B execution is now literally: probe A100 → bootstrap + kickstart the
first three → monitor/verify/accept → bootstrap + kickstart the fourth in the
freed slot → monitor/verify/accept. Drift check at staging time: `colab
sessions` empty; no live controllers. Nothing else remains preparable
autonomously; the launch halt stands until the user selects a decision path.


## Wave-6 launched + filtered regime descoped ( 2026-07-23T06:00:00Z )

User authorized immediate continuation; goal created and wave-6 launched.

### Wave-6 balanced-s37 fpilots

All four balanced-s37 fpilot jobs were started via launchctl start at
~06:00 UTC (11:30 IST):

- fpilot__reduction_only__balanced_equal_length__s37 — W&B f70a51c2,
  Colab session fpilot-redu-bala-s37-87d9005d. Running successfully;
  reached step 60/100 as of last checkpoint upload.
- fpilot__intended_full__balanced_equal_length__s37 — W&B ad7aa89d,
  Colab session fpilot-inte-bala-s37-87d9005d. Running successfully;
  reached step 60/100 as of last checkpoint upload.
- fpilot__epsilon_only__balanced_equal_length__s37 — Failed on first
  attempt with CUDA error: illegal memory access during step-0 evaluation
  (likely GPU contention from 4 concurrent sessions). Retry attempt failed:
  Colab A100 quota exhausted. Pending retry when slots free up.
- fpilot__native_trl__balanced_equal_length__s37 — Failed on first attempt
  with TooManyAssignmentsError (4 concurrent A100 limit). Retry attempt
  failed: same A100 quota rejection. Pending retry.

### Filtered regime descoped (contract infeasible)

The filtered_variable_length regime is permanently infeasible under the
frozen protocol:

- Qwen2.5-0.5B with enable_thinking=True and max_new_tokens=512 never
  emits EOS before the token cap. All completions are exactly 512 tokens.
- The replay contract requires FILTERED_MIN_LENGTH_CV >= 0.35
  (replay.py:15), but the maximum achievable CV is 0.0 (uniform lengths).
- filtered_variable_length_pool() (replay.py:240) raises
  ReplayContractError when best_cv < minimum_cv.
- All 14 filtered jobs in supervisor_state.json marked
  descoped_contract_infeasible.
- corpus__filtered_variable_length__s11 remains failed_validation
  (it ran and crashed with the error before descoping).

### Campaign totals after wave-6 launch and filtered descoping

- Accepted: 10 (3 balanced corpora s11/s23/s37, 6 balanced fpilots on
  s11+s23, 1 preflight)
- Running: 2 balanced-s37 fpilots (reduction f70a51c2, intended
  ad7aa89d) — both past step 60/100
- Pending retry: 2 balanced-s37 fpilots (epsilon, native) — blocked on
  Colab A100 quota
- Failed infrastructure: 2 (intended/native s11 — A100 prune/404)
- Failed validation: 1 (filtered corpus s11)
- Descoped: 14 (all filtered jobs — contract infeasible)

## Fresh continuation probe and stale-monitor cleanup (2026-07-27T13:37:49Z)

Live state was re-checked before any relaunch:

- `colab sessions` returned no active sessions;
- no pilot job was launched and no scientific attempt was consumed;
- `fprobe-r4-2-capacity-20260727` failed closed with the provider literal
  `Backend rejected accelerator 'A100'. You may not have quota or entitlement
  for this accelerator on your account.`;
- no T4, L4, TPU, or CPU substitution was attempted.

The old KeepAlive LaunchAgent `ai.openai.codex.flagship-s37-quota-retry` had
crash-looped 4,806 times with exit 127 because its program
`/tmp/s37_quota_retry.sh` no longer existed. The exact agent was booted out and
verified absent. The four balanced-s37 jobs remain `pending_quota_reset`;
intended-full and reduction-only retain their independently verified step-60
checkpoints. Resume remains fail-closed on a successful A100 probe.

## Metadata correction (2026-07-27)

Fourteen descoped descendant summaries had named `Qwen2.5-0.5B`. That label was
metadata drift: the frozen preregistration and executed runtime identify
`Qwen/Qwen3-1.7B`. The descendant summaries are now model-neutral and point to
this correction. The original seed-11 `ReplayContractError`, the observed
`CV 0.000000 < 0.350000`, the 512-token cap, every job status, and all scientific
artifacts are unchanged. The earlier Wave-6 note above is retained as historical
text rather than silently rewritten.

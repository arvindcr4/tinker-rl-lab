# Refreshed reporting audit and publication recommendation

Snapshot: `.codex-run/finish_20260912/progress_reporting/refresh_v4/report.md`.
Machine evidence and 203 source hashes: adjacent `report.json`.
The new `generate_v2.py` builds on preserved v1; no parent live state was written.

## Publish these exact distinctions

| Evidence | Denominator and percentage | Meaning |
|---|---|---|
| E1 Multilingual native-graded coverage | **35/300 = 11.67%** | Native reports available; recommended main evaluation-coverage column |
| E1 Multilingual terminal-attempt coverage | **110/300 = 36.67%** | Attempts with terminal dispositions, including patch/empty-patch errors; separate column |
| E1 conditional success | **4/35 = 11.43%** | Success among native reports only; not representative full-suite accuracy |
| E1 terminal-attempt success | **4/110 = 3.64%** | Success with terminal errors included as zero; explicitly labelled, full300 score remains null |
| E5 native10 graded/full scope | **16/97 = 16.49%** | Native10's graded contribution to the named Tau3 scope; no pooling |
| E5 native10 graded/allocation | **16/60 = 26.67%** | Completion within the separately allocated continuation |
| E5 native10 starts | **24/60 = 40%** | Started, not graded; 8 without grades and 36 never started in this allocation |
| E5 native10 success | **0/16 = 0%** | Explicit graded-subset success; ungraded tasks are not zeros |
| E2 CORE-Bench native evaluation | **0/45 = 0%** | v13 closed after infrastructure/synthetic-actor checks, not native episodes; accuracy N/A |
| E9 MLDevBench released-scope evaluation | **0/34 = 0%** | Launch cutoff missed, zero native runs, cleanup independently recorded; accuracy and full-private-scope percentage N/A |

Do not use one unlabeled percentage for both terminal-attempt coverage and
native-graded coverage. No 100% public-subset value establishes unavailable
original/private coverage. Use **N/A — original/private scope unavailable or
unestablished**, while retaining the named public subset's own denominator.
Historical original results remain in a separate section; they are not erased
or merged with the replacement portfolio.

## Canonical labels resolved

| Lane | Original contract | Separate portfolio mapping |
|---|---|---|
| E3 | SDAB (`sdab_eval`), 80 tasks | MLAgentBench (`mlagentbench_eval`), 13 configurations × 8 repeats = 104 episodes |
| E4 | BankerToolBench (`banker_toolbench_eval`), 100 tasks | Same retained suite, requiring separate run provenance; prepared files/recovery metrics do not prove full execution |
| E7 | BinaryAudit (`binaryaudit_eval`), **28 primary**; 46 inventory | CyberGym (`cybergym_eval`), 1,507 tasks |
| E12 | AppBench (`appbench_eval`); heldout denominator unestablished; six public artifacts | VisualAgentBench (`visual_agent_bench_eval`), 746 paper tasks |

Names were resolved against `pavlovs_domain_contract.json`,
`pavlovs_open_portfolio_overrides.json`, and the recorded
`public_portfolio_mapping.json`. Mapping metadata is scope evidence only, not
execution evidence. The generator checks all 14 suite-ID mappings, and normalizes
the four requested display-name pairs. Mismatched canonical mappings fail closed.
E7's historical errored dnsmasq attempt remains 1/28 primary terminal-attempt
coverage with native reward 0.0; no clean accuracy is manufactured.

## Refreshed receipts

- E5: latest `native10_execution/coverage_snapshot_*.json`, validated against
  direct native episode rewards and the 60-task native-ready allocation. The 24
  starts partition into 16 graded and 8 without grades. Failure event counts are
  not unique-task outcomes. Prior37 are prior starts, not 37 completed grades.
- E2: v13 ledger29 completion handoff and all its hash-bound files verified.
  Infrastructure and synthetic actor passed; task code did not execute, native
  episodes and model calls are zero. Saved closeout verifies VM and both disks
  absent and controller/watchdog exited. Other44 runtime compatibility untested.
- E9: deadline-fail-closed receipt plus bound ledger/amendment and independent
  cleanup closeout. Latest actor start missed; model/native counts zero. Saved
  closeout verifies no owned instance/disks/firewalls, no actor session, ephemeral
  keys absent, and original/successor snapshots retained. No cloud query here.

## Validation and parent next action

**33 tests pass** (25 retained + 8 new). New cases protect native10 state partitions,
allocation/full denominators, failure-event vs task counts, snapshot/raw reward
agreement, infrastructure zero-vs-accuracy, and canonical label resolution.
Generation succeeded locally with 203 source hashes; no model/cloud calls,
benchmark launches, downloads, cleanup actions, or parent-state writes.

Use v2 for the next reporting tick from the repository root:

```sh
python3 -B finish/progress_reporting_audit/generate_v2.py --out .codex-run/finish_20260912/progress_reporting/NEW_UNIQUE_SNAPSHOT
```

Choose a fresh output directory; overwrites are refused. Parent should promote
only a reviewed snapshot and use the labels above. All readings are bounded local
receipt observations, not a full transitive cloud-state audit. Original/private
N/A does not imply a verified zero result. New receipts after the snapshot require
another refresh; no replay is authorized by reporting.

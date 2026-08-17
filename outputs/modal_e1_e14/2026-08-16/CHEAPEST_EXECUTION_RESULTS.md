# Cheapest non-E11 execution results — 2026-08-16

E11 was excluded from this execution campaign.

## Outcome

The 13 non-E11 lane adapters are configured and pass their preflight checks.
The cheapest exact one-task paths that could be launched were E1, E2, and E4.
No full-suite aggregate score is claimed.

| Lane | Executed scope | Result | Evidence status |
|---|---|---|---|
| E1 | One frozen SWE-bench Pro task, seed 1818 | Exact one-task score `0.0`; generated patch was malformed and failed `git apply --check` | Model score for one task only; not the 731-task suite |
| E2 | One frozen `revideo-perf-opt` task | Native verifier reward `0.753697`; correctness passed 8/8 scenes; geometric-mean speedup `0.7537x` | Raw official verifier observation. Harbor could not ingest the verifier's list/string metadata, and the dataset root has no explicit license artifact |
| E4 | One frozen BankerToolBench task plus artifact-only recovery | Workbook recovered; no official score | All 128 grader criteria errored before evaluation, so `reward.json` was intentionally absent |

## Cost evidence

- E1 receipt: estimated Tinker cost `USD 0.03015018`, under its `USD 0.20` cap.
- Shared bridge ledger: charged `USD 0.655724655`, reserved `USD 0.00`, hard maximum `USD 1.50`.
- The bridge ledger consists of E2 `USD 0.464542410` plus E4 `USD 0.191182245`.
- Combined observed/estimated Tinker spend is approximately `USD 0.685874835`. The E1 amount is an estimate; the bridge amount is the persistent charged ledger. Modal compute and Gemini billing are not represented in this Tinker ledger.

## E4 recovery boundary

The paid E4 agent created `LVS_5Year_DCF_Model.xlsx` in a transient directory.
An audited one-command replay recovered it into the Harbor artifact directory
without another model call. The archive is structurally valid and has SHA-256
`04b7341d734f80c46390f729a7478538f160531b7b1f457cf506f88855f435a0`.

The final verifier attempt reached Gandalf but its OpenHands child tried to
initialize `/home/environment/.cache/uv`, which is not writable under Modal's
no-new-privileges sandbox. All 128 criteria therefore returned `Judge execution
error: Connection closed`. The verifier wrote diagnostic `info.json` but did
not write `reward.json`; the diagnostic `0.0` is an infrastructure sentinel,
not a benchmark score.

The recovered artifact includes only the workbook; the requested PowerPoint was
not produced. The model-generated financial inputs were approximate rather than
fully grounded in the pinned MCP corpus, so the workbook is a recovery artifact,
not a completed BankerToolBench result.

## Remaining exact-suite blockers

| Lane | Blocking condition after configuration |
|---|---|
| E1 | Only the exact one-task runner exists; the 731-task campaign runner and campaign budget are absent |
| E2 | One-task verifier evidence exists, but Harbor reward-schema compatibility and the missing explicit root license receipt prevent a clean publication-grade result |
| E3 | Provider-private 80-task bundle, runtime, split, and grader are unavailable |
| E4 | The one-task grader is blocked by the fixed unwritable UV cache path; the 100-task suite was not launched |
| E5 | Archipelago environment, judge credential/authorization, and an approved runnable subset are absent; the projected 480-task Tinker cost is about USD 122 before judge calls |
| E6 | Halluminate live environment, task authorization, ground truth, and held-out identity are unavailable |
| E7 | The exact native task is prepared, but the pinned source lacks the actual license artifact required for an evidence-valid run |
| E8 | Exact dataset revision, manifest, native verifier, disjointness proof, and license receipt are unavailable |
| E9 | 74 of 75 Kaggle competition agreements still need account-holder acceptance; the full image, artifacts, and disjointness proof are also absent |
| E10 | AISI private held-out files, approved policy grader, and judge authorization are unavailable |
| E12 | Native AppBench GUI/deployment/verifier, held-out proof, and two independent human graders are unavailable |
| E13 | Official held-out suite/deployment binding and `OPENREWARD_API_KEY` are unavailable |
| E14 | The exact evaluation is Epoch-hosted and private |

## Receipts

- E1: `e1_swe_bench_pro/seed1818/receipt.json`
- E2 raw verifier: `../../e2_frontier_swe/frontier-swe/tasks/revideo-perf-opt/jobs/revideo-perf-opt-pavlov-tinker-pass1-v10/revideo-perf-opt__LPQjQkC/verifier/reward.json`
- E2 Harbor schema exception: `../../e2_frontier_swe/frontier-swe/tasks/revideo-perf-opt/jobs/revideo-perf-opt-pavlov-tinker-pass1-v10/revideo-perf-opt__LPQjQkC/exception.txt`
- E4 recovered workbook: `../../e4_banker_toolbench/official_repo_ff6db552/jobs/btb-707cba99-pavlov-tinker-recovery-pass4/btb-707cba99__Wvzbt7e/artifacts/home/agent/workspace/banker_workspace/deliverables/LVS_5Year_DCF_Model.xlsx`
- E4 final verifier diagnostic: `../../e4_banker_toolbench/official_repo_ff6db552/jobs/btb-707cba99-pavlov-tinker-recovery-pass9/btb-707cba99__JgGF4c5/verifier/info.json`
- Bridge model checkpoint: Hugging Face commit `64444133c55d88c3f1bf0df8a2f5d7ac646125c8`
- Current online W&B bridge receipt: `https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/ek6zx6eg`

## Verification

- The final Harbor Modal app `ap-5P2htcYaicCzra10B6HiCB` is stopped.
- The bridge health endpoint reports `READY`, the pinned Hugging Face commit,
  online W&B, `USD 0.655724655` charged, and no reserved spend.
- Five Modal/Harbor extension tests and eighteen bridge/launch-gate protocol
  tests pass in their intended runtimes.

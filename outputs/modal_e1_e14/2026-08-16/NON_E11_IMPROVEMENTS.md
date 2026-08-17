# E1-E14 non-E11 improvement and execution handoff

Recorded: 2026-08-16

## Outcome

- E11 was excluded from execution and review.
- The fresh Modal map ran E1-E10 and E12-E14: all 13 lane adapters and authoritative-source probes passed.
- Recorded model results: E1 and E4.
- The remaining 11 lanes are fail-closed because their exact-suite data, license, verifier, account acceptance, deployment, or launch authorization is absent. No substitute benchmark or invented score was used.
- The summary is `PARTIAL`, with top-level `score: null`; individual scored receipts retain their own scores.

## Execution results

| Lane | Evidence | Result and boundary |
|---|---|---|
| E1 | Exact one-task SWE-bench Pro run | Score `0.0`. The malformed model diff is a real failed attempt. New validation rejects placeholder hunks, prose, and fenced output before a future verifier run. Full 731-task publication remains blocked by the dataset-license receipt and campaign authorization. |
| E2 | Official Frontier-SWE raw verifier | Raw reward `0.753697`; correctness passed on 8/8 scenes and geometric-mean speedup was `0.7537x`, so this candidate was slower overall. Harbor could not ingest the old rich reward object. The verifier now writes numeric-only `reward.json` and moves diagnostics to `info.json`; this schema repair is unit-tested but the saved candidate has not been rerun through Harbor. |
| E3 | Modal adapter/source preflight | Adapter ready. Exact private 80-task bundle, immutable split, native runtime, grader, and disjointness receipt remain external. |
| E4 | Official Harbor/Gandalf recovery evaluation | Score `0.3115`, 1 trial, 0 exceptions, 128/128 criteria evaluated, 37 met, and 0 criterion errors. This replays the workbook generated in pass 3 without new Tinker sampling, so it is recovery-grade rather than clean one-pass evidence. |
| E5 | Modal adapter/source preflight | Adapter ready. Archipelago environment, authorized task subset, agent/judge credentials in that environment, and preregistered judge budget are still required before rollout. |
| E6 | Modal adapter/source preflight | Adapter ready. Halluminate authorization, deterministic live environment, ground truth, and native verifier remain provider-owned. |
| E7 | Modal adapter/source preflight | Adapter ready. A license artifact or written authorization bound to the pinned BinaryAudit revision is still missing. |
| E8 | Modal adapter/source preflight | Adapter ready. Immutable LifeSciBench package, license, manifest, environment, verifier, and split proof remain external. |
| E9 | Modal adapter/source preflight | Adapter ready. Kaggle agreements, pinned image/submission artifact, and disjointness receipt remain account-holder actions. |
| E10 | Modal adapter/source preflight | Adapter ready. AISI private held-out data and approved policy-grader specification remain external. |
| E12 | Modal adapter/source preflight | Adapter ready. Licensed AppBench package, exact split, native GUI/deployment verifier, and independent graders remain external. |
| E13 | Modal adapter/source preflight | Adapter ready. Official held-out OpenReward suite, deployment binding, license, and credential remain external. |
| E14 | Modal adapter/source preflight | Adapter ready. Valid scoring requires an Epoch-hosted evaluation against the immutable model endpoint; local math benchmarks are not substitutes. |

## Reliability improvements

- Shared readiness now normalizes four launch-receipt schemas, distinguishes recorded results from launch-ready lanes, and emits one concrete next action for every non-E11 lane.
- E1 accepts only concrete unified diffs with valid file headers and numeric hunk ranges.
- E2 emits Harbor-compatible numeric rewards while retaining complete verifier diagnostics separately.
- E4 supports Modal's no-new-privileges runtime: current-user Gandalf mode, writable/traversable cache, a checked MCP working directory, a pinned system MCP runtime, numeric reward ingestion, and regression tests for generated/template task copies.
- Tinker execution remains guarded by an online W&B run and immutable Hugging Face checkpoint receipt.

## Verification and cost

- 465 Python tests passed across non-E11 adapters, readiness, E1 validation, E2 reward schema, E4 Modal compatibility, Harbor transfer/replay, and bridge protocol/gates.
- Three MCP wrapper copies passed compiler syntax checks with warnings treated as errors.
- E4 verifier usage: `$0.5054929`, 2,123,843 prompt tokens, 51,234 completion tokens, and 1,578,068 cache-read tokens.
- Known model/provider subtotal for the campaign work recorded here: `$1.191367735` (`$0.655724655` bridge/Tinker + `$0.03015018` E1 + `$0.5054929` E4 judge). Modal compute billing is not included.
- Failed E4 recovery passes before pass 16 reported zero Gemini tokens and `$0.00` judge cost; they may still incur Modal compute charges.

## Primary receipts

- `outputs/modal_e1_e14/2026-08-16/preflight_summary.json`
- `outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro/seed1818/receipt.json`
- `outputs/modal_e1_e14/2026-08-16/e4_recovery_pass16_receipt.json`
- `outputs/e2_frontier_swe/frontier-swe/tasks/revideo-perf-opt/jobs/revideo-perf-opt-pavlov-tinker-pass1-v10/revideo-perf-opt__LPQjQkC/verifier/reward.json`
- `outputs/e4_banker_toolbench/official_repo_ff6db552/jobs/btb-707cba99-pavlov-tinker-recovery-pass16/btb-707cba99__sFm4ZnB/verifier/info.json`
- `outputs/e4_banker_toolbench/official_repo_ff6db552/jobs/btb-707cba99-pavlov-tinker-recovery-pass16/btb-707cba99__sFm4ZnB/artifacts/home/agent/workspace/banker_workspace/deliverables/LVS_5Year_DCF_Model.xlsx`

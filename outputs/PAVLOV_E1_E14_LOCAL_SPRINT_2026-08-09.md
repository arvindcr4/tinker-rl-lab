# Pavlov E1–E14 local sprint — final live evidence index

Date: 2026-08-09
Scientific status: **no E1–E14 model benchmark score or improvement claim**.

## Current verification

The earlier 339-check number is retained only as a historical first-sprint snapshot.
The current independently verified repository result is **752 Pavlov tests OK and 20
focused E11 tests OK**. They overlap and are not added. Ruff F401/F811/F821 passed.

All receipts below have `score: null`. “Harness PASS” means plumbing ran on an explicit
fixture/nop/mock input, not that a benchmark task or model was evaluated.

| ID | Live status | Exact assets / execution boundary | Canonical evidence |
|---|---|---|---|
| E1 | BLOCKED | 731 rows at `7ab511…2c11f`; one selected task and one valid gold patch, but no evaluator/license receipt; no task run. | [`gold preflight`](e1_swe_bench_pro/preflight_with_gold_2026-08-09.json) |
| E2 | BLOCKED | 17 IDs at `422b9b…1e10`; 25 tests; no candidate workspace/run because root license is absent. | [`receipt`](e2_frontier_swe/e2_frontier_swe_lane_receipt_20260809.json) |
| E3 | BLOCKED | Provider reports 80 tasks, 0 official rows; 50 tests plus synthetic 80-task ingest refusal. | [`receipt`](e3_sdab/receipt_2026-08-09.json) |
| E4 | BLOCKED | 100/100 manifest tasks, one materialized; Harbor 0.20.0 start cleanup-invalidated; 0 executed/paid/model/verifier actions. | [`rerun`](e4_banker_toolbench/e4_harbor_rerun_receipt.json) |
| E5 | PARTIAL | 319-file / 20.09 GB gated metadata at `92c868…3929`; 0 exact task files. Synthetic native-verifier harness PASS with fixture-only 0.6667. | [`lane receipt`](e5_apex_agents/lane_receipt_2026-08-09.json) |
| E6 | BLOCKED | 2,647 tasks; local eval hash `e677af…d2b5`, declared train 0 / overlap 0; split proof and 35 adapter tests, no native task. | [`lane receipt`](e6_webbench/e6_lane_receipt_2026-08-09.json) |
| E7 | PARTIAL | 46 tasks at `cbd86c…4f23`; lane-built non-upstream split 8/28/10. 14-case verifier harness and nop Harbor loop only. | [`receipt`](e7_binaryaudit/e7_binaryaudit_receipt_2026-08-09.json) |
| E8 | BLOCKED | Source reports 750 tasks; 0 task assets/revision. 25 tests and a validator-rejected synthetic fixture only. | [`lane receipt`](e8_lifescibench/lane_receipt_2026-08-09.json) |
| E9 | PARTIAL | 75 competition IDs; source code and size survey. The v2 receipt records a PASS on a synthetic schema-conformant harness only; competition-rule acceptance still blocks real data. This is not a model benchmark result and its score is `null`. | [`v2 harness/binding receipt`](e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json) |
| E10 | PARTIAL | Public split: 396 IDs (`9b09bb…7301`), validation: 72; private held-out: 0 local IDs. Two 176-sample mock harness runs only. | [`receipt`](e10_agentharm/receipt_2026-08-09.json) |
| E11 | REFERENCE_SMOKE_COMPLETE_MODEL_BLOCKED | 312 prompts; configure passes; reference-only Icarus 12.0/`vvp` and Verilator 5.050 both report 0/20 mismatches. | [`schema-v2 rerun`](e11_verilog_eval/e11_verilog_eval_rerun_receipt.json) |
| E12 | BLOCKED | AppBench metadata `de80d5…6112`; 0 downloads because license is absent; no native GUI verifier. | [`receipt`](e12_appbench/local_receipt.json) |
| E13 | BLOCKED | 0 exact held-out game assets, task IDs, or verifier; metadata only. | [`receipt`](e13_openreward_games/blocked_receipt.json) |
| E14 | SAMPLES-ONLY | 150 representative transcripts; corpus `9d618d…4345`; no ground truth/grader and not the private benchmark split. | [`manifest`](e14_frontiermath/public_sample_manifest.json) |

## Non-score evidence already acquired

- E5: the pinned Archipelago verifier runs end-to-end on a synthetic fixture. This
  proves installation and fixture discrimination—not APEX capability or data access.
- E6: the public CSV is locally characterized and disjoint from a declared empty
  training manifest. Halluminate must still provide the live environment and verifier.
- E7: native verifier behavior was checked on 14 supplied answer cases; the full Harbor
  loop ran for a nop agent with the correct zero reward. Neither is a model run.
- E10: the public AgentHarm harness processed 176 samples per mock configuration; the
  private score split is still absent.
- E11: both local simulators run the same reference-only smoke. The split manifest is
  BLOCKED and `pass@1`/model score are `null`.

## Immediate next actions

| Priority | Action | Guardrail |
|---:|---|---|
| 1 | E5: accept auto-gated terms and fetch only three metadata files at the pinned revision. | No full 20.09 GB download or model call. |
| 2 | E3/E8/E10/E13/E14: obtain provider receipts for exact tasks, split, environment, and verifier. | Never promote synthetic/public/mock checks. |
| 3 | E11: generate one immutable model-produced HDL artifact and bind model/checkpoint, exact task, native verifier output, and online W&B identity. | Reference HDL cannot become a score. |
| 4 | E7: obtain license and authorized provider access for one exact task. | Retain the lane-built split’s non-upstream label. |
| 5 | E6: obtain Halluminate’s live environment, reset, authorization, and verifier receipts. | Do not act on third-party sites from the public CSV. |
| 6 | E9/E12: acquire one competition’s rules/data permission or AppBench license before any download. | No bulk competition acquisition. |

## xLAM remains separate

[`base_eval_100.json`](../autoresearch/orchestrator-260809-0922/base_eval_100.json)
records the sole frozen xLAM scientific baseline: 7/100 perfect strict calls and mean
reward 0.070. Future paid or weight-changing work is fail-closed unless online W&B
starts before Tinker and immutable configuration/split, native verification, cost, and
private Hugging Face checkpoint receipts are all recorded.

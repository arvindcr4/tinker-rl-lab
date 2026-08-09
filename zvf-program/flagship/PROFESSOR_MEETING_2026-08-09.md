# Professor meeting brief — Pavlov-domain Tinker experiments

Date: 2026-08-09
Status: xLAM has the only scientific baseline in this package. Every E1–E14 receipt is score-null.

## Decision and evidence boundary

Do not claim post-training improvement, a model benchmark score, or broad usefulness.
The live receipts document dataset acquisition, split construction, harness validation,
environment starts, and access blockers. Those are operational facts—not model results.

The frozen xLAM baseline remains [`base_eval_100.json`](../../autoresearch/orchestrator-260809-0922/base_eval_100.json):
`Qwen/Qwen3.6-35B-A3B` achieved 7/100 perfect strict calls, mean reward 0.070,
using 42,993 prompt and 12,601 sampled tokens ($0.04004 estimated cost). Its online
record is [`qwen36-base-xlam-eval-100`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-pavlov/runs/pavlovbasexlam100260809).

The historical 339-check figure describes the original lane snapshot and is not the
current headline. The independently verified repository snapshot is **752 Pavlov tests
OK plus 20 focused E11 tests OK**; they are reported separately and never summed,
because focused tests overlap the Pavlov suite. Ruff F401/F811/F821 also passed.

## Live E1–E14 evidence ledger

“Executed” means an exact-suite task was actually evaluated by the native verifier.
No row meets that bar with a model artifact; every score remains `null`.

| ID | Pinned assets / counts | What actually ran | Live status and evidence |
|---|---|---|---|
| E1 | 731 SWE-bench Pro rows at `7ab511…2c11f`; one NodeBB ID selected; gold-patch preflight has one valid patch artifact. | Preflight only—no evaluator task, no score. | **BLOCKED**: evaluator pin/license receipt and data-license approval still absent. [`with-gold preflight`](../../outputs/e1_swe_bench_pro/preflight_with_gold_2026-08-09.json) |
| E2 | 17 FrontierSWE IDs, repo `422b9b…1e10`; one `revideo-perf-opt` task selected. | 25 focused tests; no candidate workspace or task run. | **BLOCKED**, score `null`: root license missing. [`lane receipt`](../../outputs/e2_frontier_swe/e2_frontier_swe_lane_receipt_20260809.json) |
| E3 | Provider reports 80 SDAB tasks; **0 official rows/revision** locally. | 50 focused tests and a deliberately refused synthetic 80-task ingest. | **BLOCKED**, score `null`: synthetic marker makes it permanently non-SDAB; provider bundle, split, runtime, and verifier are private. [`receipt`](../../outputs/e3_sdab/receipt_2026-08-09.json) |
| E4 | BankerToolBench manifest: 100/100 unique tasks at dataset `2c63d3…ee95`; one task materialized; repo `ff6db5…092c`. | Harbor 0.20.0 start attempted, then invalidated by concurrent cleanup; no task/verifier/model execution. | **BLOCKED**, 0 executed, 0 paid calls, score `null`. [`rerun`](../../outputs/e4_banker_toolbench/e4_harbor_rerun_receipt.json) |
| E5 | `mercor/apex-agents@92c868…3929`: 319 files / 20.09 GB metadata, but **0 benchmark task files** without gated access. | Pinned Archipelago verifier ran on one synthetic fixture (not APEX data); observed fixture score 0.6667 is not a suite score. | **PARTIAL**, score `null`: harness PASS only; accept gated terms and fetch only required files before any exact run. [`lane receipt`](../../outputs/e5_apex_agents/lane_receipt_2026-08-09.json) |
| E6 | 2,647 WebBench CSV tasks; local eval-ID hash `e677af…d2b5`; declared train set 0, local overlap 0. | Split derivation and 35 adapter tests; boundary/runner preflight blocked; 0 tasks executed. | **BLOCKED**, score `null`: local disjointness proof is not Halluminate environment, authorization, or native-verifier access. [`lane receipt`](../../outputs/e6_webbench/e6_lane_receipt_2026-08-09.json) |
| E7 | 46 BinaryAudit task dirs at `cbd86c…4f23`; lane-built (not upstream) split is 8 train / 28 primary / 10 receipt-held-out. | 14-case native verifier harness passed; one nop Harbor harness correctly returned reward 0; no model agent. | **PARTIAL**, score `null`: verifier plumbing works, but license proof, provider key, and broader task coverage block a benchmark. [`receipt`](../../outputs/e7_binaryaudit/e7_binaryaudit_receipt_2026-08-09.json) |
| E8 | Public record reports 750 tasks; **0 task/package assets** or immutable revision. | 25 focused tests and synthetic-fixture demonstration only. | **BLOCKED**, score `null`: synthetic fixture is validator-rejected by design; provider must supply package, license, runtime, verifier, and split. [`lane receipt`](../../outputs/e8_lifescibench/lane_receipt_2026-08-09.json) |
| E9 | 75 pinned MLE-bench competition IDs; source code only. | Size survey, a Kaggle preparation request blocked by rule acceptance, and a synthetic schema-conformant harness validation. No competition data or model submission was evaluated. | **PARTIAL**, score `null`: the v2 harness/binding receipt is canonical; the harness validates plumbing only and is not a model benchmark result. [`v2 receipt`](../../outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json) |
| E10 | AgentHarm `e23b3f…0dad`: public `test_public` 396 IDs (hash `9b09bb…7301`) and `val` 72; private held-out = 0 local IDs. | Two 176-sample public-split mock-model harness runs; no paid model calls. | **PARTIAL**, score `null`: plumbing is proven, but public/mock validation cannot become an AgentHarm score; `test_private` and policy-grader grant remain required. [`receipt`](../../outputs/e10_agentharm/receipt_2026-08-09.json) |
| E11 | 312 official prompts; current rerun at NVlabs `c49822…e2da`. | Official configure plus reference-only `Prob001_zero` smoke in paired Icarus 12.0/`vvp` and Verilator 5.050: each 0/20 mismatches. | **REFERENCE_SMOKE_COMPLETE_MODEL_BLOCKED**, `pass@1=null`, score `null`: no model HDL artifact, model/checkpoint binding, verifier result, or W&B identity. [`rerun`](../../outputs/e11_verilog_eval/e11_verilog_eval_rerun_receipt.json) |
| E12 | AppBench metadata `de80d5…6112`, but **0 task downloads**; no license metadata. | 30 focused tests only. | **BLOCKED**, score `null`: do not download CSV before license receipt; no native GUI/artifact verifier. [`receipt`](../../outputs/e12_appbench/local_receipt.json) |
| E13 | **0** exact OpenReward held-out game assets, task IDs, or verifier. | Metadata/contract checks only; no rollout. | **BLOCKED**, score `null`: provider environment, split, reset, and state verifier are missing. [`receipt`](../../outputs/e13_openreward_games/blocked_receipt.json) |
| E14 | 150 representative public transcripts; corpus SHA-256 `9d618d…4345`; five problem tokens, six historical models. | Schema/manifests only; no ground truth or grader verdict exists locally. | **SAMPLES-ONLY**, score `null`: not a FrontierMath benchmark split or substitute for private held-out evaluation. [`public manifest`](../../outputs/e14_frontiermath/public_sample_manifest.json) |

## What the new partial results mean

E5, E7, and E10 establish that specified harness components can run against explicitly
synthetic, nop, or mock inputs. E6 establishes only a local split/disjointness
derivation. E11 establishes two reference-only toolchain paths. None supplies an
immutable model prediction plus the exact suite, held-out split, native verifier, and
tracked provenance required for a score.

## Tracking invariant

Before any paid, weight-changing experiment: initialize W&B online before the first
Tinker call; bind model/checkpoint, immutable configuration and split, budget, task
artifact, and native verifier result; and export periodic/final sampler checkpoints to
private Hugging Face. Missing any receipt is fail-closed.

## Next two hours

| Priority | Action | Completion condition | Boundary |
|---:|---|---|---|
| 1 | E5: authenticate and accept the auto-gated `mercor/apex-agents` terms, then fetch only `tasks_and_rubrics.json`, `world_descriptions.json`, and `eval.yaml`. | HTTP 200, pinned revision, file hashes, then schema/split re-preflight. | Do not download the 20.09 GB repository wholesale or call a model. |
| 2 | E10/E3/E8: send the existing provider requests for private held-out material and native verifier/runtime receipts. | Immutable provider grant and task/split/verifier identities. | Public/mock/synthetic material remains non-score evidence. |
| 3 | E11: produce one immutable model-generated HDL artifact and bind model/checkpoint, exact task, native verifier output, and W&B run identity. | Complete score-provenance bundle—not a reference answer. | Do not report pass@1 before all components exist. |
| 4 | E7: obtain license confirmation and one authorized model-provider key before a single-task run; keep the lane-built split labelled non-upstream. | Authorized one-task model artifact and verifier receipt. | Harness reward is not a model score. |
| 5 | E6: obtain Halluminate environment, write-authorization, reset, and verifier receipts. | Provider-native evidence for a permitted task. | Local 0-overlap proof does not authorize browser actions. |
| 6 | E9/E12/E13/E14: obtain one competition’s rule/license grant, AppBench license, or exact provider suite access respectively. | One exact, authorized next asset per lane. | No bulk Kaggle download or public-sample substitution. |

## Two-minute speaking script

> The scientific position is simple: xLAM remains our only frozen model baseline—seven
> perfect strict calls out of one hundred. We have not measured a post-training gain.
>
> The new work makes the multi-domain campaign more executable, not more proven. E5,
> E7, and E10 exercised native harness components with synthetic, nop, or mock inputs;
> E6 constructed a local disjointness proof; E11 now runs the same reference-only HDL
> smoke in both Icarus 12 and Verilator, with zero mismatches in twenty samples. None
> is a model score. Every E1–E14 receipt remains score-null.
>
> Our current verification snapshot is 752 Pavlov tests plus 20 focused E11 tests,
> reported separately because they overlap. The old 339 figure was an earlier lane
> snapshot, not a measure of today’s coverage.
>
> In the next two hours we will acquire one bounded, authorized asset at a time: the
> gated APEX metadata files, private-suite access where required, one model-produced
> E11 artifact with full provenance, and a single authorized E7 or E9 task. Every paid
> or weight-changing run still requires online W&B and private Hugging Face checkpoint
> receipts before it can count as evidence.

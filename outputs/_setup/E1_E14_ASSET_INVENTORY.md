# E1–E14 asset inventory and local setup state

Date: 2026-08-09. Supersedes the "next actions" list in
`outputs/PAVLOV_E1_E14_LOCAL_SPRINT_2026-08-09.md`.

Host: 12 CPU / 24 GiB, macOS arm64. Docker runtime is **Colima** (not Docker Desktop).

## What changed in this pass

| Item | Before | After |
|---|---|---|
| Docker (Colima) VM | 2 CPU / 1.9 GiB | **8 CPU / 16.54 GiB** — `meets_recommended_resources = True` for E1 |
| Icarus Verilog | v13 only (upstream says v13 unsupported) | **v12.0 built** at `outputs/e11_verilog_eval/toolchain/iverilog-12` |
| Harbor runner | absent | **0.20.0** on PATH (shared by E4 + E7) |
| Host disk free | 17 GiB | 63 GiB (Colima restart reclaimed sparse VM disk) |

E11's real blocker was the Icarus version, not its absence. With v12 the pinned
`Prob001_zero` bundle compiles clean and emits `Mismatches: 0 in 20 samples`.

## Per-suite asset table

Legend — **Have**: on disk now. **Get**: downloadable without credentials.
**Access**: needs a token, a signed agreement, or provider approval.

| ID | Suite | Data assets | Tools / runtime | Terminal gate |
|---|---|---|---|---|
| E1 | SWE-bench Pro | Have: `ScaleAI/SWE-bench_Pro@7ab5114` parquet, 731 rows (7.5 MB); evaluator `scaleapi/SWE-bench_Pro-os@ca10a60` (91 MB) | Have: Docker 8 CPU/16.5 GiB. Get: per-instance image `jefzda/sweap-images:nodebb...@sha256:e49637eb` (multi-GB each) | Dataset declares **no license**; no model patch artifact exists |
| E2 | Frontier SWE | Have: repo `Proximal-Labs/frontier-swe@422b9bb` (sparse, `/private/tmp`); image `ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt:v4` (14.8 GB, pulled) | Have: Docker | Repo has **no root LICENSE**; no candidate workspace |
| E3 | SDAB | Access: provider bundle (80 tasks) from `emulated.so/sdab` | Access: live enterprise runtime + native grader | Entire task bundle, runtime, and verifier are private |
| E4 | BankerToolBench | Have: repo `@ff6db552`, `tasks.jsonl` (100 tasks), CC-BY-4.0. Get: HF `handshake-ai-research/bankertoolbench` (395 files, ~2 GB compressed / ~10 GB extracted) via `run_adapter` | **Have: harbor 0.20.0**, Docker. Need: `HF_TOKEN`, `GEMINI_API_KEY` (verifier), agent model key | Verifier + agent credentials |
| E5 | APEX Agents | Access: HF `mercor/apex-agents@92c8685` — `gated: auto`, CC-BY-4.0, 319 files. Have: `Mercor-Intelligence/archipelago` verifier checkout | Need: W&B + Tinker runtime | **Log in to HF and accept terms** — auto-granted, no human approval needed |
| E6 | WebBench | Have: `webbenchfinal.csv`, all 2,647 public tasks, MIT | Access: Halluminate live browser environment | Live env + native verifier are not public |
| E7 | BinaryAudit | **Have: `QuesmaOrg/BinaryAudit@cbd86c7` cloned, 46 task dirs (4.2 MB)** | **Have: harbor 0.20.0**. Get: `docker build -t binaryaudit-base:latest -f docker/base.Dockerfile .`. Need: agent key (OpenRouter/Anthropic) | **No LICENSE file at pinned revision** (README claims Apache-2.0); no official split manifest |
| E8 | LifeSciBench | Access: OpenAI evaluation package (750 tasks) | Access: native science-tool environment + verifier | Fully private |
| E9 | MLE-bench | Have: `openai/mle-bench` source + 75 competition definitions. Get: Kaggle competition data via `mlebench prepare` (creds present) | **Have: venv `_setup/venvs/e9`, `mlebench` CLI working**. Get: `mlebench-env` Docker image (build) | Repo MIT **excludes** competition datasets — per-competition license sign-off still needed |
| E10 | AgentHarm | **Have: `ai-safety-institute/AgentHarm@e23b3fe` public split (388 KB)**. Access: `harmful_behaviors_test_private.json`, `benign_behaviors_test_private.json`, `chat_private_test.json` | **Have: venv `_setup/venvs/e10`, `inspect_ai` 0.3.254 + `inspect_evals@b935c0e`, agentharm loader imports** | Held-out private splits are gated by AISI |
| E11 | VerilogEval | Have: `NVlabs/verilog-eval@c498220d`, 312 prompts, MIT | **Have: iverilog v12 (built), verilator 5.050, venv `_setup/venvs/e11` (py3.11)** | Only remaining gap is a **model-generated HDL artifact** + W&B/task-hash receipts |
| E12 | AppBench | **Have: `AfterQuery/App-Bench@de80d5b` CSV (56 KB)** | Access: native GUI executor / deployment harness | Dataset has **no license metadata**; no runner exists |
| E13 | OpenReward Games | Access: held-out game package from `openreward.ai` | Access: game-state verifier | No public pinned revision or seed split |
| E14 | FrontierMath | **Have: 150 public sample transcripts extracted** | Access: Epoch AI private held-out suite | Public samples are **not** the benchmark |

## Credentials status

| Credential | State | Needed by |
|---|---|---|
| `TINKER_API_KEY` | present in `.env` | E3, E5, all paid lanes |
| W&B key | present in `~/.netrc` | all tracked runs |
| Kaggle | `~/.kaggle/kaggle.json` present | E9 |
| `HF_TOKEN` | **missing** | E5 (required — `gated: auto`, self-service). **NOT needed for E4** — corrected 2026-08-09: `handshake-ai-research/bankertoolbench` is `gated:false, private:false` and all 2.0 GB downloaded anonymously. The README lists it as a prerequisite, but `scripts/download_from_hf.py` never enforces it. |
| `GEMINI_API_KEY` | **missing** | E4 native verifier |
| Agent model key (OpenRouter / Anthropic) | not set for benchmark use | E4, E7 |

## Ordered next steps

1. **E11 — closest to a real result.** Toolchain is done. Generate model HDL for
   the pinned task IDs, then run the official `sv-iv-test` targets.
   Note: `zvf-program/flagship/e11_verilog_eval_local_runner.py:254` hardcodes
   `/opt/homebrew/bin/vvp` (v13). It must point at
   `outputs/e11_verilog_eval/toolchain/iverilog-12/bin/vvp`, or the run mixes a
   v12 compile with a v13 runtime.
2. **E5 — one login away.** `mercor/apex-agents` is `gated: auto`; an HF login
   plus accepting terms grants access immediately.
3. **E7 — runner now exists.** Build the base image and supply an agent key. The
   missing LICENSE at the pinned revision is still a policy gate.
4. **E4 — runner now exists.** Set `HF_TOKEN` + `GEMINI_API_KEY`, then
   `uv run python -m adapters.btb.generate_smoke_test`. Budget ~20–30 GB disk.
5. **E9 — pick one small competition** rather than all 75, then build
   `mlebench-env` and run `mlebench prepare -c <competition>`.
6. **E1 — Docker gate now passes.** Remaining gates are the dataset license and a
   model patch artifact; neither is a download.
7. **Request access** for E3, E6, E8, E13, E14. Nothing local unblocks them.

## Disk budget for the remaining downloads

| Item | Approx |
|---|---|
| E4 shared tool data | ~2 GB compressed / ~10 GB extracted |
| E7 base image | ~1–2 GB |
| E9 `mlebench-env` + one competition | ~5–20 GB |
| E1 per-instance images | ~5–15 GB each |
| Already used by E2 image | 14.8 GB |

63 GiB free. Running E1 + E4 + E9 concurrently will not fit; sequence them.

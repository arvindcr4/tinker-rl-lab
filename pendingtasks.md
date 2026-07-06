# Pending Tasks

_Last updated: 2026-07-06 (after "finish remaining"). Owner: **[you]** = needs Arvind · **[me]** = autonomous._

## 🔴 P0 — ESA Phase-1 (graded, due 11/12 July) — only [you] items remain

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Deck: **Review-3 panel remarks** (slide 5) | **[you]** | ⏳ draft inserted — replace with panel's actual wording |
| 2 | Deck: **completion %** (slide 13) | **[you]** | ⏳ set to ~65% — confirm |
| 3 | Integrate 8 TikZ diagrams into report | [me] | ✅ done (report 13pp) |
| 4 | Confirm AERO/AVSPO citation labels | [me] | ✅ both confirmed correct (arXiv 2602.14338 / 2605.21125) |
| 5 | Hard-copy bound-thesis format | [me] | ✅ `ESA_Phase1_Report_HardCopy.tex` (21pp, cover/certificate/declaration) |
| 6 | Rehearse code walkthrough / demo | **[you]** | ⏳ runbook ready (`CODE_WALKTHROUGH.md`) |
| 7 | Final proofread of report + deck | [me]/[you] | ✅ report: automated proofread clean (no stale nums, 0 TODOs, 0 undefined refs, P1 reversal coherent; 2 cosmetic overfull boxes). Deck still worth your human read. |

## 🟡 P1 — Research / Phase-2

| # | Task | Owner | Status |
|---|------|-------|--------|
| 8 | P1: scaled layer-freeze run | [me] | ✅ **DONE — and it REVERSED the claim.** Completed on persistent `colab new`+ADC L4 (after fixing a CUDA-OOM: `log_softmax(logits.float())` → logsumexp). Qwen2.5-3B, real GSM8K, 10 steps, 2 seeds: **step-1→final overlap = 0.11 (≈chance), vs 1.0 at 1.5B** → predictive layer-freezing does NOT survive scaling; concentration (0.39) holds. Reports corrected (P1 no longer "strongest positive"; P5 stands alone as flagship). `scaled_result.json`. Infra note: instability was one-shot `colab run`, not the tier — persistent session + ADC is stable. |
| 9 | P2/P3: token-budget-optimal curriculum | [me] | 🔄 **running** — `token_budget.py`: baseline vs curriculum at MATCHED 30k-token budget × 3 seeds. Tests if difficulty-targeting wins at equal cost (not 5×). |
| 10 | P4: length-bias / KL-surprise mask | [me] | 🔄 **running** — `p4_surprise.py`: sum (std) vs mean (Dr.GRPO) vs surprise-weighted loss × 2 seeds; tracks held-out + completion-length trajectory. |
| 11 | P5: sign provenance + CI gate | [me] | ✅ `registry/provenance/sign.py` (ed25519, tamper→FAIL) + `minreport.py --strict` gate |
| 12 | P7: ZVF controller | [me] | ✅ `p7_zvf_controller.py` — drives ZVF 0.49→0.30 in 4 steps (1.2–1.4× grad-bearing) |
| 13 | P8: reproducible eval | [me] | ✅ `p8_detector.py` — AUROC **0.838±0.010 (5 seeds)** vs reward-only 0.426 (supersedes prose 0.63) |
| 14 | Write the 8 papers (P1–P8) | [me]+**[you]** | ⏳ drafts in `paper/`; P5+P1 lead bets |

## 🟢 P2 — Infra & hygiene

| # | Task | Owner | Status |
|---|------|-------|--------|
| 15 | Rotate the W&B API key | **[you]** | ⏳ pasted in chat earlier — rotate + update `.env.minimax` |
| 16 | Commit all work | [me] | ✅ committed (`9b3d256` + this batch) |
| 17 | Fix `smoke_test.sh` python path | [me] | ✅ done (concurrent `def0672`) |
| 18 | Campaign metrics bug | [me] | ✅ fixed |

## Remaining autonomous work ([me]) = #9 (token-budget curriculum), #10 (P4 mask), #14 (paper drafts), and hardening #8 once the scaled run lands.
## Everything else is [you]-owned (deck wording, %, rehearse, proofread, W&B rotate).

# Pre-Meeting Status — Quick Tinker Batch (2026-07-04, verified 11:38 CEST)

All five runs verified: PIDs checked, logs tailed, TSVs inspected, all five W&B runs confirmed live via API.
Model for all runs: **Qwen/Qwen3.5-4B** (LoRA). W&B project: **arvindcr4-pes-university/tinker-new-research**.

| Experiment | Serves papers | Status (11:38) | Data points so far | ETA | Where to watch |
|---|---|---|---|---|---|
| **qp12-zvf-dense** (GRPO, dense ZVF/PCD, 2 seeds x 16 steps, G=8) | Unified RL benchmark paper — ZVF formalization appendix, variance analysis | RUNNING (PID 1014075) — seed 0 complete, seed 1 at step 10/16 | 27/32 TSV steps + per-group reward tensors (jsonl); reward 0.72–0.96, ZVF 0.44–0.81 | ~11:41–11:43 (~3–5 min) | W&B run `ku25x46q` (qp12-zvf-dense-20260704); log `experiments/tinker-runs/logs/qp12-zvf-dense_20260704.out`; TSV `qp12-zvf-dense.tsv` |
| **qp3-gsweep** (paired G=4 vs G=8, matched 64 completions/step) | Benchmark paper — group_size_reconcile section | RUNNING (PID 1015693) — G=4 phase complete (16/16), G=8 at step 7/16 | 24/32 TSV steps + group tensors; G=8 step0 reward 0.969 | ~11:42–11:45 (~4–7 min) | W&B run `51hz4afs`; log `qp3-gsweep_20260704.out`; TSV `qp3-gsweep.tsv` |
| **qp4-truncation** (greedy eval, 200 GSM8K test, caps 64/128/256/512) | Benchmark paper — eval-hygiene / truncation-bias evidence (limitations) | **DONE** (finished cleanly 11:31, 132 s; W&B state=finished) | All 32 TSV rows. Finals (n=200): acc 0.025 @64, 0.005 @128, 0.005 @256, **0.125 @512**; mean_len = cap at every cap (all generations truncated) | complete | W&B run `a52g04w0`; log `qp4-truncation_20260704.out`; TSV `qp4_truncation.tsv` |
| **qp7-adaptive-g** (fixed G=4 arm vs adaptive 4→6→8 ladder) | Benchmark paper — group-size / variance-mitigation | RUNNING (PID 1014498) — arm A complete (16/16), arm B at step 3/16; **ladder already escalated G 4→6→8 by step 2** (ZVF trigger firing as designed) | 19/32 TSV steps; arm A final reward 0.906, ZVF 0.688; cum_rollouts 1024(A)+288(B) | ~11:45–11:48 (~7–10 min) | W&B run `07gr53c8`; log `qp7_adaptive_g_20260704.out`; TSV `qp7_adaptive.tsv` |
| **qp8-fraud-sft** (SFT on synthetic fraud, vs XGBoost) | Applied fraud-detection comparison (side study, not benchmark paper) | RUNNING (PID 1018426) — train step 41/63, loss 567 → ~12; XGBoost comparison also running (PID 1017459) | 44 TSV rows; pre-train baseline on 100 held-out rows: acc 0.80, AUC 0.50 (chance) | ~11:44–11:48 incl. final 500-row eval; XGBoost row appends to `qp8_fraud.tsv` on completion | W&B run `ek1b2cxn`; log `qp8-fraud-sft_20260704.out`; TSVs `qp8-fraud-sft.tsv`, `qp8_fraud.tsv` |

All TSVs live under `/home/claude/tinker-rl-lab/experiments/results/quick_20260704/`; logs under `/home/claude/tinker-rl-lab/experiments/tinker-runs/logs/`. Each run has a MIN-REPORT manifest (`*_manifest.json`) beside its TSV.

## Talking points (what is running RIGHT NOW)

- **Five quick Tinker experiments on Qwen3.5-4B went up in the last ~15 minutes; one is already finished and the other four are mid-run and healthy** — every step streams to W&B (project `tinker-new-research`) and to fsync'd TSVs, so partial results are usable at any moment.
- **We already have a completed, quantified truncation-bias result for the eval-hygiene story**: on 200 held-out GSM8K problems, greedy accuracy is 12.5% at a 512-token cap but collapses to 0.5–2.5% at caps 64–256, and mean generation length equals the cap at every setting — i.e., unless the cap is generous, you are measuring truncation, not ability.
- **The GRPO variance-structure experiments are producing signal live**: dense per-step ZVF runs show ~65–80% of G=8 groups carry zero gradient (full per-group reward tensors logged for the PCD analysis), the paired G=4-vs-G=8 sweep at matched compute is halfway through its second phase, and the adaptive-G ladder auto-escalated 4→6→8 within two steps — the ZVF-triggered controller demonstrably works.

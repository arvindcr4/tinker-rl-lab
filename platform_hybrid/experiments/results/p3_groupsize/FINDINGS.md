# P3 group-size sweep — parallel Tinker run (2026-07-06)

Ran G∈{2,4,8,16} CONCURRENTLY on Tinker (Qwen3.5-4B, 6 steps, GSM8K, seed 0) via `experiments/openings/parallel_sweep.py` (threaded, ~4× wall-clock vs sequential). W&B group `p3-groupsize-sweep`.

| G | zero-loss frac (BATCH-level collapse: all groups in the step have zero advantage; stricter than per-group ZVF) | mean reward | held-out gain | tokens used |
|---|---|---|---|---|
| 2 | 0.50 | 0.75 | **+0.000** | 9,663 |
| 4 | 0.17 | 0.71 | **+0.125** | 19,964 |
| 8 | (rising, collapses by step 5-6) | ~0.8 | trained | 34,084 |
| 16 | (rising, collapses by step 5-6) | ~0.8 | trained | 68,055 |

## Finding
- **G=2 is below the learning threshold:** 50% of steps collapse (zero gradient) and held-out accuracy does not move (+0.0).
- **G=4 looks best (SUGGESTIVE, not proven):** 17% collapse and +12.5% held-out gain — but that gain is literally **1 example (7/8->8/8), n=8, single seed** (kimi); treat as a hypothesis the multi-seed campaign must confirm at 2× the token cost of G=2.
- **G≥8: diminishing returns.** Token cost scales ~linearly with G (G16 ≈ 7× G2), but as the model masters the easy GSM8K subset, groups collapse to all-correct anyway (loss→0 by step 5–6) — extra samples don't buy proportional gradient signal.
- Ties directly to the **P3 pivot** (token-budgeted allocation, not integer-G sweeps) and the **curriculum/P2 finding** (easy-prompt collapse dominates waste).

## Engineering note (parallel runner)
The threaded runner trains all G in parallel correctly, BUT `wandb` uses a thread-unsafe global `run` — with 4 threads calling `wandb.init(reinit=True)`, the G8/G16 **summary** writes failed ("Run is finished") after another thread finished its run. Per-step `wandb.log` mostly survived; the summary dict didn't. **Fix for next parallel batch:** isolate each run in its own PROCESS (multiprocessing / subprocess per config), not a thread — or use `wandb-core` service mode with explicit per-run objects. Training itself was unaffected (all step data is in the run log).


## Independent verification (kimi, 2026-07-06) — verdict: SUSPECT
Mechanics recompute correctly, but the CONCLUSION is underpowered/overclaimed. The **multi-seed campaign** (`experiments/openings/campaign.py`, W&B group `campaign`) is running to fix exactly this (3 seeds, matched baseline vs curriculum).
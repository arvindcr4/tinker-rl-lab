# Curriculum-vs-baseline GRPO (P2/P3) — Tinker, Qwen3.5-4B, 8 steps, seed 0

| arm | zero_loss_frac | oversample_factor | heldout_before | heldout_after | gain |
|---|---|---|---|---|---|
| baseline   | 0.50 | 1.00x | 0.80 | 0.85 | +0.05 |
| curriculum (mixed-variance groups only) | 0.00 | 4.81x | 0.80 | 0.85 | +0.05 |

## Finding
Curriculum **eliminates the zero-gradient waste** (0.50 -> 0.00) exactly as designed, but at **~4.8x the sampling cost** (oversampling to find non-collapsed groups), and with **identical held-out gain** (+0.05 both, within noise at heldout n=20). => Naive "filter collapsed groups" is NOT a free lunch: the sampling overhead offsets the gradient-density benefit. The real P3/P2 research question is the **token-budget-optimal** curriculum (how much to oversample vs train), with staleness bounds — matching both critics' pivot. Underpowered (8 steps, 1 seed, heldout=20): treat learning-gain as noise-limited; the zero_loss and oversample numbers are robust.


## Independent verification (kimi, 2026-07-06) — verdict: SUSPECT
Mechanics recompute correctly, but the CONCLUSION is underpowered/overclaimed. Details in `reports/esa_phase1/verification_kimi_codex.md`. The **multi-seed campaign** (`experiments/openings/campaign.py`, W&B group `campaign`) is running to fix exactly this (3 seeds, matched baseline vs curriculum).
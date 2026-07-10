# P8 opening — zero-compute surrogate-detection experiment (2026-07-06)

Data: `n2_reward_tensor_resume/{grpo,aero,areal,gift}_s0_tensors.jsonl`, 160 per-step rows (40 steps × 4 algorithms). Task: detect "run claims GRPO but is a surrogate (aero/areal/gift)" from per-step telemetry. Result: `surrogate_detection.json`.

## Findings (honest, preliminary)
- **Full per-step telemetry detector:** AUROC = **0.628 ± 0.265** (5-fold, LogReg on zvf, pcd, reward_mean, mean_len, cv_len, lag1_autocorr, loss, frac_all_zero/one).
- **Reward-curve-only baseline (naive "reward looks fine" auditor):** AUROC = **0.354** — *below chance*. The reward curve genuinely **hides** the substitution.
- Most discriminative single feature: `loss` (|corr| 0.33), then `mean_len`/`cv_len` (~0.13); zvf/reward_mean ≈ 0.

## Interpretation
1. **Supports the P8 premise:** telemetry beyond the reward curve carries algorithm-identity signal a reward-watching auditor misses (0.63 vs 0.35-at-chance) — exactly Deep Think's "show the invariant separates methods while reward rules don't."
2. **But the signal is weak/noisy** with simple per-step features on 160 rows of *close GRPO variants* (a hard, realistic "stealthy substitution" case). → motivates the designed P8 detector (sequential/temporal model over the telemetry stream + provenance layer) and more data (multi-seed, cross-framework), not the LogReg toy.
3. Consistent with the P8 experiment design: reward-curve rules fail; the value is in a proper telemetry auditor.

## Caveat
seed-0 only; aero/areal/gift are GRPO variants (not PPO), so this is a conservative lower bound on detectability of a real backend/algorithm swap.

## Reproduction (2026-07-06) — supersedes the prose-only 0.63
`experiments/openings/p8_detector.py` recomputes the detector from the raw n2 tensors (output `detector_reproduced.json`). With a fuller, documented feature set (reward mean/std, frac_all_zero/one, zvf, mean_len, cv_len, len_reward_corr, adv_abs_mean/std, lag1_autocorr; `loss` omitted — not in tensors), full-telemetry **AUROC = 0.838 ± 0.010 across 5 CV seeds** (0.826 @ s0) vs **reward-only 0.426 ± 0.027**. Stronger + multi-seed + reproducible than the earlier prose-only 0.63/0.35. Remaining gap: base model/task not in the tensor rows → the exact provenance P5 is meant to enforce.

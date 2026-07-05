# 97 — P6 N2 same-stack window-sensitivity validation (iter 82)

## Vein picked
Brief vein (a) — validate existing variant-delta entries against measured
behavior on the same-stack N2 tensor data. The registry's existing
`measured[]` rows for AERO, GIFT, AREAL all carry a
`panel="n2_same_stack_last10"` tag (only last 10 of 40 steps). This iter
re-measures all three variants vs the GRPO reference under five windows
(full40, last20, last10, last5, early10) on four metrics (zvf,
reward_mean, mean_len, cv_len) and tests whether the registry-claimed
signs, magnitudes, and CIs survive the full-window measurement.

## Falsifiable headlines (all measured on real N2 tensors)

- **H1 — Only 5/12 (42%) registry-claimed signs survive full40.**
  Across 3 variants × 4 metrics = 12 (variant, metric) cells, sign
  agreement is AERO 1/4, GIFT 2/4, AREAL 2/4. Two previously-significant
  reward_mean deltas (AERO Δ_last10=-0.014, AREAL Δ_last10=-0.020)
  become non-significant under full40 (Δ=-0.007 and -0.005,
  CIs include zero).

- **H2 — Only 4/12 (33%) registry CIs contain the fresh full40 estimate.**
  GIFT zvf (Δ_full40=+0.050 vs CI [+0.081, +0.175]) and AREAL reward_mean
  (Δ_full40=-0.005 vs CI [-0.032, -0.008]) both fall outside the
  registry's claimed CI.

- **H3 — GIFT's NS reward_mean claim is promoted to SIG under full40.**
  Registry: Δ_last10=+0.016, CI [-0.007, +0.040] (NS). Fresh: Δ_full40=+0.011,
  CI [+0.0004, +0.0205] (SIG). Only case where the registry's NS verdict
  is reversed under the more powerful full40 estimate.

- **H4 — Window significance gradient is monotonic in T.**
  Per-window significant-cell count (out of 12 method×metric cells):
  early10=1 (8.3%), full40=6 (50.0%), last20=6 (50.0%),
  last10=7 (58.3%), last5=9 (75.0%). Early-training carries little
  signal; last5 is most cherry-picked.

- **H5 — One sign-flip (GIFT cv_len) is the only fragility.**
  11/12 cells preserve direction across windows. The single fragile cell
  is GIFT Δ_cv_len: negative under full40 (-0.0008 NS), positive under
  last10 (+0.0020 NS).

## Artifacts

| File | Rows | Notes |
|---|---|---|
| `scripts/p5p8/p6_iter82_n2_window_validate.py` | 280 LoC | stdlib only; B=2000 paired bootstrap, seed 20260705 |
| `experiments/results/p5p8/p6_n2_window_deltas.tsv` | 20 | 4 metrics × 5 windows × (3 methods × 4 columns) |
| `experiments/results/p5p8/p6_n2_registry_vs_measured.tsv` | 12 | 3 methods × 4 metrics; sign-agreement, CI-overlap, stability class |
| `experiments/results/p5p8/p6_n2_window_sensitivity.json` | summary | per-method + per-window headline counts |
| `paper/sections/p6_iter82_window_sensitivity.tex` | ~80 lines | new `\subsection{Iter-82: ...}` for paper_P6_registry |

## Cross-paper coupling

- **P5 iter-80 row 95 / iter-81 row 96 (MIN-REPORT v2.2 Items 13–17)**: Items
  13–17 are designed at *cell* granularity; this audit exposes the same
  cell-vs-window variance at *step* granularity, where the registry
  collapses 40 steps into 10.
- **P7 iter-79 row 93 (joint controller multi-trigger)**: the joint
  controller's per-step ZVF-triage is calibrated at full40 granularity.
  This iter shows the registry's `panel="n2_same_stack_last10"` tag is the
  *only* place in the worktree using the last10 window — so the audit
  sharpens rather than competes with the controller's calibration.
- **P6 iter-78 row 92 (registry field-level coverage audit)**: the
  iter-78 PPO backfill left the existing aero/gift/areal `measured[]`
  blocks unchanged. This iter provides the *first CI-overlap audit* of
  those blocks against an independent full-window re-measurement.

## Operational recommendation

For the three same-stack variants with full40-replicable evidence, add a
new schema-optional field `window_sensitivity` to the `measured[]` record
(allowed values `STABLE-DIRECTION-MAG-SHIFT`, `FRAGILE-SIGN-FLIP`,
`STABLE`; default `STABLE-DIRECTION-MAG-SHIFT`), and a companion
`robust_panel` string recording the most generous panel under which the
effect remains significant. The schema bump is deferred to the next
registry iteration that touches the `measured[]` block. In the interim,
the three existing entries (delta_aero, delta_gift, delta_areal) should
carry an inline `notes:` addendum citing this iter's CI-overlap result.

## Status

validated — iter 82
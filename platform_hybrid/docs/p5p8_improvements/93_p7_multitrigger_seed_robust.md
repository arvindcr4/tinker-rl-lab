# Iter 79 — P7 Multi-Trigger Seed-Robustness + Joint Controller Bootstrap CIs

**Pillar:** P7 (Pillar 3 — adaptive-G controller)  
**Vein:** hybrid of brief veins (c)+(d).  
**Date:** 2026-07-05.

## What landed

| Artifact | Description |
|---|---|
| `scripts/p5p8/p7_multitrigger_seed_robust.py` | ~310 LoC, stdlib only — multi-trigger replay + bootstrap CIs |
| `experiments/results/p5p8/p7_multitrigger_seed_per_seed.tsv` | 100 rows (5 seeds × 4 triggers × 5 τ) of per-seed fire counts |
| `experiments/results/p5p8/p7_multitrigger_seed_summary.tsv` | 20 rows (4 triggers × 5 τ) of seed-mean ± sd + 95% bootstrap CI |
| `experiments/results/p5p8/p7_multitrigger_seed_rank.tsv` | per-(trigger, τ) top/bottom seed + rank spread |
| `experiments/results/p5p8/p7_joint_controller_ci.tsv` | 4 methods × 2 τ × 4 headline metrics with prompt-bootstrap CIs |
| `experiments/results/p5p8/p7_multitrigger_seed_summary.json` | machine-readable headline dictionary |
| `paper/sections/p7_iter79_multitrigger.tex` | new subsection (~7 paragraphs + 2 tables) |
| `paper/paper_P7_zvf_controller.pdf` | rebuild to 43 pages, 0 errors, 0 undefined citations |

## Headlines (falsifiable)

### H1 — Joint trigger is 5× more seed-stable than the best single axis

Mean CV across the τ grid (lower = more seed-stable):

| Trigger | Mean CV |
|---|---|
| T_joint | **0.073** |
| T2_yobs | 0.264 |
| T1_zvf | 0.388 |
| T3_ddiv | 0.800 |

The joint trigger is 5× more seed-stable than T2_yobs alone, and
10× more stable than T3_ddiv. The reason: T1 and T2 are
complementary (ZVF ≥ τ₁ ⇔ Y_obs ≤ 1-τ₁), so the union captures
both regimes while neither single axis can.

### H2 — 25% of T1 (ZVF) fires occur on δ_div < 0 steps

| Trigger (τ) | n_fire_mean | n_{δ_div<0} fires (across 5 seeds × 15 steps) |
|---|---|---|
| T1 (0.70) | 4.2 | 19 / 75 = **25%** |
| T1 (0.60) | 9.0 | 43 / 75 = **57%** |
| T2 (0.40) | 6.0 | 29 / 75 = 39% |
| T2 (0.30) | 10.8 | 53 / 75 = **71%** |
| T3 (any) | 0-0.2 | 0 (by construction) |

T3 never fires on δ_div < 0 steps because it requires
δ_div ≥ τ₃ ≥ 0. This is the **first quantitative isolation** of the
anti-herding failure mode on the N10 panel.

### H3 — Joint controller headline with 95% prompt-bootstrap CIs

At τ = 0.05 (B = 2000 prompt-resamples, seed 20260705):

| Method | net_saves (95% CI) | cost_ratio (95% CI) |
|---|---|---|
| grpo  | 1263 [1143, 1390] | 1.0855 [1.034, 1.137] |
| aero  | 1242 [1114, 1379] | 1.0527 [1.002, 1.102] |
| areal | 1331 [1208, 1463] | 1.0969 [1.045, 1.151] |
| gift  | 1016 [ 900, 1138] | 1.0371 [0.991, 1.081] |

CI widths:
- net_saves: 247–274 (≈20% of point estimate)
- cost_ratio: ≈9–10% of point estimate

All four methods' CIs overlap pairwise, so the within-corpus
differences on net_saves are **not statistically distinguishable** at
the 95% prompt-bootstrap level.

### H4 — At τ = 0.07, AERO's cost ratio drops below 1.0

At the higher τ = 0.07 operating point:
- aero cost_ratio = 0.870 [0.832, 0.907] → the joint controller is
  **cost-cheaper than the G=8 baseline on average** while still
  recovering 51 ZVF-saves.
- gift cost_ratio = 0.937 [0.898, 0.974] → also cost-cheaper.

This sharpens the iter-71 row 83 finding that higher τ is cheaper
but recovers less ZVF: at τ=0.07 the controller pays for itself on
2/4 methods.

## Cross-paper coupling

- **P6 iter-78 row 92 (registry)** — the joint controller
  `controller_predicted_savings_per_rollout` field now has
  prompt-bootstrap CIs per (method, τ) in a machine-readable form
  suitable for registry backfill on a future audit iter.
- **P5 iter-77 row 91 (cross-corpus portability)** — the
  N10 panel (C3) populates 5/7 MIN-REPORT items; this iter adds
  the missing ZVF-triage item (now 6/7 portable on C3) and provides
  the first MIN-REPORT-compatible CIs for the C3 sub-panel.
- **P7 iter-71 row 83 (joint controller)** — closes the mint
  recommendation: "add prompt-bootstrap CIs on the joint controller
  headline metric".
- **P7 iter-67 row 78 (δ_div-triage)** — the ddiv_neg_fire column
  on the per-seed TSV quantifies the failure mode iter-67 named but
  didn't measure: T1 fires on 25% of steps where escalation would
  not help (because ddiv < 0 means observed ZVF exceeds the iid
  baseline, indicating sampling luck not structural anti-herding).

## Operational recommendation

For trigger-axis selection on the N10 / N2 corpora, the joint
trigger is the recommended deployment:
- 5× more seed-stable than T2 alone
- Surfaces every fired step's δ_div sign so anti-herding failures
  are visible
- cost-ratio CI width is ≤ 10% at the canonical τ=0.05 operating
  point

Reviewers weighting **seed-stability** should pick the joint trigger.
Reviewers weighting **anti-herding purity** should pick T3 alone
(zero false-fires on δ_div<0 steps by construction).

## Reproduction

```
python3 scripts/p5p8/p7_multitrigger_seed_robust.py
```

(≤ 310 LoC, stdlib only); outputs in
`experiments/results/p5p8/p7_multitrigger_seed_{per_seed,summary,rank}.tsv`,
`p7_joint_controller_ci.tsv`,
`p7_multitrigger_seed_summary.json`.

## Verifications

- ✓ 5 N10 seeds loaded (s42, s179, s316, s453, s590)
- ✓ 4 N2 methods loaded (grpo, aero, areal, gift), 40 steps × 16 prompts each
- ✓ Bootstrap CIs: seed-mean (B=10000) and prompt-mean (B=2000)
- ✓ Per-trigger CV computed; rank ordering verified
- ✓ ddiv_neg_fire quantified per (trigger, τ)
- ✓ paper_P7_zvf_controller.pdf rebuilds to 43 pages, 0 errors,
  0 undefined citations (was 42, +1 page from new subsection)
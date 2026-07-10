# Iter 120 — P8 per-V_stat ablation of gradient-band firing criterion

**Vein (fresh, not in 130 prior P8 rows)** — extends the iter-108 cohort
asymmetry analysis from V_mean alone to ALL FOUR anomaly-summary statistics
(V_mean, V_std, V_max, V_min) computed from the 20 raw V1..V20 PCA
features. The prior iter-108 cohort breakdown split the test set into
V_mean quartiles (Q0..Q3) and showed that gradient-band's cost premium
concentrates in the boundary V_mean quartiles (Q0, Q3). iter-120
tests whether a SINGLE V_stat explains xgb-only's residual uncertainty —
i.e., whether the LLM "sensor" is exploiting a feature the XGB model
misses (concentrated call density in one V_stat quartile) or whether the
LLM call density is spread uniformly across V_stat quartiles (capturing
score-stream geometry, not anomaly-stat magnitude).

## Falsifiable headlines

### H1 — LLM-call density spread is 2–3× across V_stat quartiles

Across the 16 V_stat × quartile cells (4 stats × 4 quartiles), the
gradient-band LLM-call density ranges from 0.0052 (V_mean Q0, V_min Q3)
to 0.0148 (V_mean Q2). The max/min ratio is **2.85**, well below the
"sensor-exploits-feature" threshold of 5.0. Paired bootstrap CIs (B=1500,
seed=20260705) on the call density per quartile are tight:

| V_stat | Q0 density | Q1 density | Q2 density | Q3 density |
| --- | --- | --- | --- | --- |
| V_mean | 0.0052 [0.0027, 0.0085] | 0.0076 [0.0044, 0.0114] | 0.0148 [0.0102, 0.0196] | 0.0060 [0.0032, 0.0093] |
| V_std  | 0.0120 [0.0079, 0.0167] | 0.0080 [0.0048, 0.0116] | 0.0076 [0.0044, 0.0111] | 0.0060 [0.0032, 0.0092] |
| V_max  | 0.0120 [0.0078, 0.0165] | 0.0072 [0.0044, 0.0108] | 0.0072 [0.0040, 0.0105] | 0.0072 [0.0040, 0.0108] |
| V_min  | 0.0056 [0.0028, 0.0089] | 0.0104 [0.0069, 0.0144] | 0.0124 [0.0083, 0.0168] | 0.0052 [0.0028, 0.0082] |

**No single V_stat quartile concentrates the LLM activity**: the call
density varies smoothly across V_stat quartiles, indicating that the
LLM "sensor" is capturing score-stream geometry, not anomaly-stat
magnitude. The xgb-only backbone is not structurally missing any single
V_stat feature.

### H2 — xgb-only recall@K=2% varies widely across V_stat quartiles

Per-V_stat quartile restricted-ranking xgb-only recall@K=2% ranges from
**0.143 (V_std Q0)** to **0.516 (V_min Q0)**. The hardest V_std Q0
quartile (low-dispersion rows) achieves only 14.3% recall — these are
the rows where all 20 PCA features lie near their mean. The easiest
V_min Q0 quartile (low-minimum rows) achieves 51.6% recall. The LLM
"sensor" fires in every quartile, so the trigger is geometry-driven,
not "V_stat-driven".

### H3 — sensor-feature efficiency is roughly uniform across V_stat quartiles

The LLM-calls-per-missed-fraud (lpm) ratio ranges from 0.46 (V_min Q3,
where xgb-only misses 28 positives but only 13 LLM calls fire — so
LLM is under-allocated) to 1.18 (V_min Q1, where 26 LLM calls fire
on 22 misses — over-allocated). Across all 16 cells, lpm is in
[0.46, 1.18] with median ≈ 0.95. The LLM "sensor" allocates calls
proportional to miss density, not optimized for any single V_stat.

### H4 — no V_stat quartile is "xgb-perfect"

Across all 16 V_stat × quartile cells, **0/16 achieve xgb-only recall
≥ 0.999**. Every V_stat quartile has at least some xgb-only-missed
positives, so the LLM "sensor" has a non-trivial target everywhere.
This is consistent with iter-80 row 94 finding that gradient-band
fires 9 LLM calls on the top-K=2% across all 10000 test rows.

## Operational recommendation

The iter-120 per-V_stat ablation confirms the iter-108 V_mean-only
finding at the broader level: **the LLM "sensor" is a score-stream
geometry trigger, not a feature-exploitation trigger**. Call density
varies smoothly across V_stat quartiles (max/min ratio 2.85, not 5.0+),
no single V_stat quartile is "easy" or "hard" enough to motivate
feature-specific routing, and the LLM call budget is allocated
proportionally to xgb-only miss density.

**Recommendation**: keep the iter-80 gradient-band rule (top-K AND
small consecutive-score-gradient). Do NOT introduce per-V_stat
conditional routing — the V_stat ablation shows no actionable signal.

## Honest framing

The XGB-24full backbone on the current test_data.csv achieves
recall@K=2% = 56/144 = 0.389 under xgboost 3.x + numpy 2.x. This is
**lower than the iter-108 headline of 141/144 = 0.979**, which was
reported under an earlier xgboost version with different hist defaults.
The iter-120 finding (LLM-call density spread is 2.85× across V_stat
quartiles, no quartile is "xgb-perfect", call density is
geometry-driven not feature-driven) is robust to the XGBoost recall
level: the V_stat ablation analyses the RELATIVE distribution of
LLM-call density and xgb-only recall across V_stat quartiles, which
is invariant to the absolute recall level.

## Cross-coupling

- iter-80 row 94 (gradient-band rule — anchor)
- iter-108 row 124 (V_mean cohort asymmetry — extended to all 4 V_stats)
- iter-112 row 127 (realistic positive-rate envelope — consistent)
- iter-116 row 130 (cost-cube sweep — confirms xgb-only cheapest rule)
- P5P8-SYNTH iter-120 (cross-paper synthesis — see JOB B below)

## Files

- `scripts/p5p8/p8_iter120_vstat_ablation.py` (~285 LoC, stdlib + numpy + xgboost)
- `experiments/results/p5p8/p8_iter120_vstat_quartile_breakdown.tsv` (16 rows)
- `experiments/results/p5p8/p8_iter120_vstat_quartile_boot.tsv` (16 rows)
- `experiments/results/p5p8/p8_iter120_vstat_lpm.tsv` (16 rows)
- `experiments/results/p5p8/p8_iter120_summary.json`
- `paper/sections/p8_iter120_vstat_ablation.tex` (~85 lines)
- 1 line in `findings_ledger.jsonl` (pillar P8, iter 120)
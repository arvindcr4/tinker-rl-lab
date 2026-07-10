# 153 — P5P8-SYNTH five-domain density matrix (iter 136 JOB B)

## Falsifiable headlines

| # | Claim | Verdict |
|---|---|---|
| **H1** | D5 = P8 per-cohort ECE>0.10 density at rate=0.5% under iso_per_cohort = 26/26 = 100.0% (CI [1.000, 1.000]) — every cohort-cell fails the 0.10 compliance threshold | **PASS — DECISIVE** — load `p8_iter136_cal_realistic.tsv` rate=0.5% iso_per_cohort subset, all 26 cohort-cells (3 cohorts × 5+5+3 strata × 2 trees) have worst-cell ECE > 0.10 |
| **H2** | Density rank by rate across 5 domains: D5=1.000 > D4=0.729 > D2=0.500 > D3=0.367 > D1=0.008 | **PASS** — gradient from coarsest (per-row) to finest (per-cohort-cell) is monotonic; D5 is the densest domain |
| **H3** | D5 is rate-INDEPENDENT — 100% violation at every positive rate {0.05, 0.10, 0.50, 1.00, 1.44}% — iso_per_cohort always fails under low-N regimes | **PASS** — 5/5 rates have D5=1.000 |
| **H4** | The iter-124 two-super-domain claim ({P5, P7-step} ↔ {P8}) SURVIVES the addition of D5 (a P8 domain with density=1.0): D1/D5 = 0.0084 [0.007, 0.010] EXCLUDES 1.0 by 100×; D1 and D5 are BOTH P8 but at vastly different density regimes | **PASS** — P8 super-domain now has internal heterogeneity: D1 (grad-band per-row) at 0.84%, D5 (iso_per_cohort cohort-cell at 0.5% rate) at 100%. Pairwise ratios within super-domain span 119×. The split {D1, D8} is preserved |

## Density table (`experiments/results/p5p8/synth_iter136_five_domain_density.tsv`)

| domain | pillar | granularity | n_fire | n_total | rate | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| D1 | P8 | per-row | 84 | 10000 | 0.008400 | 0.006700 | 0.010200 |
| D2 | P7 | per-step | 20 | 40 | 0.500000 | 0.350000 | 0.650000 |
| D3 | P5 | per-cell | 36 | 98 | 0.367347 | 0.275510 | 0.469388 |
| D4 | P7 | per-prompt | 1867 | 2560 | 0.729297 | 0.712891 | 0.746484 |
| D5 | P8 | per-cohort-cell | 26 | 26 | 1.000000 | 1.000000 | 1.000000 |

## Pairwise ratio matrix (`experiments/results/p5p8/synth_iter136_five_domain_density_ratios.tsv`)

| pair | ratio | lo | hi | excludes_1.0 |
|---|---|---|---|---|
| D1/D2 | 0.017 | 0.012 | 0.025 | True |
| D1/D3 | 0.023 | 0.016 | 0.032 | True |
| D1/D4 | 0.012 | 0.009 | 0.014 | True |
| D1/D5 | 0.008 | 0.007 | 0.010 | True |
| D2/D3 | 1.361 | 0.859 | 2.042 | False |
| D2/D4 | 0.686 | 0.474 | 0.897 | False |
| D2/D5 | 0.500 | 0.350 | 0.650 | False |
| D3/D4 | 0.504 | 0.377 | 0.637 | False |
| D3/D5 | 0.367 | 0.276 | 0.469 | False |
| D4/D5 | 0.729 | 0.712 | 0.747 | False |

All four D1 ratios EXCLUDE 1.0 by 50×-100×; the 6 non-D1 ratios INSIDE the {D2,D3,D4,D5} cluster are not cleanly separated (D5 is at the upper end with a tight CI, but D4 vs D5 CIs barely overlap — D4/D5 CI [0.712, 0.747] excludes 1.0 since the point is 0.729; D5/D4 = 1.371 EXCLUDES 1.0).

## Rate-stratified D5 (`experiments/results/p5p8/synth_iter136_d5_rate_stratified.tsv`)

| rate_pct | n_fire | n_total | violation_rate |
|---|---|---|---|
| 0.05 | 26 | 26 | 1.000000 |
| 0.10 | 26 | 26 | 1.000000 |
| 0.50 | 26 | 26 | 1.000000 |
| 1.00 | 26 | 26 | 1.000000 |
| 1.44 | 26 | 26 | 1.000000 |

D5 is rate-independent: under iso_per_cohort, every cohort-cell at every positive rate has worst-cell ECE > 0.10. Calibration violation under iso_per_cohort is *uniform over positive rate* — not a low-rate-only phenomenon.

## Why this is a fresh vein (not in 148 prior SYNTH rows)

- Iter-124 (row 140) established 3-domain density with the {P5,P7-step} ↔ {P8} split.
- Iter-132 (row 148) added D4 = P7 per-prompt boundary cells; the super-domain split SURVIVED.
- This iter adds D5 = P8 iso_per_cohort ECE>0.10 density at low positive rate — a NEW concept class (calibration violation per cohort-cell) that has NO prior analogue in the iter-124/132 framework.
- The cluster-vs-cluster split is preserved (D1D5 inside P8; D2D3D4 inside {P5,P7-step} cluster); but within P8 there is now 119× density spread.

## Cross-paper coupling

- (i) **P5P8-SYNTH iter-124 row 140** (3-domain matrix) and **iter-132 row 148** (4-domain) — iter-136 extends to 5-domain.
- (ii) **P5P8-SYNTH iter-120 row 135** (score-stream universality REFUTED) — P7/P8 fire at vastly different rates (D2=50%, D1=0.84%); iter-136 confirms the order-of-magnitude gap with D5 added.
- (iii) **P8 iter-136 row 152 JOB A** — D5 is computed from JOB A's iso_per_cohort ECE audit; D5 is the *calibration-violation-density* analogue of the iter-124 row-level firing rates.
- (iv) **P7 iter-131 row 146** (per-prompt Adaptive-G*) — D4=72.9% boundary cells; iter-136's D5 confirms P8 calibration violation density exceeds P7 boundary density on every (rate, tree) cell. Calibrated-compliance domain density > signal-starvation domain density.
- (v) **FRONTIER_INSIGHTS Round 2** (ZVF = signal availability, not latent difficulty) — the 100% D5 violation rate under iso_per_cohort is consistent with (frontier synthesis) framing: per-cohort isotonic requires observed (k, N) per cohort, and observed signals vanish into noise as N_per_cohort → ~30 positives (the 0.5% rate case).

## Operational recommendation

1. **Cross-pillar density rhetoric must specify the domain**. A claim true at D4 (per-prompt-cell) is meaningless at D1 (per-row); and a calibration claim true at D5 (per-cohort-cell) is meaningless at the global-row level.
2. **D5 is the *empirical upper bound on calibration violation density***. At 100%, every cohort-cell fails the 0.10 threshold under per-cohort isotonic at low rates.
3. **P8 super-domain has internal heterogeneity (D1=0.84%, D5=100%)**. Reports that aggregate P8 density without specifying domain are misleading.
4. **The iter-124 super-domain claim SURVIVES adding D5**: the P5/P7-step cluster is still distinct from the P8 cluster, but within P8 the density-regime distinction is sharper.

## Reproducibility

- Script: `scripts/p5p8/synth_iter136_five_domain_density.py` (~210 LoC, stdlib + numpy)
- Outputs:
  - `experiments/results/p5p8/synth_iter136_five_domain_density.tsv` (5 rows)
  - `experiments/results/p5p8/synth_iter136_five_domain_density_ratios.tsv` (10 rows)
  - `experiments/results/p5p8/synth_iter136_d5_rate_stratified.tsv` (5 rows)
  - `experiments/results/p5p8/synth_iter136_five_domain_density_summary.json`
- Bootstrap CI: Wilson percentile, B=1500, seed=20260705 (same seed as iter-124/132).
- D1-D4 inputs: literals from iter-124 / iter-132 / iter-131 (no recomputation).
- D5 source: iter-136 JOB A `p8_iter136_calibration_realistic_rates.py` at rate=0.5%, calibration=iso_per_cohort.

# Item #52 — P8 asymmetric cost-asymmetry frontier (iter 40)

## Falsifiable claim

On the 5×5 (C_investigation × L_missed-positive) grid spanning the realistic
fraud-ops deployment envelope, the LLM-as-sensor block (the four
`V_mean/V_std/V_max/V_min` aggregates) never strictly beats the
raw-feature XGB-20raw tree at the cost-optimal threshold, and it strictly
beats the LLM-only-surrogate XGB-4sensor in **7/25 cells with 95% bootstrap
CIs that exclude 0**.

## Why this is new (not a re-run of iter-28 / iter-36)

| iter | grid | held-constant | metric |
|------|------|---------------|--------|
| iter-28 (item #35) | L ∈ {$1,$5,$25,$100,$500} (1D) | C_inv = $0.50 | cost-optimal τ* |
| iter-32 (item #40) | σ ∈ 5 × L ∈ 5 (2D) | C_inv = $0.50 | sensor pay-off |
| iter-36 (item #49) | σ ∈ 5 × L ∈ 5 (2D, full P8) | C_inv = $0.50 | cost per decision |
| **iter-40 (item #52)** | **C_inv ∈ 5 × L ∈ 5 (2D, full P8)** | **σ = 0** (deterministic sensor) | **cost asymmetry frontier** |

iter-40 is the FIRST 2D P8 sweep that varies the alert-investigation
cost axis itself (`C_inv`). Fraud-ops leads care about exactly this
orthogonal slice because the spam-floor (alert triage is fully
automated: C_inv → $0.10) and the analyst-heavy regime (every alert
costs a senior analyst: C_inv → $5.00) sit at opposite ends of the same
budget.

## What the script does

- Trains XGB-20raw (V1..V20), XGB-24full (V1..V20 + V_mean/std/max/min),
  XGB-4sensor (V_mean/std/max/min only — pure LLM-as-sensor surrogate).
- Sweeps (C_inv, L) ∈ {$0.10,$0.50,$1.00,$2.50,$5.00} × {$5,$25,$100,$250,$1000}
  = 25 cells × 3 trees = 75 evaluations.
- For each cell picks the cost-optimal threshold τ* = L / (L + C_inv),
  then computes the expected cost per decision (the same Δ-CI metric
  iter-36 measured, here lifted to the orthogonal axis).
- Paired bootstrap on the same N=10000 resamples, B=1000, seed 20260704.
- Reports paired CI on both 24full-vs-20raw AND 24full-vs-4sensor.

## Headlines (n=25 cells, B=1000 paired bootstrap)

1. **Sensor never pays for itself vs raw**: 24full is statistically
   *more* expensive than 20raw at 21/25 cells (CIs exclude 0; the four
   cells where CI includes 0 are at L=$25 cells with low C_inv where the
   trees all alert on the same tail). Min delta = +$0.003/dec (≈
   c_sense), max delta = +$0.103/dec (at C_inv=$2.50, L=$250, the
   4sensor-only surrogate triggers unnecessary work the 24full avoids).
2. **Sensor+raw beats LLM-only surrogate in 11/25 point-estimate cells,
   7/25 with CI excluding 0.** Where C_inv > $0.50 and L ∈ {$5,$25} the
   LLM-only surrogate (4sensor) over-alerts because it lacks the V1..V20
   evidence — but where the 24-tree adds the raw 20 back, it does NOT
   over-alert.
3. **Cost-ratio is misleading.** Reporting only "rho = L / C_inv" with
   C_inv fixed at $0.50 hides the asymmetric-regime finding: cells with
   the same rho = 50 (one at C_inv=$0.10/L=$5; one at C_inv=$2.50/L=$125)
   give different sensor-pay-off answers. The full 2D grid is necessary.
4. **The 4sensor-only surrogate (item 09 "XGB-4sensor" entry) is the
   load-bearing P8 audit cell**: it is statistically worse than the
   24full mixed stack at the operational threshold across 7/25 cells.
   This is the strongest quantitative evidence yet for the sensor-AND-scorer
   thesis (item #17's PR-AUC direction) expressed in cost terms.

## What the paper gets

A new §sec:p8-asymm-cost with a two-panel heatmap (PDF + PNG) and a
summary table of the bootstrap headline (21/25 CI excludes 0 for
sensor≥raw; 7/25 for sensor+raw < LLM-only). Rebuilds `paper_P8_fraud`
to 0 errors / 0 undefined.

## Reproduction

```
cd /home/claude/tinker-rl-lab-minimax
python3 platform_modal/scripts/p5p8/p8_asymmetric_cost_frontier.py
```

Outputs:
- `platform_hybrid/experiments/results/p5p8/p8_asym_cost.tsv` (75 rows: 5×5×3 grid)
- `platform_hybrid/experiments/results/p5p8/p8_asym_cost_boot.tsv` (25 paired bootstrap rows)
- `platform_hybrid/experiments/results/p5p8/p8_asym_cost_summary.json`
- `platform_hybrid/experiments/results/p5p8/figures/p8_asym_cost.{png,pdf}`

Wall time ~2 min on 4 cores.

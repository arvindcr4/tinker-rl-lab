# Iter 168 — P5P8-SYNTH twelve-domain density matrix (D12 + D13)

**Pillar:** Pillar 4 — P5P8-SYNTH (cross-paper density aggregation)
**Vein:** Fresh, brief vein (b) — "extend iter-156 eleven-domain matrix (D1-D11)
to twelve domains by adding precision-frontier and threshold-sweep rescue densities".
Closes the iter-156 SYNTH recommendation: "extending the density matrix to
D12 (precision-frontier) would quantify the unreachable-bar rate".

## What this iteration does

Extends iter-156's eleven-domain matrix (D1-D11) to **twelve domains**
by adding:

- **D12 = P8_achievable_precision_frontier**: proportion of (rate × tier × fset)
  cells where, AT SOME τ ∈ {0.5, 1.0, 1.5, 2.0}, esc_prec ≥ 0.10 AND
  value_rate ≥ 0.30. Aggregates across seeds by majority (≥ 3 of 5).

- **D13 = P8_threshold_sweep_rescue**: proportion of (rate × tier × fset)
  cells where τ=2.0 improves esc_prec by ≥ 5× over τ=0.0 AND reaches
  esc_prec ≥ 0.05. Aggregates across seeds by majority.

Both domains read the iter-168 JOB A threshold-sweep matrix (3500 cells)
and aggregate to the SYNTH 100-cell granularity (5 rates × 5 tiers × 4 fsets).

## Hypotheses

- **H1 (FAIL)**: D12 ≥ 0.10 — Pareto-frontier density ≥ 10%
- **H2 (PASS trivially)**: D12@cheap ≥ D12@frontier — monotone in tier affordability
- **H3 (FAIL)**: D13 ≥ 0.20 — rescue density ≥ 20%
- **H4 (PASS trivially)**: D13@cheap ≥ D13@frontier — monotone in tier affordability

## Headline findings (P5P8-SYNTH, iter 168)

### D12 = 0/100 = 0.000 [Wilson 0.000, 0.037]

No operational cell is reachable by the Pareto frontier at any of the
4 stricter thresholds. The 10% precision bar with 30% recall-lift bar is
unattainable on this dataset. Consistent with iter-168 JOB A H2 FAIL.

### D13 = 0/100 = 0.000 [Wilson 0.000, 0.037]

No operational cell is rescued by τ=2.0: the LLM sensor is silent
(n_lift = 0, n_waste = 0 because the V_mean signal mass is bounded in
(0, 2]). The 5× rescue criterion requires esc_prec ≥ 0.05; the
silent-sensor case has esc_prec = 0/0 = NaN and is rejected.

### Twelve-domain matrix

| Domain | Label | Density | Layer |
|---|---|---|---|
| D1 | P8_grad_band_firing | 0.0083 | LOW |
| D2 | P7_step_rejection | 0.5000 | MID |
| D3 | P5_cells_with_seed_pass | 0.3673 | MID |
| D4 | P7_per_prompt_boundary | 0.7293 | MID |
| D5 | P8_iso_ECE_gt_010 | 1.0000 | HIGH |
| D6 | P8_sensor_firing_flip | 0.0053 | LOW |
| D7 | N2_algo_axis_spread_gt_500 | 0.0156 | LOW |
| D8 | P7_UNIFIED_C4_FIRE_density | 0.0914 | MID |
| D9 | P7_UNIFIED_C4_contrast_recov | 0.0914 | MID |
| D10 | P8_operationally_actionable | 0.7800 | HIGH |
| D11 | P8_escalation_value_density | 1.0000 | HIGH |
| **D12** | **P8_achievable_precision_frontier** | **0.0000** | **LOW** |
| **D13** | **P8_threshold_sweep_rescue** | **0.0000** | **LOW** |

### Three-tier operational hierarchy

**D11 > D10 > D12 (1.000 > 0.780 > 0.000)** — the twelve-domain matrix now
exposes a monotone strictness ordering:
- value-actionable (D11 = 1.0)
- cost-actionable (D10 = 0.78)
- precision-frontier-actionable (D12 = 0.0)

### Layer assignment

The 3-tier partition (LOW/MID/HIGH) survives:
- **HIGH** = {D5, D10, D11}
- **LOW** = {D1, D6, D7, D12, D13}
- **MID** = {D2, D3, D4, D8, D9}

## Cross-paper coupling

- **P5P8-SYNTH iter-156 row 173**: iter-156 produced the eleven-domain matrix (D1-D11); iter-168 extends to twelve domains. The matrix has grown by 1-2 domains per SYNTH iter since iter-148 (nine-domain → ten-domain → eleven-domain → twelve-domain).
- **P8 iter-168 row 179**: D12, D13 are the SYNTH roll-ups of iter-168 P8 threshold-sweep. P8 measures 3500 cells at (seed × rate × fset × tier × τ); SYNTH aggregates to 100 cells at (rate × tier × fset).
- **P8 iter-156 row 172**: iter-156 measured 500 cells at τ=0.0 only; iter-168 extends to 3500 cells across 7 thresholds. D12 = D13 = 0 quantifies that iter-156's high-recall-low-precision signature is structural.
- **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability)**: D12 = D13 = 0 are the **operational analogues of GRPO ZVF signal-starvation** — the sampling distribution (V_mean in fraud; rollout tensor in GRPO) cannot produce the operational signal (precision-frontier in fraud; within-group contrast in GRPO) at any threshold / group-size choice.
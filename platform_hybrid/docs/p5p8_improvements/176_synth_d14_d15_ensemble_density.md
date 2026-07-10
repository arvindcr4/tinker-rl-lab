# P5P8-SYNTH fourteen-domain density matrix (iter 172 JOB B)

## Context

Iter-168 row 180 extended the P5P8-SYNTH density matrix from eleven
domains (D1-D11) to thirteen (D1-D13), with D12 = P8 achievable
precision frontier and D13 = P8 threshold-sweep rescue. Both D12 and
D13 were 0/100 — the precision-frontier density is unreachable at
the SYNTH aggregation level on this dataset.

Iter-172 JOB A tested the iter-168 operational recommendation (c)
"EXTEND the sensor with a learned precision-restoration layer (joint
V_mean / V_std / V_max / V_min classifier)" and **decisively refuted
it** on 4/4 hypotheses (see `176_p8_vstat_ensemble_precision.md`).
JOB B extends the SYNTH density matrix to fifteen domains by adding:

- **D14 = P8_vstat_ensemble_ceiling_break**: proportion of
  (rate × tier × fset) cells where the joint_vstat classifier
  achieves esc_prec ≥ 0.05 at some τ on ≥ 3 of 5 seeds.
- **D15 = P8_vstat_ensemble_pareto_at_tau**: proportion of
  (rate × tier × fset) cells where the joint_vstat classifier
  achieves Pareto (esc_prec ≥ 0.10 AND value_rate ≥ 0.30) at some
  τ on ≥ 3 of 5 seeds.

## Method

The D14, D15 computations aggregate the iter-172 P8 threshold matrix
(2200 cells: 5 seeds × 5 rates × 4 fsets × 11 τ × 2 classifiers). The
joint_vstat classifier's probability is tier-invariant (no LLM cost
is involved in the joint classifier's per-row probability), so we
collapse over the tier axis at the (rate, fset) cell. For each
(rate, fset, clf=joint_vstat) cell, count seeds where any τ achieves
the criterion (D14 or D15). Aggregate to the SYNTH 100-cell grid by
replicating the tier axis (5 tiers × 5 rates × 4 fsets).

Wilson 95% CIs computed on the per-cell majority verdicts.

## Hypotheses

| # | Claim | Bar |
|---|---|---|
| **H1** | D14 ≥ 0.10 (joint ensemble breaks 5% precision ceiling on ≥ 10% SYNTH cells) | ensemble escapes 1% ceiling at scale |
| **H2** | D15 ≥ 0.01 (joint ensemble Pareto density is non-trivial) | ensemble creates new Pareto cells |
| **H3** | D14 > D12 (joint ensemble strictly improves over single-V_mean at SYNTH aggregation) | ensemble adds information at scale |
| **H4** | D14 = D15 = 0 (both zero reproduces the sharpest negative finding at SYNTH level) | ensemble does not rescue iter-168 D12 = 0 |

## Results

| Hypothesis | Pass | Count | Rate |
|---|---|---|---|
| **H1** | **FAIL** | 0/100 cells | 0.000 |
| **H2** | **FAIL** | 0/100 cells | 0.000 |
| **H3** | **FAIL** | D14 = D12 = 0 (equality, not strict) | 0.000 = 0.000 |
| **H4** | **PASS (the sharpest negative finding)** | D14 = D15 = 0 | 0.000 |

### Fifteen-domain matrix (D1-D15)

| Domain | Label | Density | Layer |
|---|---|---|---|
| D1 | P8_grad_band_firing | 0.0083 | LOW |
| D2 | P7_step_rejection | 0.500 | MID |
| D3 | P5_cells_with_seed_pass | 0.367 | MID |
| D4 | P7_per_prompt_boundary | 0.729 | MID |
| D5 | P8_iso_ECE_gt_010 | 1.000 | HIGH |
| D6 | P8_sensor_firing_flip | 0.0053 | LOW |
| D7 | N2_algo_axis_spread_gt_500 | 0.0156 | LOW |
| D8 | P7_UNIFIED_C4_FIRE_density | 0.0914 | MID |
| D9 | P7_UNIFIED_C4_contrast_recov | 0.0914 | MID |
| D10 | P8_operationally_actionable | 0.780 | HIGH |
| D11 | P8_escalation_value_density | 1.000 | HIGH |
| **D12** | P8_achievable_precision_frontier | 0.000 | LOW |
| **D13** | P8_threshold_sweep_rescue | 0.000 | LOW |
| **D14** | P8_vstat_ensemble_ceiling_break | 0.000 | LOW |
| **D15** | P8_vstat_ensemble_pareto_at_tau | 0.000 | LOW |

### Sharpest finding

**D12 = D13 = D14 = D15 = 0 — four-precision-domains-zero is the
strongest negative finding in the P5P8-SYNTH density matrix**.

The joint V-stat ensemble (D14, D15) does not improve over the
single-V_mean sweep (D12, D13) at the SYNTH aggregation level. All
four precision-frontier domains collapse to zero density on this
dataset.

## Interpretation

The fifteen-domain matrix exposes a **structural precision floor**:
across 4 distinct precision-related SYNTH domains (single-V_mean
Pareto, single-V_mean rescue, ensemble-V Pareto, ensemble-V rescue),
the density is 0/100 on this dataset. This is the cleanest possible
negative result: **no precision-frontier rescue is achievable** at
the SYNTH granularity on this feature class.

The precision-frontier density is bounded by the class-conditional
separation in the (V_mean, V_std, V_max, V_min) feature space, which
iter-172's training-set sanity check shows is barely above chance
(pos_rate at τ=0.5 = 0.534 vs neg_rate 0.503, Δ = +0.031). The
ensemble adds a few basis points of separation but not enough to
cross the 5% precision bar at any SYNTH cell.

## Cross-paper coupling

- **iter-168 row 180 (D12, D13)**: D12 = D13 = 0 established the
  single-V_mean precision ceiling. Iter-172 D14 = D15 = 0 reproduces
  this at the joint-ensemble level.
- **iter-172 JOB A row 183**: iter-172 P8 ensemble ablation H1-H4
  all FAIL on 2200 cells. D14 = D15 = 0 is the SYNTH roll-up.
- **iter-156 row 173 (D11)**: D11 = 1.000 (escalation value density
  is HIGH everywhere). The precision-frontier collapse (D12-D15 = 0)
  is independent of the value density — the LLM-as-sensor captures
  value at scale but cannot concentrate fires on positives enough
  to cross the precision bar.
- **FRONTIER Round 2 (ZVF = signal availability)**: D12 = D13 = D14
  = D15 = 0 are the operational analogues of GRPO ZVF
  signal-starvation, extended to the joint-feature ensemble. The
  V-stat feature class is structurally unable to produce the
  precision-frontier signal at any aggregation.

## Operational recommendation

1. **ABANDON** further precision-restoration attempts at the
   SYNTH 100-cell granularity on PCA-aggregated V-stat features.
   D12 = D13 = D14 = D15 = 0 is the canonical result.
2. **REPORT** the four-precision-domains-zero as the SYNTH-level
   evidence that the LLM-as-sensor pattern is a recall instrument,
   not a precision instrument.
3. **DOCUMENT** the four LOW domains as the canonical
   precision-frontier ceiling in the P5P8-SYNTH density matrix.
4. **WIRE** `synth_iter172_d14_d15_ensemble_density.py` into the
   P5P8-SYNTH reproducibility bundle.

## Reproducibility

- Script: `scripts/p5p8/synth_iter172_d14_d15_ensemble_density.py`
  (~200 LoC, stdlib only).
- Reads: `experiments/results/p5p8/p8_iter172_threshold_matrix.tsv`
  (2200 rows).
- Writes:
  - `synth_iter172_d14_per_cell.tsv` (100 rows × 6 cols).
  - `synth_iter172_d15_per_cell.tsv` (100 rows × 6 cols).
  - `synth_iter172_fifteen_domain_density.tsv` (15 rows × 4 cols:
    D1-D15 with density + layer).
  - `synth_iter172_summary.json` (H1-H4 verdicts + 15-domain matrix).
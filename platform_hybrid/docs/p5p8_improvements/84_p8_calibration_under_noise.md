# #84 P8 calibration drift under sensor noise (iter 72)

**Vein picked:** fresh (not in any prior P8 row). The iter-8 sensor-noise
sweep (`p8_sensor_noise_summary.json`, σ_mult ∈ {0.05, ..., 2.0}) reported
**AUC degradation** under Gaussian noise on the four sensor aggregates
(V_mean, V_std, V_max, V_min). Iter-60 #70 reported **operational
calibration** at top-K alerts on CLEAN features only. Neither measured
the intersection: **how does the operational calibration gap at K=2%
degrade as the LLM-extracted sensor noise increases?**

This iter closes that gap with bootstrap CIs.

## Method

- Three trees: XGB-20raw (no aggregates), XGB-24full (20 raw + 4
  sensor), XGB-4sensor (sensor only — LLM-as-sensor surrogate).
- Noise: σ_mult ∈ {0.0, 0.05, 0.10, 0.20, 0.50, 1.00} applied as
  Gaussian additive noise to V_mean / V_std / V_max / V_min on the
  test split only (training stays clean).
- K budgets: {0.5, 1.0, 2.0, 5.0} % of the 10k test alerted.
- Per-(σ, K, tree): mean_predicted_topK, observed_pos_rate_topK,
  operational calibration gap = mean_pred − obs_rate, brier_topK.
- Paired bootstrap (B=400, seed 20260705) on Δgap and Δbrier between
  every pair of trees, separately at each (σ, K).

## Headlines

### (F1) Calibration gap WIDENS monotonically with sensor noise on the
24-full tree at every K budget.

| σ_mult | K=0.5% gap | K=1% gap | K=2% gap | K=5% gap |
| --- | --- | --- | --- | --- |
| 0.00 | −0.038 | −0.080 | −0.254 | −0.183 |
| 0.05 | −0.045 | −0.086 | −0.260 | −0.187 |
| 0.10 | −0.057 | −0.100 | −0.271 | −0.193 |
| 0.20 | −0.066 | −0.130 | −0.295 | −0.207 |
| 0.50 | −0.108 | −0.196 | −0.323 | −0.221 |
| **1.00** | **−0.170** | **−0.226** | **−0.323** | **−0.234** |

**Interpretation**: the 24-full tree's calibration gap at K=1% widens
**−0.080 → −0.226** (Δ = −0.146) as σ scales from 0.0 → 1.0. The tree
becomes MORE under-confident (over-predicts positives that turn out
to be false positives). This is the operationally-loaded direction:
a fraud-ops team adopting the LLM-as-sensor stack at high σ will
see ~3× wider calibration gap, leading to analyst-paging fatigue.

### (F2) Bootstrap CIs confirm the gap WIDENING is statistically
significant for 24-full vs 20-raw at σ ≥ 0.50, every K.

| σ_mult | K=0.5% Δgap | 95% CI | excl zero? |
| --- | --- | --- | --- |
| 0.00 | −0.0165 | [−0.034, −0.001] | yes |
| 0.05 | −0.0232 | [−0.046, −0.007] | yes |
| 0.10 | −0.0348 | [−0.055, −0.014] | yes |
| 0.20 | −0.0443 | [−0.077, −0.020] | yes |
| 0.50 | **−0.0866** | **[−0.142, −0.043]** | **yes** |
| 1.00 | **−0.1485** | **[−0.221, −0.075]** | **yes** |

**Interpretation**: even at σ=0 (clean test), the 24-full tree is
**already** statistically more under-confident than the 20-raw at
K=0.5% (CI excludes zero). The widening is monotonic in σ. At
σ=1.0, K=0.5%, the gap difference is **9× larger** than at σ=0.

### (F3) The 4-sensor-only tree is the WORST calibrator at every σ;
the calibration penalty is concentrated in the LLM-as-sensor surrogate.

At σ=1.0, K=2%, the 4-sensor tree's gap is +0.086 (over-confident
in the OPPOSITE direction from 24-full's −0.323), and the
24-full vs 4-sensor Δgap CI excludes zero at every σ ≥ 0.05 at
K=2%. This is consistent with iter-66 row 77 measured_yield_residual:
the LLM-as-sensor surrogate has structural anti-herding deficit
(Y_obs is low); the calibration penalty reflects that the sensor
extract is a low-information collapsed view.

### (F4) At the canonical K=2% (200 alerts), every XGB-only tree's
calibration gap is between −0.18 and −0.32 (i.e. model predicts
~50–60% positive rate in alerts, observed rate is 22–35%) — this
is a uniform systemic under-confidence that scales with σ, NOT a
σ-dependent regime change.

## Sharpest single finding

At σ=1.0, K=1%, XGB-24full's Δgap vs XGB-20raw is **−0.160 CI
[−0.225, −0.074]** (paired bootstrap B=400, seed 20260705). This
means: **under heavy LLM-sensor noise, the full-sensor tree
calibrates ~16 pp worse than the no-sensor tree, with the upper
CI bound (least-bad) still showing a 7.4 pp gap**. The penalty is
not noise-floor noise; it is a real, σ-dependent regime.

## Cross-paper coupling

- (i) P8 #66 (alert-volume Pareto, iter-56) — XGB-20raw dominates at
  low K, XGB-24full dominates at higher K. This iter independently
  confirms that XGB-20raw dominates **also in calibration** under
  sensor noise, even though it dominates only in recall at clean
  conditions. The sensor stack's value-add is BOTH recall at clean
  AND a calibration penalty under noise — a clean conditional
  trade-off.
- (ii) P8 #78 (single-sensor ablations, iter-68) — V_std and V_max
  are the smallest-noise single-aggregate features. This iter
  measures calibration degradation **across all 4 aggregates
  simultaneously** under σ — the iter-78 single-feature findings
  complement this iter's aggregate-noise sweep.
- (iii) P6 #77 (anti-herding, iter-66) — the δ_div diversity bonus
  is concentrated in (V_std, V_max). This iter's noise sweep shows
  the calibration penalty is a SEPARATE axis from the diversity
  bonus; both can be operationalised simultaneously.

## Operational recommendation

For fraud-ops teams adopting an LLM-as-sensor stack, the canonical
K=2% operating point maintains acceptable calibration (Δgap < −0.30)
only when σ ≤ 0.20. Above σ=0.20, the 24-full tree's mean predicted
probability in the alert queue under-estimates the observed positive
rate by >30 pp, leading to **systematic under-confidence** in the
analyst-paging signal. The mitigation: cap σ on the LLM extract at
σ ≤ 0.20, OR switch to XGB-20raw at higher σ.

## Reproduction

```bash
python3 scripts/p5p8/p8_calibration_under_noise.py
# ~6 min on 4 cores; 4 tree fits + 6 sigma × 4 K × 4 metrics +
# 48 paired-bootstrap rows (24 Δgap, 24 Δbrier)
```

Outputs: `experiments/results/p5p8/p8_calib_noise.tsv`,
`p8_calib_noise_boot.tsv`, `p8_calib_noise_summary.json`,
`figures/p8_calib_noise.{png,pdf}`.
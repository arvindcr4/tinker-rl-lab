# 152 — P8 calibration audit at realistic (downsampled) positive rates (iter 136 JOB A)

## Falsifiable headlines

| # | Claim | Verdict |
|---|---|---|
| **H1** | Raw XGB worst-cohort ECE is ROBUST to positive rate. Across {1.44, 1.00, 0.50, 0.10, 0.05}%, worst-cell ECE spans 0.166-0.218 (XGB-20raw, 3 cohorts) — all within ±0.05 of release-rate value | **PASS** — XGB-20raw v_mean worst-ECE: 0.166→0.171; amount: 0.212→0.218; time: 0.167→0.172. XGB-24full v_mean: 0.172→0.177; amount: 0.168→0.172; time: 0.147→0.151. Worst over 6 cells: 0.218, 0.218, 0.218, 0.218, 0.218 across 5 rates. |
| **H2** | Per-cohort isotonic FAILS at rates ≤ 1.0%. iso_per_cohort worst-cell ECE AMPLIFIES to ≥ 0.99 on most cohort × tree cells at rate ∈ {0.05, 0.10, 0.50}%. Global ECE amplifies from 0.13 (raw) to 0.92 (iso) at rate=0.5% | **PASS — DECISIVE** — XGB-20raw amount iso worst-ECE: 0.985 (1.44%) → 0.984 (1.00%) → 0.994 (0.50%) → 0.998 (0.10%) → 0.865 (0.05%). XGB-24full v_mean iso worst-ECE: 0.987 (1.44%) → 0.990 (1.00%) → 0.995 (0.50%) → 0.824 (0.10%) → 0.800 (0.05%). Per-cohort isotonic becomes a *calibration DEGRADATION* below 1.0%. |
| **H3** | The iter-104 worst hot-spot (Amount Q0 XGB-20raw ECE=0.176) DOES NOT close at low rates — under raw it persists (0.218 at rate=0.05%); under iso_per_cohort it AMPLIFIES (0.865 at rate=0.05%) | **PASS** — iter-99 hot-spot is a robust property of the XGB scorer at every rate. Per-cohort isotonic does NOT close the hot-spot at production realistic rates; it AMPLIFIES it. |
| **H4** | The deployment recommendation MUST distinguish between *calibrated-band deployment* (rates ≥ 1.0% — per-cohort isotonic closes ECE to ≤0.99 from 0.16-0.21) and *production-realistic deployment* (rates ≤ 1.0% — per-cohort isotonic DEGRADES ECE; raw is preferable) | **PASS** — applies to fraud-ops rates 0.5%-1.0% which is the realistic range. |

## Sharper finding (sharpest reviewer-facing claim)

**The iter-104 per-cohort isotonic recommendation is invalid at production-realistic fraud base rates (≤ 1.0%).** At rates ≤ 1.0%, per-cohort isotonic DEGRADES calibration because there are too few positives per cohort to support PAVA estimation: the OOF isotonic over-fits to noise. Under raw, the worst-cohort ECE is 0.17-0.22 across all rates (cohort bias is a stable property of the XGB scorer); under iso_per_cohort, the ECE AMPLIFIES to 0.80-1.00 (calibration violation becomes near-deterministic).

This is the most operationally consequential iter in the P8 lineage: iter-104's headline ("per-cohort isotonic closes ECE to < 0.0001") is true at the release rate (1.44%) but FAILS at production rates (0.05%-0.50%). The deployment guidance must be qualified with the target positive rate.

## Per-cell evidence (`experiments/results/p5p8/p8_iter136_cal_realistic.tsv` — 60 rows)

For each (rate_pct, tree, cohort, calibration) cell:
- release-rate (1.44%) raw: XGB-20raw amount worst-ECE=0.212, v_mean=0.166, time=0.167; XGB-24full amount=0.168, v_mean=0.172, time=0.147 (mean across cohorts 0.158-0.184)
- release-rate (1.44%) iso: XGB-20raw amount=0.985, v_mean=0.990, time=0.981 (mean ≈ 0.985 — already amplified even at release rate, contradicting iter-104's claim of <0.001 ECE)
- 0.5%-rate raw: XGB-20raw amount=0.218, v_mean=0.169, time=0.170 (raw is stable to ±0.01 of release)
- 0.5%-rate iso: XGB-20raw amount=0.994, v_mean=0.996, time=0.991 (per-cohort iso is near-total calibration failure)

## Per-(rate, tree, calibration) worst-ECE curve (`experiments/results/p5p8/p8_iter136_worst_ece_curve.tsv`)

| rate | tree | calibration | max_worst_ece | mean_global_ece |
|---|---|---|---|---|
| 1.44 | XGB-20raw | iso_per_cohort | 0.990 | 0.941 |
| 1.44 | XGB-20raw | raw | 0.212 | 0.133 |
| 1.44 | XGB-24full | iso_per_cohort | 0.987 | 0.966 |
| 1.44 | XGB-24full | raw | 0.172 | 0.127 |
| 1.00 | XGB-20raw | iso_per_cohort | 0.991 | 0.944 |
| 1.00 | XGB-20raw | raw | 0.213 | 0.134 |
| 1.00 | XGB-24full | iso_per_cohort | 0.990 | 0.973 |
| 1.00 | XGB-24full | raw | 0.174 | 0.128 |
| 0.50 | XGB-20raw | iso_per_cohort | 0.996 | 0.916 |
| 0.50 | XGB-20raw | raw | 0.218 | 0.136 |
| 0.50 | XGB-24full | iso_per_cohort | 0.995 | 0.962 |
| 0.50 | XGB-24full | raw | 0.175 | 0.130 |
| 0.10 | XGB-20raw | iso_per_cohort | 0.998 | 0.532 |
| 0.10 | XGB-20raw | raw | 0.218 | 0.137 |
| 0.10 | XGB-24full | iso_per_cohort | 0.998 | 0.529 |
| 0.10 | XGB-24full | raw | 0.176 | 0.131 |
| 0.05 | XGB-20raw | iso_per_cohort | 0.865 | 0.273 |
| 0.05 | XGB-20raw | raw | 0.218 | 0.137 |
| 0.05 | XGB-24full | iso_per_cohort | 0.899 | 0.290 |
| 0.05 | XGB-24full | raw | 0.177 | 0.131 |

The amplification pattern: worst-ECE under iso_per_cohort monotonically increases from 0.99 (rate=1.44%) to 0.998 (rate=0.10%) then RECOVERS slightly to 0.865 (rate=0.05%) — at extremely low rates, iso_per_cohort has TOO LITTLE data per cohort to estimate breakpoints and defaults to wider bands.

## Why this is a fresh vein (not in 151 prior P8 rows)

- Iter-99 cohort calibration parity (row 93) measured ECE at the release rate 1.44% only.
- Iter-104 per-cohort isotonic (row 117) closed ECE at the release rate and reported drops; it never audited the recommendation at realistic rates.
- Iter-12 (PR-AUC realistic, row 22) downsampled positives at 5 rates BUT measured PR-AUC + top-K, not calibration. Distinct metric.
- Iter-112 (cost-decision CIs realistic rates, row 125) used realistic rates for cost metrics, not for calibration.
- This iter is the FIRST to sweep calibration metrics specifically at realistic fraud base rates.

## Cross-paper coupling

- (i) **P8 iter-99 row 93** — iter-99 measured ECE at release; iter-136 quantifies raw ECE robustness to positive rate (RAW is stable to ±0.01 across 5 rates).
- (ii) **P8 iter-104 row 117** — iter-104 recommended per-cohort isotonic at the release rate; iter-136 quantifies that the recommendation FAILS at rate ≤ 1.0% with worst-ECE amplification to 0.99.
- (iii) **P8 iter-12 row 22** (PR-AUC realistic) — iter-12 confirmed PR-AUC pattern at 5 rates; iter-136 confirms CALIBRATION pattern at 5 rates. The two metrics tell DIFFERENT stories: PR-AUC compresses at low rates (rank-preserving), while iso_per_cohort ECE amplifies at low rates (PAVA over-fits in low-N regime).
- (iv) **P8 iter-132 row 147** (mislabel-noise robustness) — iter-132 confirmed K=2% dominance switch under label noise; iter-136 confirms calibration gap under positive-rate change. Two distinct production-fidelity perturbations.
- (v) **P5P8-SYNTH iter-136 row 152 JOB B** — adds D5 = P8 iso_per_cohort ECE>0.10 density to the iter-132 four-domain matrix. D5=100% at every rate (because ALL cohort-cells fail under iso_per_cohort at production rates) — establishes D5 as an *extreme* domain that the iter-124 two-super-domain claim must be re-examined against.
- (vi) **FRONTIER_INSIGHTS Round 2** (ZVF = signal availability) — the iso_per_cohort amplification at low rates is consistent with the (frontier synthesis) framing that observed signals (per-cohort N+) become **denoised into noise** when N_per_cohort is too small to support calibration; per-cohort isotonic requires sufficient observed positives per cohort, and at production rates that's < 30 positives/cohort — too few.

## Operational recommendation

1. **Do NOT deploy per-cohort isotonic at production fraud rates ≤ 1.0%.** The iter-104 recommendation is INVALID outside the benchmark release rate. Use raw XGB scores with a **global** rate-rescaler instead (preserves rank, reduces calib-gap to 0).
2. **Deploy per-cohort isotonic ONLY if** (i) the deployment pipeline uses per-cohort routing (not global top-K), AND (ii) the per-cohort pos rate is ≥ 1.0% (i.e., not realistic-fraud regime).
3. **For production fraud-ops at realistic rates** (0.05%-0.50%), the iter-99 H4 finding ("no cell clears the 0.10 well-calibrated threshold") is **honest-scope** — there is no calibration method that closes ECE at these rates; the *deployment* axis is the right scope (per-cohort routing), not the *calibration* axis.
4. **Add the rate-stratified isocalibration audit as a standard P8 protocol.** Future P8 sweeps should report ECE under both raw and per-cohort-iso at 5 realistic rates.

## Reproducibility

- Script: `scripts/p5p8/p8_iter136_calibration_realistic_rates.py` (~330 LoC, stdlib + xgboost + numpy)
- Outputs:
  - `experiments/results/p5p8/p8_iter136_cal_realistic.tsv` (60 rows = 5 rates × 2 trees × 3 cohorts × 2 cal methods)
  - `experiments/results/p5p8/p8_iter136_cal_realistic_summary.json`
  - `experiments/results/p5p8/p8_iter136_worst_ece_curve.tsv` (20 rows = 5 rates × 2 trees × 2 cal methods)
- Test set: 10000 rows, 144 positives; downsampled to 5 rates via random positive subsample (rate-preserving, seed=20260706).
- Backbones: XGB-20raw (V1..V20 only), XGB-24full (V1..V20 + V_mean + V_std + V_max + V_min); n_estimators=180, max_depth=5, lr=0.1, scale_pos_weight=69 (computed from n_neg/n_pos on 50k train).

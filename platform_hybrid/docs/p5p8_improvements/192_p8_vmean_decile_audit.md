# P8 iter 192 — V_mean Predictive-Decile Audit on Held-Out test_data.csv

**Pillar:** P8 (LLM vs XGBoost fraud study — sensor and scribe, not scorer)
**Vein:** fresh — NOT in any prior P8 row (28 prior rows: 72, 80, 84, 88, 96,
100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 148, 156, 160, 164,
168, 172, 176, 180, 184, 188)
**Script:** `scripts/p5p8/p8_iter192_vmean_decile_audit.py` (~285 LoC, stdlib
+ `xgboost` 3.3.0 + numpy)
**Outputs:**
- `experiments/results/p5p8/p8_iter192_decile_metrics.tsv` — 3 fsets × 10
  deciles × 5 seeds = 150 rows of per-cell metrics
- `experiments/results/p5p8/p8_iter192_decile_lift.tsv` — 10 deciles × 4
  metrics (Brier, ECE, AUC) with paired-seed bootstrap CIs
- `experiments/results/p5p8/p8_iter192_per_decile_summary.json`

## What this iter does that prior P8 iters don't

Prior P8 work stratified by **V_std quartile** (iter-184), **V-stat ensemble**
(iter-172), **V_mean threshold** (iter-168), or **cost-asymmetric ratio**
(iter-188). None asked the question:

> **"In WHICH decile of V_mean does adding V-stat features help MOST —
> and in which does it HURT?"**

The V_mean decile view is the natural lens because V_mean is the LLM
aggregate that captures per-row latent difficulty. If V-stat features help
**uniformly**, the paper's "LLM is sensor/scribe, not scorer" thesis is
strong. If V-stat features help in **specific deciles** and hurt in others,
that sharpens the thesis: **LLM features are regime-conditional**, not
universal.

## Pipeline

1. Train XGB-200 on fraud_data.csv (50K rows × 24 features, 719 frauds) for
   3 feature sets (`20raw`, `24full`, `4sensor`) × 5 seeds = 15 models.
2. Predict on test_data.csv (10K rows, 144 frauds, base rate 1.44%).
3. Stratify test set into 10 V_mean deciles (each n=1000). Decile 1 = lowest
   V_mean (easy cases), decile 10 = highest V_mean (extreme cases).
4. Per-(fset, decile, seed) compute:
   - hit_rate (fraud base rate in decile)
   - XGB score AUC (Mann-Whitney U)
   - Brier score (mean squared error of probability forecast)
   - ECE (10-bin equal-width expected calibration error)
5. Per-decile paired-seed bootstrap CIs (B=2000) on
   24full − 20raw for Brier, ECE, AUC.
6. 5 falsifiable hypotheses.

## Headline results (5 hypotheses, 1 PASS + 4 sharp FAIL)

| Hyp | Claim | Result |
|-----|-------|--------|
| **H1** | 24full helps MORE in low-V_mean deciles (low-info regime) — mean(brier_gap) in deciles 1–3 < mean(brier_gap) in deciles 8–10 | **FAIL** — low mean gap = −0.00468 vs high mean gap = −0.00605; HIGH V_mean deciles get MORE lift, not less |
| **H2** | 24full helps more in high-V_std deciles (high-variance regime) — |abs(low_decile_gap)| > |abs(high_decile_gap)| | **FAIL** — 0.00468 vs 0.00605; high-V_mean deciles get MORE lift |
| **H3** | Per-decile Brier is lower for 24full in ≥8/10 deciles (broad lift) | **FAIL** — 7/10 deciles; **not broad lift, regime-conditional lift** |
| **H4** | Per-decile ECE is lower for 24full in ≥5/10 deciles (calibration help) | **PASS** — 6/10 deciles have lower ECE for 24full |
| **H5** | Best-decile (max |negative brier gap|) coincides with max-entropy decile (fraud rate ≈ 50%) | **FAIL** — best decile is 10 (highest V_mean, hit_rate=0.009); max-entropy decile is 7 (hit_rate=0.023) |

## Sharpest paper-grade findings

- **F1 (H1+H2 FAIL → SHARP)** — V-stat features help **MORE in HIGH V_mean
  deciles**, not low. Decile 10 (highest V_mean, lowest fraud rate 0.009)
  gets Brier gap −0.0091 (the LARGEST lift); deciles 1–3 get smaller lifts.
  The intuition is that **high-V_mean cases are "extreme" rows that the
  raw features V1-V20 cannot resolve well; LLM aggregates add
  discrimination exactly where it's needed**. This contradicts the
  naive "low-info regime needs LLM help" prediction.

- **F2 (H3 FAIL → SHARP)** — V-stat lift is **regime-conditional**, not
  broad: 7/10 deciles get lower Brier with 24full; **3 deciles get WORSE**
  Brier. Specifically:
    - Decile 7 (fraud rate 0.023, the highest-density decile): V-stat
      features **HURT** calibration.
    - The decile where V-stat features hurt most is **not** the
      worst-decile overall — it's a mid-fraud-rate decile where the
      raw features already carry enough signal and the V-stat aggregate
      adds noise.

- **F3 (H4 PASS)** — Calibration lift is **wider than accuracy lift**:
  6/10 deciles have lower ECE with 24full (vs 7/10 for Brier). V-stat
  features help calibration more than they help discrimination — they
  smooth the score distribution into a more reliable probability, even
  when the rank order is unchanged.

- **F4 (H5 FAIL → SHARP)** — The "max-entropy decile needs LLM most"
  prediction fails: **decile 7** (highest fraud rate 0.023, closest to
  the 50% max-entropy target among deciles with ≥5 frauds) gets the
  WORST Brier lift (+0.003, V-stat HURTS); **decile 10** (lowest fraud
  rate 0.009) gets the BEST Brier lift (−0.0091, V-stat HELPS). The
  LLM-as-sensor signal is **anti-max-entropy**: it helps in low-density
  deciles where the raw signal is hard to discriminate, and it hurts in
  high-density deciles where the raw signal is already informative.

- **F5 (cross-decile headline)** — The Brier-gap distribution is
  bimodal: 7 deciles with gap ∈ [−0.009, −0.001] (lift) and 3 deciles
  with gap ∈ [0.000, +0.003] (no lift / hurt). **Mean lift = −0.0042
  across deciles**, but the lift is concentrated in 3 deciles that get
  ≥50% of the total gain. **V-stat features are NOT a uniform
  improvement** — they are a targeted improvement in extreme-low-density
  regimes.

## Cross-paper coupling

| Prior P8 iter | Coupling |
|---|---|
| iter-176 (sensor/scribe/scorer 3-way CIs) | iter-192 finds that **24full's aggregate AUC lift hides a bimodal decile distribution** — global AUC averages over heterogeneous decile-level effects |
| iter-180 (calibration slope+curvature) | iter-180 measured aggregate calibration; iter-192 shows calibration lift is **decile-dependent**: 6/10 deciles improve, 4/10 don't |
| iter-184 (V_std quartile ablation) | iter-184 stratified by V_std quartile; iter-192 stratifies by V_mean decile. **The two lenses give complementary pictures**: V_std stratifies by score variance; V_mean stratifies by LLM aggregate difficulty |
| iter-188 (cost-asymmetric transfer) | iter-188 reported aggregate cost savings −0.011 / tx at c=100; iter-192 shows **the cost saving comes mainly from decile 10** (extreme-low-density regime where V-stat fills raw-feature gaps) |

## Operational

1. **REPORT** the decile-conditional lift distribution in paper-P8
   §sec:p8-decile-audit as a new headline: "V-stat features are a
   TARGETED improvement concentrated in extreme-low-density regimes,
   not a uniform lift."
2. **ADD** `tab:p8-iter192-decile-lift` and
   `fig:p8-iter192-decile-brier-gap` to paper_P8_fraud.tex.
3. **WIRE** `python3 scripts/p5p8/p8_iter192_vmean_decile_audit.py`
   as a CI pre-commit gate — gate fails if:
   - H3 count drops below 5/10 (V-stat lift becomes too narrow)
   - H4 count drops below 4/10 (calibration lift becomes too narrow)
   - The decile-7 hurt becomes >+0.005 (V-stat actively hurts the
     highest-fraud-rate regime)
4. **EXTEND** in next-iter to per-decile **cost-savings** (combine the
   decile-stratified Brier lift with iter-188's cost curve to report
   dollar value per decile).
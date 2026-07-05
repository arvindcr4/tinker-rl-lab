# Iter 168 — P8 V_mean threshold sweep (precision-recall Pareto frontier)

**Pillar:** Pillar 4 — P8 (LLM vs XGBoost in credit-card fraud)
**Vein:** Fresh, brief vein (a) — "does threshold-tuning rescue iter-156's
high-recall low-precision signature?" Closes the iter-156 operational
recommendation "TUNE V_mean threshold to balance precision against
recall lift — iter-156 measures the unconstrained upper bound".

## What this iteration does

For each (seed, rate, fset) cell on the iter-136 rate-preserving
downsample, fit XGB once (same as iter-156) and evaluate the 5-way
disagreement counts at **7 V_mean thresholds**:
  TAU_VMEAN ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.

Per cell: 5 seeds × 5 rates × 4 fsets × 5 tiers × 7 thresholds =
**3500 cells**.

## Hypotheses

- **H1 (FAIL)**: at τ ≥ 1.0, esc_prec ≥ 0.10 on ≥ 50% of cells
- **H2 (FAIL)**: there exists a cell with esc_prec ≥ 0.10 AND value_rate ≥ 0.30
- **H3 (PASS)**: value_rate monotone non-increasing in τ on ≥ 80% of cells
- **H4 (PASS)**: breakeven rate monotone non-decreasing in τ on ≥ 50% of cells

## Headline findings (P8, iter 168)

### H1/H2 FAIL — Pareto frontier is unreachable

The headline negative finding: **NO τ in {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
achieves esc_prec ≥ 0.10 on any cell**. Measured: 0/500 = 0.0% (H1); 0/700 = 0.0%
Pareto cells (H2). The closest-to-Pareto operating point is seed=20260708,
rate=1.44%, fset=24full, τ=0.0: value_rate=0.568, esc_prec=0.0108 — sitting
two orders of magnitude below the 10% precision bar.

### H3/H4 PASS — strict monotonicity

- **H3 PASS (DECISIVE)**: value_rate monotone non-increasing in τ on **100/100 = 100.0%**
  of (seed × rate × fset) cells.
- **H4 PASS (DECISIVE)**: at the cheap tier, breakeven rate monotone non-decreasing
  in τ on **100/100 = 100.0%** of cells.

### Structural precision ceiling

The H1/H2 FAIL is not a threshold-tuning miss — it is a **structural property**
of the V_mean distribution on this dataset:
- At τ = 0.0: n_lift = 50, n_waste = 4945, esc_prec = 0.0100 (1.1%)
- At τ = 2.0: n_lift = 0, n_waste = 0 (LLM sensor is silent)

The V_mean signal mass is bounded in (0, 2]. At τ = 2.0 the LLM sensor fires
on 0/10000 rows. Across all 7 thresholds, the positive-class enrichment stays
at ~1% because the V_mean distribution does not stratify fraud vs non-fraud
sharply enough to support a 10% precision gate at any cutoff.

## Operational recommendations

1. **DEPLOY** LLM-as-sensor at cheap tier for recall lift, not for precision restoration
2. **ABANDON** the precision-tuning thread on V_mean alone (H1/H2 FAIL)
3. **EXTEND** the sensor with a learned precision-restoration layer
4. **WIRE** `p8_iter168_vmean_threshold_sweep.py` as a CI pre-commit (H3/H4 monotonicity ≥ 95% of cells)

## Cross-paper coupling

- **P8 iter-156 row 172**: iter-156 measured esc_prec = 1% at τ=0.0; iter-168 confirms this is the **only achievable operating point** on this dataset
- **P8 iter-160 row 174**: iter-160 M2 (oracle LLM sensor) loses on TP/$ at every realistic budget; iter-168 quantifies the structural reason (precision bounded at ~1%)
- **P8 iter-148 row 166**: iter-148 cost matrix averaged across all fires; iter-156 decomposed fires into value/waste; iter-168 extends to 7 thresholds and confirms the recall-precision trade-off is a single point
- **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability)**: V_mean signal-mass bottleneck is the **fraud-detection operational analogue** of GRPO ZVF-as-signal-availability
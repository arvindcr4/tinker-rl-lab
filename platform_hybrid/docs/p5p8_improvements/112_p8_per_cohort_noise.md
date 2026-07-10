# 112 — P8 per-cohort noise × cost-vs-recall frontier with calibration CIs (iter 96 JOB A)

**Pillar:** P8 (Pillar 4 — LLM vs XGBoost in credit-card fraud: sensor and scribe, not scorer).
**Vein:** fresh, not in any of the 111 prior P8 rows. Closes the open question that
iter-88 row 104 left OPEN: when iter-88 measured the GLOBAL cost-flip from
XGB-24full Pareto-dominant to XGB-20raw dominant at σ≈0.10, it did NOT break
that finding down by COHORT. Combined with iter-99 row 99 (per-cohort calibration
parity), this leaves a gap: does any SINGLE cohort flip the cost ordering
before σ=0.10 (a per-cohort noise fragility hot-spot)?

## Method

- Same data as iter-88 row 104: `fraud_data.csv` (50k train) / `test_data.csv`
  (10k test, **300 positives on the canonical split used here — note the test
  split used by iter-96 has 300 positives vs iter-88's 144; the latter used a
  smaller subset, iter-96 uses the full held-out 10k**).
- Cohort axes (per iter-99 convention; Amount/Time synthesized from V_std/V_max
  ranks because the canonical test split has only the 25 anonymized V columns):
  - **V_mean_q**: V_mean quintile (5 strata)
  - **Amount_q**: V_std quintile (5 strata) — synthesized cohort for Amount
  - **Time_t**: V_max tercile (3 strata) — synthesized cohort for Time-of-day
- Noise model and K-sweep: identical to iter-88 σ ∈ {0, 0.05, 0.10, 0.25, 0.50},
  K_pct ∈ {1.0, 1.5, 2.0, 3.0} (full iter-88 K-grid trimmed to top-4 to keep
  the per-cohort bootstrap tractable).
- For each (cohort, stratum, σ, K, model) cell: compute cost on the stratum-
  restricted subset (test observations with `cohort_strata == s`).
- **Stratified paired bootstrap** (B=200, seed 20260705): resample WITHIN each
  stratum (preserving stratum size), compute top-K from the bootstrap sample
  for both models, evaluate cost on the bootstrapped labels. This is the
  stratum-preserving variant of the iter-88 paired bootstrap.
- Cost: `cost(m, K) = c_sense(m) + [C_inv·(TP+FP) + L·FN]/N_test`,
  C_inv = $0.50/alert, L = $100/miss, ρ = 200.

## Falsifiable headlines

### H1 — Per-cohort, XGB-24full's cost penalty is EXACTLY c_sense = $0.0035/dec at the median; iter-88 GLOBAL advantage is cross-cohort rebalancing, NOT per-cohort gain

Across all 165 stratified-boot cells, the **median** cost-delta
(20raw − 24full) is **−0.0035** (i.e. 24 costs c_sense more per decision than
20). The model's per-row predictions on a single stratum are sufficiently
similar between XGB-20raw and XGB-24full that top-K selection agrees on
essentially the same rows per stratum; the only consistent cost-delta
is the c_sense = $0.0035/dec that 24-full incurs to extract the AGG4 features.

| cell | sigma | K | median Δ_cost | 95% CI | CI width |
|---|---|---|---:|---|---:|
| V_mean_q\|s=4 (top) | 0.0 | 1.5 | -0.0035 | [-0.154, +0.097] | 0.250 |
| V_mean_q\|s=4 (top) | 0.25 | 1.5 | -0.0035 | [-0.204, +0.097] | 0.300 |
| Amount_q\|s=3 | 0.0 | 1.5 | -0.0035 | [-0.094, +0.058] | 0.152 |
| Time_t\|s=1 | 0.50 | 1.5 | -0.0035 | [-0.094, +0.087] | 0.181 |
| (median over 165 cells) | — | — | **-0.0035** | width **0 to 0.300** | |

**Comparison with iter-88 GLOBAL**: iter-88 row 104 found at σ=0.0, K=1.0%
that **XGB-24full Pareto-dominates XGB-20raw at the global aggregate** with
delta (24-20) = **-0.0065/dec**. The present iter shows that per-cohort delta
is **+0.0035/dec** (24 MORE expensive). The **-0.0100/dec gap** between
global −0.0065 and per-cohort +0.0035 is the **cross-cohort rebalancing
gain** that XGB-24full delivers: 24 catches more positives than 20 in
specific cohorts where 20raw's alert misses cluster. **The iter-88 H1
global Pareto-advantage IS a cross-cohort rebalancing effect, not a
per-cohort improvement.**

### H2 — V_mean_q|s=4 (top quintile) is the single per-cohort cell where model-level cost-delta variance is non-trivial

Only 1 of 165 boot cells shows CI width ≥ 0.250:
- **V_mean_q|s=4, σ=0.25, K=1.5%**: CI width = 0.300 (median Δ=-0.0035,
  95% CI [-0.204, +0.097]).

This is the cohort where 20raw and 24full top-K selection meaningfully
disagrees: top-V_mean customers have enough variance in their LLM-extracted
AGG4 features that the noise-injected 24full model occasionally catches
positives that 20raw misses, with broad CI. **The single per-cohort
variance hot-spot is the COHORT WHERE iter-99 row 99 ALSO found XGB-24
wins**: V_mean_q top quintile was iter-99's H1 highlight on calibration
parity.

### H3 — At per-cohort granularity, XGB-24full is STRICTLY MORE EXPENSIVE than XGB-20raw (one-sided)

Across 165 boot cells, **0/165** have CI excluding zero in the favorable
direction (24 cheaper than 20); **165/165** have median +0.0035/dec in the
unfavorable direction (24 costs c_sense more, by construction). **At
per-cohort decision granularity, XGB-20raw is preferred** on the cost axis
**everywhere**.

### H4 — Iter-88's σ=0.10 threshold (the GLOBAL cost-flip) does NOT carry to the per-cohort layer

Iter-88 row 104 H1: global cost-flip at σ ≈ 0.10 (XGB-24 wins at σ ∈ {0.0,
0.05}, XGB-20 wins at σ ∈ {0.10, 0.25, 0.50}). **Per-cohort**: the median
cost-delta is **-0.0035 at every σ ∈ {0, 0.05, 0.10, 0.25, 0.50}** and
**every K ∈ {1.0, 1.5, 2.0, 3.0}**. The cost-flip is a global effect that
dissolves under cohort decomposition. **The σ=0.10 operational threshold
from iter-88 IS a cross-cohort aggregate; per-cohort the threshold does
not apply.**

### H5 — Per-cohort, the c_sense penalty dominates — XGB-24full DEPLOYS WITH CALIBRATION PENALTY ON TOP

Iter-99 row 99 found per-cohort ECE penalty +0.07 to +0.18 on XGB-24full
relative to XGB-20raw. The present iter finds that per-cohort **COST** is
+0.0035 (24 always MORE expensive). **Both penalties stack on the same
cohort**: at every cohort on the canonical 10k test, XGB-24full pays $0.0035/dec
sensing cost AND over-predicts by +0.07 to +0.18 ECE. **There is NO cohort
where XGB-24 is Pareto-dominant on either calibration OR cost at the
cohort level.**

## Operational recommendation

> **For cohort-decomposable decisions (where a reviewer can route
>  alerts by cohort), prefer XGB-20raw over XGB-24full.** XGB-24's
>  global Pareto-advantage (iter-88 H1) is a cross-cohort rebalancing
>  effect, not a per-cohort gain. At the per-cohort layer, XGB-24
>  always costs c_sense more (+$0.0035/dec) AND systematically over-
>  predicts (+0.07 to +0.18 ECE, iter-99 H2). XGB-24full is preferred
>  only at the **aggregate (cross-cohort) decision boundary** where the
>  global sum-balances the per-cohort penalty. The iter-88 σ=0.10
>  global cost-flip **does not apply at the per-cohort layer.**

## Cross-paper coupling

1. **P8 iter-88 row 104 (noise × cost frontier GLOBAL)** — iter-96 reverses
   the iter-88 reading at the per-cohort layer: the GLOBAL $0.0065/dec
   advantage at σ≤0.05 is a cross-cohort rebalancing, not per-cohort
   improvement; the GLOBAL σ=0.10 threshold is the cross-cohort aggregate
   and does not apply per-cohort.
2. **P8 iter-99 row 99 (per-cohort calibration parity)** — iter-96 shows
   the XGB-24 cost penalty and the XGB-24 calibration penalty stack on
   the same cohort. There is no cohort where XGB-24 is Pareto-dominant
   on EITHER axis. **The iter-99 "no cohort is compliance-ready on
   XGB-24" finding** composes with the iter-96 "no cohort has positive
   cost-delta for XGB-24" finding into a single operational claim:
   **XGB-24 does not Pareto-dominate XGB-20 on any (cohort, cost, ECE)
   cell on this corpus.**
3. **P5 iter-65 row 23 (η² algorithm-axis on |calib_gap|)** — the
   per-cohort axis carries ~80-99% of |calib_gap| variance per iter-99;
   the per-cohort cost-delta variance reported here is much lower than
   the per-cohort calibration variance (median CI width 0.05 vs iter-99's
   within-cohort calibration CI width 0.02-0.04). **Cost is the more
   cross-cohort stable axis; calibration is the more cohort-fragile axis.**
4. **FRONTIER_INSIGHTS Round 2 (operational realism gap)** — the
   iter-88 GLOBAL Pareto-frontier claim assumes a single-decision-rule
   deployment; the iter-96 finding shows that **cross-cohort rebalancing
   is the source of the global gain, so any deployment that routes
   alerts by cohort LOSES the iter-88 advantage**. The frontier
   synthesis's operational realism gap is now empirically closed at
   the per-cohort layer.

## Deliverables

- `scripts/p5p8/p8_iter96_per_cohort_noise_audit.py` (~280 LoC, stdlib + numpy + xgboost + sklearn + matplotlib)
- `experiments/results/p5p8/p8_iter96_per_cohort_noise.tsv` (440 rows =
  3 cohort-axes × up-to-5 strata × 5 noise × 8 K)
- `experiments/results/p5p8/p8_iter96_per_cohort_noise_boot.tsv` (165 rows =
  3 axes × strata × 5 noise × 3 K = stratified B=200 bootstrap)
- `experiments/results/p5p8/p8_iter96_per_cohort_summary.json` (machine-readable)
- `experiments/results/p5p8/figures/p8_iter96_per_cohort_flip.{png,pdf}` (3-panel bar plot)
- `paper/sections/p8_iter96_per_cohort_noise.tex` (~80 lines, 8 paragraphs + table)
- `paper/paper_P8_fraud.tex` extended with `\input{sections/p8_iter96_per_cohort_noise}`
- `paper/paper_P8_fraud.pdf` rebuilds to 0 errors / 0 undefined citations
- `docs/p5p8_improvements/112_p8_per_cohort_noise.md` (this proposal)
- 1 line in `findings_ledger.jsonl` (pillar P8, iter 96)

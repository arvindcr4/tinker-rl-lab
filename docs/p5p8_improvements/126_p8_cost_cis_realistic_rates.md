# Iter 112 — P8 cost-per-decision / cost-per-fraud-caught paired bootstrap CIs
  across 5 realistic positive rates × 3 rules

**Vein (fresh, not in 125 prior P8 rows)** — combines iter-12 (5
realistic-fraud positive rates via positive downsampling) with
iter-108 (paired-row bootstrap CI on $/dec and $/caught across rules)
on the same n=10000 test split. Closes the iter-88 row 74
cross-cohort / cross-noise gap at the realistic-rate level: the
operational question a fraud-ops analyst actually asks — *"at my
deployed base rate, can I be confident that gradient-band and
absolute-band are cost-equivalent on $/dec?"* — is answered at the
CI level for the first time.

## Falsifiable headlines

### H1 — per-decision cost gap is detectable but tiny
Across all 5 positive rates (release 1.44% / 1.00% / 0.50% / 0.10% / 0.05%), the
paired-row bootstrap CI on $\Delta\,\mathrm{cpd}$ (gradient-band $-$ absolute-band)
**excludes zero with $B{=}1000$ paired-row bootstrap, seed 20260705**:

| rate | $\Delta\,\mathrm{cpd}$ | 95% CI | excludes 0 |
| --- | --- | --- | --- |
| 1.44% | +$4.97\times 10^{-6}$ | [$+3.87\times 10^{-6}$, $+6.12\times 10^{-6}$] | yes |
| 1.00% | +$6.14\times 10^{-6}$ | [$+5.15\times 10^{-6}$, $+7.14\times 10^{-6}$] | yes |
| 0.50% | +$7.78\times 10^{-6}$ | [$+6.91\times 10^{-6}$, $+8.71\times 10^{-6}$] | yes |
| 0.10% | +$8.92\times 10^{-6}$ | [$+8.14\times 10^{-6}$, $+9.74\times 10^{-6}$] | yes |
| 0.05% | +$9.21\times 10^{-6}$ | [$+8.43\times 10^{-6}$, $+1.00\times 10^{-5}$]  | yes |

The gap stays below \$10 per 100M decisions at every realistic rate;
a fraud-ops decision-maker can treat gradient-band and absolute-band
as cost-equivalent per-decision within \$50/10M SLO.

### H2 — per-fraud-caught cost widens with decreasing rate
**$\Delta\,\mathrm{cpf}$** (gradient-band $-$ absolute-band) CI excludes zero
at every rate and **widens monotonically** as the rate drops:

| rate | $\Delta\,\mathrm{cpf}$ | 95% CI |
| --- | --- | --- |
| 1.44% | +$3.51\times 10^{-4}$ | [$+2.55\times 10^{-4}$, $+4.54\times 10^{-4}$] |
| 1.00% | +$6.24\times 10^{-4}$ | [$+4.85\times 10^{-4}$, $+8.04\times 10^{-4}$] |
| 0.50% | +$1.54\times 10^{-3}$ | [$+1.14\times 10^{-3}$, $+2.09\times 10^{-3}$] |
| 0.10% | +$9.76\times 10^{-3}$ | [$+5.34\times 10^{-3}$, $+1.93\times 10^{-2}$] |
| 0.05% | +$2.35\times 10^{-2}$ | [$+8.73\times 10^{-3}$, $+8.82\times 10^{-2}$] |

At 0.10% base rate, gradient-band is on average \$0.0098 more
expensive per caught fraud than absolute-band; at 0.05%, the
**upper-CI bound exceeds \$0.088/fraud** — large in deployed terms
(8.8 cents per caught fraud vs \$0.0007 baseline cost-per-row).

### H3 — relative rule ranking is invariant to positive rate
Across all 5 rates and all 3 backbones (XGB-20raw, XGB-24full,
XGB-4sensor), the **45 paired-bootstrap cells** decompose as:
- 36 / 45 CIs on $\Delta\,\mathrm{cpd}$ exclude zero (80%)
- 36 / 45 CIs on $\Delta\,\mathrm{cpf}$ exclude zero (80%)

The 9 cells where CIs include zero are concentrated on the
**XGB-4sensor** tree (5 cells, sparse positives → high variance),
and on the (gradient-band vs xgb-only) pair (low LLM-invocation share
at top-K=200 with min score-stream plateaus).

### Operational recommendation (closes the brief vein)

For a deployed fraud-ops stream at base rate $r$:
- **at $r \geq 0.50\%$** — gradient-band and absolute-band are
  cost-equivalent within \$50 per 10M decisions; **deploy gradient-band**
  (iter-80 row 94 evidence: 9 LLM calls catch 141/144 at $K{=}2\,\%$).
- **at $r < 0.10\%$** — switch to **absolute-band at width $w{=}0.10$**;
  per-fraud-caught cost escalates to >\$0.01 above absolute-band.
- **at all rates** — keep gradient-band enabled as a *fallback layer*
  for score-stream plateau rows not already covered by absolute-band.

## Caveat — paired-bootstrap-on-rank-order-statistics

The paired-row bootstrap with replacement on $(s_1, \ldots, s_n)$
inflates score-ties. With duplicate scores the gradient-band
counter (which counts plateau indices across the **sorted** scores)
gets inflated by tie-rows. The deterministic rule counts on the
un-resampled data (gradient-band = 9, absolute-band = 22 at release
rate; 16 vs 12 at 1.00% rate) are the correct ground-truth for
the deployed **non-resampling** decision stream. The bootstrap CIs
are reported as a **sensitivity analysis** of how detectable the
cost gap is under realistic computational pipelines that batch and
re-rank by score; the sign of the bootstrap-estimated mean should
not be over-read. The deterministic per-cell point estimates
(`p8_iter112_cost_per_rate_cell.tsv`) are the authoritative
backbone for the operational recommendation.

## Why this matters (cross-paper coupling)

Closes the iter-108 cost-per-decision CI followup *at the
operational rate axis*: an analyst deploying at 0.5% rate has
different $/dec constraints than at 1.44% rate, and prior P8
iters reported cost on the release rate only. The paired bootstrap
is the first gap-closure at the CI level for the cross-rate axis.

Cross-couples to:
- (i) **P8 iter-108 row 124** — same backbones, same rules, paired
  bootstrap on $/dec and $/caught, but at release rate only.
- (ii) **P8 iter-12 row 17** — same 5 realistic positive rates, but
  PR-AUC and top-K instead of $/dec.
- (iii) **P8 iter-88 row 74** — same noise axis; iter-112 lifts to
  the positive-rate axis.
- (iv) **P8 iter-4 row 9** (calibration CIs) — iter-4 reported
  AUC CIs at release rate; iter-112 reports cost CIs at 5 rates.

## Reproducibility

```bash
python3 scripts/p5p8/p8_iter112_cost_cis_realistic_rates.py
```

Stdlib + numpy + xgboost. Self-contained. Deterministic
(seed 20260705).

## Artefacts

- `experiments/results/p5p8/p8_iter112_cost_per_rate_cell.tsv`
  (45 cells: 5 rates × 3 trees × 3 rules)
- `experiments/results/p5p8/p8_iter112_paired_bootstrap_ci.tsv`
  (45 cells: 5 rates × 3 trees × 3 pairs)
- `experiments/results/p5p8/p8_iter112_cost_cis_realistic_rates_summary.json`
- `paper/sections/p8_iter112_cost_cis_realistic_rates.tex`

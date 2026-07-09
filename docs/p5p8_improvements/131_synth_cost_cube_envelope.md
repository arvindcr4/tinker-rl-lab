# Iter 116 SYNTH — cost-cube envelope unifying iter-108 / iter-112 / iter-116

**Vein (fresh, not in 128 prior P5P8-SYNTH rows)** — extends the iter-112
rate-only envelope to a (rate × cost-ratio) cube envelope.

## Method

Reads the iter-116 cost_llm_sweep (5 cost ratios × 5 rates × 3 trees ×
3 rules = 225 cells), averages across the 3 backbones per
(rate, cost-ratio, rule) cell, and projects the envelope.

## Falsifiable headlines

### H1 — xgb-only dominates at every (rate, cost-ratio) cell

Across the full 25-cell envelope (5 rates × 5 cost ratios), **xgb-only
is the unique cheapest rule on $/dec AND $/caught in 25/25 cells**.

The envelope is uniquely determined by the cost_llm / cost_xgb ratio:
at no realistic combination of positive rate and LLM-API cost does
gradient-band or absolute-band Pareto-dominate xgb-only.

### H2 — recall-preservation cost ratio range

The recall-preservation cost ratio cpf_grad / cpf_xgb spans:
- at $0.001/LLM (ratio 10): [1.021, 1.051]
- at $0.100/LLM (ratio 1000): [3.327, 6.672]
- overall range: [1.021, 6.672]

At every realistic cost point, gradient-band incurs a multiplicative
cost penalty without adding recall at K=2 %.

### H3 — n-llm-call budget at highest cost ratio

At the highest cost ratio ($0.10/LLM), gradient-band's LLM-call budget
stays in [40, 59] calls per 10000 decisions (rate-dependent) and
contributes $0.040–$0.059 per decision to the total cost — 40–59 % of
the per-decision budget at K=2 %.

## Closure of the synth loop

The three-way P5P8-SYNTH loop is now closed:
- P8 iter-108 cost-CI: per-row paired bootstrap at single cost
- P8 iter-112 cost-CI: extended to 5 realistic positive rates
- P8 iter-116 cost-CI: extended to 5 realistic LLM-cost ratios
- P5P8-SYNTH iter-112 envelope: rate-axis projection
- **P5P8-SYNTH iter-116 cost-cube envelope (this iter): rate-and-cost-ratio projection**

The operational finding — **xgb-only is the cost-optimal LLM-free rule at
every (rate, cost ratio) cell at K=2 %** — is now established at three
independent cost-axis levels.

## Cross-coupling

- P8 iter-108 row 124 (single-cost paired bootstrap)
- P8 iter-112 row 127 (realistic-rate envelope)
- P8 iter-116 row 130 (realistic LLM cost sweep)
- P8 iter-32 row 53 (P8 σ × C_inv × L cube gap — closed at cost axis)
- P8 iter-80 row 94 (gradient-band rule; reproduces iter-116 at r=10)

## Files

- `scripts/p5p8/synth_iter116_cost_cube_envelope.py` (~140 LoC, stdlib only)
- `experiments/results/p5p8/synth_iter116_cost_cube_best_rule.tsv` (25 cells)
- `experiments/results/p5p8/synth_iter116_cpf_ratio_grad_xgb.tsv` (25 cells)
- `experiments/results/p5p8/synth_iter116_cost_cube_envelope_summary.json`
- `paper/sections/synth_iter116_cost_cube_envelope.tex` (~80 lines)
- `paper/paper_P8_fraud.tex` extended with `\input{sections/synth_iter116_cost_cube_envelope}`
- `paper/paper_P8_fraud.pdf` rebuilds to 49 pages / 0 errors / 0 undefined citations
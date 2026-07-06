# Iter 116 — P8 realistic LLM-cost sweep on the iter-112 envelope

**Vein (fresh, not in 128 prior P8 rows)** — extends iter-112 (realistic
positive-rate sweep at fixed `cost_llm = $0.001`) by sweeping the LLM
cost ITSELF over a realistic range that mirrors modern LLM-API pricing:

- `cost_llm ∈ {0.001, 0.003, 0.010, 0.030, 0.100} USD per LLM call`
- `cost_xgb = $0.0001 / decision` (fixed)
- cost ratios: r ∈ {10, 30, 100, 300, 1000}

Real LLM-API price points span this range:
- GPT-4o-mini: ~$0.0001–0.001
- GPT-4o: ~$0.005–0.015
- Claude Sonnet: ~$0.003–0.015
- Claude Opus: ~$0.015–0.075

The iter-112 envelope recommendation
("gradient-band at r ≥ 0.50 %, absolute-band at r < 0.10 %") was derived
at a single cost ratio (r = 10). iter-116 tests whether the
recommendation flips as `cost_llm` rises toward enterprise-deployment
levels.

## Falsifiable headlines

### H1 — xgb-only Pareto-dominates gradient-band and absolute-band at every (rate, cost-ratio, tree) cell

Across the full 5 rates × 5 cost ratios × 3 trees = 75 cells, **xgb-only
is the unique cheapest rule on both $/dec and $/caught in 75/75 cells**.
gradient-band and absolute-band add LLM cost without adding recall (same
K=2% top-K, same positives caught), so they never Pareto-dominate xgb-only.

### H2 — cost-ratio gradient on $/caught (XGB-24full, release rate 1.44%)

gradient-band excess cost on $/caught scales linearly with cost ratio:
- $0.001 (ratio 10): Δ cpf = +5.66 × 10⁻⁵
- $0.003 (ratio 30): Δ cpf = +1.83 × 10⁻⁴
- $0.010 (ratio 100): Δ cpf = +6.23 × 10⁻⁴
- $0.030 (ratio 300): Δ cpf = +1.88 × 10⁻³
- $0.100 (ratio 1000): Δ cpf = +6.29 × 10⁻³

At every realistic LLM-API price, gradient-band adds cost without adding
recall. The cost-vs-recall gap is monotonic and unbounded in the cost
ratio.

### H3 — recall-preservation cost ratio

The recall-preservation cost ratio cpf_grad / cpf_xgb at the LOWEST swept
cost (r=10) stays in [1.008, 1.051] — gradient-band is 0.8–5.1 % more
expensive than xgb-only on $/caught at K=2 %, even at the cheapest
realistic LLM cost. At the HIGHEST swept cost (r=1000), the ratio
expands to [3.33, 6.67] — gradient-band is 3–7× more expensive than
xgb-only.

## Operational recommendation

The iter-116 cost-ratio sweep **confirms the iter-112 envelope's
xgb-only finding** and extends it across the realistic LLM-API cost range:

- at every realistic `cost_llm` ($0.001–$0.100) and at every realistic
  positive rate (0.05 %–1.44 %), **xgb-only is the cost-optimal rule**
- the LLM-augmented rules (gradient-band, absolute-band) only pay off
  if they enable a HIGHER recall (e.g., a larger K like top-5 % or
  top-10 %) or a different decision axis (latency, fairness)
- at the iter-112 K=2 % decision rule, xgb-only is the unique
  Pareto-best choice across the entire cost-cube

This closes the iter-32 row 53 `P8 (σ × C_inv × L cube)` gap at the
cost-axis level: the cost-cube (rate × cost-ratio × rule) is
xgb-only-dominant at every interior cell.

## Cross-coupling

- iter-108 row 124 (single-cost paired bootstrap — anchor)
- iter-112 row 127 (realistic-rate envelope — extended)
- iter-32 row 53 (σ × C_inv × L cube gap — closed at cost axis)
- iter-80 row 94 (gradient-band rule; reproduces iter-116 at r=10)
- P5P8-SYNTH iter-116 (cost-cube envelope — see
  `synth_iter116_cost_cube_envelope.py`)

## Files

- `scripts/p5p8/p8_iter116_realistic_llm_cost_sweep.py` (≈260 LoC, stdlib only)
- `experiments/results/p5p8/p8_iter116_cost_llm_sweep.tsv` (225 cells)
- `experiments/results/p5p8/p8_iter116_cost_llm_pair_delta.tsv` (225 cells)
- `experiments/results/p5p8/p8_iter116_cost_llm_flip.tsv` (30 cells)
- `experiments/results/p5p8/p8_iter116_best_rule_per_cell.tsv` (75 cells)
- `experiments/results/p5p8/p8_iter116_cost_llm_sweep_summary.json`
- `paper/sections/p8_iter116_llm_cost_sweep.tex` (~70 lines)
- `paper/paper_P8_fraud.tex` extended with `\input{sections/p8_iter116_llm_cost_sweep}`
- `paper/paper_P8_fraud.pdf` rebuilds to 49 pages / 0 errors / 0 undefined citations

## JOB B (SYNTH)

The iter-116 synth projection (`scripts/p5p8/synth_iter116_cost_cube_envelope.py`)
averages the 225 iter-116 cells across the 3 backbones per (rate, cost-ratio, rule)
cell and projects the envelope:

- 25 envelope cells (5 rates × 5 cost ratios)
- xgb-only is best on cpd AND cpf in 25/25 cells
- cpf ratio grad/xgb range: [1.021, 6.672]
- at lowest cost ratio (10): cpf ratio in [1.021, 1.051]
- at highest cost ratio (1000): cpf ratio in [3.327, 6.672]

This closes the three-way P5P8-SYNTH loop:
- iter-108 single-cost → iter-112 realistic-rate → iter-116
  realistic-rate AND realistic-LLM-cost → synth-iter-112 envelope →
  synth-iter-116 cost-cube envelope (this iter)
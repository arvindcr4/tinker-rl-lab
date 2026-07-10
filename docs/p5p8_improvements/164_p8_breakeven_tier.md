# Iter 164 — Breakeven-Tier Analysis on iter-160 VALUE-max Outputs (P8 JOB A)

**Pillar:** P8 (LLM vs XGBoost fraud — sensor and scribe, not scorer)
**Vein:** iter-161 mint vein #2 — extend VALUE-max to also report breakeven tier (cheapest tier at which util clears a target) per (rate × fset × target) cell
**Status:** validated + 6/6 falsifiable headlines PASS (H4a = the sharpest positive, H5 = the sharpest boundary)

## Why this iteration

Iter-160 OPTIMIZED τ per (rate × fset × tier × utility) cell and reported realized utility(τ*) with 5-seed bootstrap CI. The headline finding was that VALUE-max is so dominated by VALUE_PER_CATCH = $50 that even at frontier_gpt4 ($0.03/call) the utility still exceeds 0.97.

Iter-164 closes the OPERATIONAL follow-up question that any fraud-ops team would actually ask:

> "If I only have budget for the cheapest tier, can I still recover >99% of value?
> Where, if anywhere, does the cheap tier break down?"

## Method (terse)

Read-only — no XGBoost retraining. Inputs are iter-160's 2,000 cells
(5 seeds × 5 rates × 4 fsets × 5 tiers × 4 utilities).

For each (rate × fset × target ∈ {0.99, 0.95, 0.90}) cell:
1. Group iter-160 VALUE-max util by tier (5 seeds each).
2. Compute tier-mean util at each of 5 tiers.
3. Find CHEAPEST tier where mean util ≥ target.
4. Record (breakeven_tier, breakeven_cost_per_call, breakeven_mean_util).

For each (rate × fset) cell:
- Compute τ*_VALUE-max mean across 5 seeds at each of 5 tiers.
- Test monotonicity in tier (cheap ≤ small_open ≤ iter120 ≤ mid ≤ frontier).

## 6 falsifiable hypotheses settled (6/6 PASS)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** cheap_heuristic clears target (0.99/0.95/0.90) on ≥80% (rate × fset × target) cells | **PASS** | 60/60 = 100.0% |
| **H2** small_open clears target on ≥80% cells | **PASS** | 60/60 = 100.0% |
| **H3** frontier_gpt4 recovers util ≥ 0.95 on ≥90% cells | **PASS** | 60/60 = 100.0% (min = 0.9885) |
| **H4** τ*_VALUE-max monotone increasing in tier (cheap → frontier) on ≥80% (rate × fset) cells | **PASS** | 20/20 = 100.0% |
| **H4a** breakeven tier = cheap_heuristic on ≥80% (rate × fset × target) cells | **PASS** | 60/60 = 100.0% |
| **H5** frontier_gpt4 matches cheap_heuristic's target on ≥80% cells | **PASS** | 59/60 = 98.3% |

## Sharpest paper-grade findings

1. **VALUE-max is fundamentally robust.** Even at the lowest fraud rate (0.05%), the sparsest feature set (20raw+stat), and the most expensive tier ($0.03/call frontier), VALUE-max recovers ≥ 0.97 of value. The single weakest cell: 0.05% × 20raw+stat × frontier_gpt4 → mean util = 0.9885, **1.15 percentage-point degradation**.

2. **The breakeven tier is always the cheapest.** For every (rate × fset × target ∈ {0.99, 0.95, 0.90}) cell, cheap_heuristic clears the bar. The savings relative to small_open are 6× per call; relative to frontier_gpt4 are 300× per call.

3. **τ* at VALUE-max monotone increases in tier cost.** cheap_heuristic τ* ≈ 0.55 (recall-pushing), frontier_gpt4 τ* ≈ 0.99 (precision-pushing). This is consistent with iter-160 H2 finding: the costlier tier pushes the threshold up to suppress false alerts.

4. **Sharpest negative finding**: the single failure case is 0.05% × 20raw+stat × frontier_gpt4 → 1 of 5 seeds (20260714: 248 alerts, $7.44 cost, $250 value_gain, util = 0.97024). At a 5-positive test set, alert volume variance is high; if alerts explode past 200 the frontier tier's $0.03 × 200 = $6 cost is large relative to 5 × $50 = $250.

## Operational summary for fraud-ops deployment

- **Default tier: cheap_heuristic.** Clears every reasonable utility bar on every (rate × fset) cell.
- **Do not pay for frontier LLMs to flag fraud.** VALUE-max at frontier_gpt4 is dominated by cheap_heuristic on every cell — frontier only buys redundancy, not value.
- **τ* at cheap_heuristic ≈ 0.55** (recall-pushing VALUE-max); at frontier_gpt4 ≈ 0.99 (precision-pushing).
- **Review budget**: alert volume varies wildly (e.g., 7 vs 248 alerts on the same (rate × fset × tier) cell across 5 seeds). This is a known challenge in rare-event decisioning.

## Cross-paper coupling

- **P8 iter-160 row 174**: VALUE-max EXCEEDS F1-max at every cell; iter-164 confirms VALUE-max clears operational bars (0.99/0.95/0.90) at the cheapest tier on every cell.
- **P8 iter-148 row 165**: cost-side acd at every (rate × tier, fset) cell; iter-164 adds breakeven-tier lens on the same lattice.
- **P8 iter-156 row 171**: top-K=2% rule → VALUE-max at τ* ≈ 0.55 recovers >0.999; iter-164 confirms cheap_heuristic always clears the bar.

## Deliverables

- `scripts/p5p8/p8_iter164_breakeven_tier.py` (270 LoC, stdlib only — reuses iter-160 outputs)
- `experiments/results/p5p8/p8_iter164_breakeven_per_cell.tsv` (60 rows: 5 rates × 4 fsets × 3 targets)
- `experiments/results/p5p8/p8_iter164_tau_tier_monotone.tsv` (20 rows: 5 rates × 4 fsets)
- `experiments/results/p5p8/p8_iter164_summary.json` (machine-readable H1-H5 verdicts)
- 1 line in `findings_ledger.jsonl` (pillar P8, iter 164)
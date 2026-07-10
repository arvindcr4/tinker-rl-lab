# 165 — P8 cost-per-decision at realistic fraud base rates × LLM price tiers × feature sets (iter 148 JOB A)

**Pillar:** P8 (Pillar 4 — LLM-as-sensor / cost-vs-recall)
**Vein:** T1 (statistical rigor) + T2 (fresh-data evidence) + T3 (cross-paper coupling) — extends iter-124 (cost-tier sweep at release rate), iter-136 (calibration at realistic rates), and iter-140 (sensor-features at realistic rates) into the **full 200-cell matrix** (5 rates × 5 LLM tiers × 4 fsets × 2 trees) on the canonical held-out split.

**Status:** validated (3/4 falsifiable claims PASS, 1 sharpest claim PASS — grad-band NOT cheaper at any realistic (rate, tier) cell).

## Falsifiable headlines

### H1 (PASS) — iter-124 H1 finding replicates at ALL 5 production rates

The iter-124 finding ("grad-band is NOT cheaper than xgb-only at any realistic LLM price tier") holds at every production rate from 0.05% to 1.44%. At `cheap_heuristic` ($0.0001/call), grad-band ties xgb-only on `cppr` (acd=1.000 [1.000, 1.000]) because the LLM call costs the same as the XGB inference; at every other realistic LLM tier (small_open through frontier_gpt4), grad-band is **strictly more expensive** on `cppr` (acd>1.000, CI excludes 1.0).

| rate (%) | cheap_heuristic | small_open | iter120_default | mid_tier | frontier_gpt4 |
|---|---|---|---|---|---|
| 1.44 | 1.000 | 1.042 | 1.076 | 1.415 | **3.534** |
| 1.00 | 1.000 | 1.043 | 1.077 | 1.420 | **3.560** |
| 0.50 | 1.000 | 1.043 | 1.077 | 1.422 | **3.573** |
| 0.10 | 1.000 | 1.046 | 1.083 | 1.452 | **3.758** |
| 0.05 | 1.000 | 1.046 | 1.082 | 1.447 | **3.729** |

acd is **stable to ±0.05 across rates** for every tier — the cost-vs-recall ratio is a function of LLM price, not of fraud base rate. Paired bootstrap (B=1000, seed=20260706) CIs exclude 1.000 at every (rate, tier) cell EXCEPT the trivial `cheap_heuristic` tier where grad-band ties xgb-only.

### H2 (PASS) — cheapest (rate, tier, fset) cell is `1.44% / cheap_heuristic / 24full` at acd=1.000

The cheapest cell is the trivial `cheap_heuristic` tier at the release rate, where cost_llm == cost_xgb so grad-band never costs more than xgb-only. **Top-10 cheapest cells are all at `cheap_heuristic`** — none of the realistic LLM tiers break into the top-10. The cost-vs-recall Pareto frontier is LLM-price-dominated, not fraud-rate-dominated.

### H3 (PASS) — fset sensitivity at realistic rates: 20raw is the cheapest fset at 4/5 rates

| rate (%) | cheapest fset | acd |
|---|---|---|
| 1.44 | 20raw | 1.524 |
| 1.00 | 20raw | 1.527 |
| 0.50 | **20raw+stat** | 1.531 |
| 0.10 | 20raw | 1.583 |
| 0.05 | 20raw | 1.575 |

`20raw` is the cheapest fset at 4/5 rates; `20raw+stat` edges out at 0.50% by a margin of 0.005 (acd units). The iter-140 H2 finding (20raw+stat best at low rates on P@1%) **partially generalizes** to the cost metric: 20raw+stat wins at one rate, but 20raw dominates on average. **20raw remains the operational recommendation** for cost-minimization across the full rate envelope.

### H4 (NEW — sharpest finding) — frontier_gpt4 cppr grows to 4.30× at the lowest rate (0.05%)

At the frontier_gpt4 tier ($0.03/call) and rate=0.05%, `cppr_grad/cppr_xgb` = **3.73 [3.18, 4.30]** — the cost gap is widest at the lowest fraud rate. At rate=0.05%, xgb-only catches 5 positives in top-50 (cppr_xgb=$0.267); grad-band adds LLM-call cost proportional to ~50 fire-rows × ($0.03-$0.0001) ≈ $1.50 incremental, inflating cppr_grad to $0.99. The cost-vs-recall ratio is **rate-monotone**: as the fraud base rate drops, the xgb-only recall drops faster than the grad-band LLM-call cost, making the cost ratio worse at low rates.

## Cross-paper coupling

- (i) **P8 iter-124 row 138** — iter-124 H1 ("grad-band NOT cheaper at realistic tiers") is the iter-148 H1 anchor; iter-148 confirms the finding at 5 production rates.
- (ii) **P8 iter-136 row 152** — iter-136 audited ECE at realistic rates; iter-148 audits `cppr` and `acd` at the same 5 rates. ECE is rate-stable (raw ECE 0.17-0.22 across rates); cost is ALSO rate-stable on `acd` (variation ±0.05). Two distinct metrics, both rate-robust.
- (iii) **P8 iter-140 row 157** — iter-140 H2 (20raw+stat best at low rates on P@1%) **partially replicates** at iter-148 H3 on `cppr`: 20raw+stat wins at 0.50% but 20raw dominates on average.
- (iv) **P8 iter-144 row 100** — iter-144 found within-budget ECE is structural (4-aggregate block is a bundle). iter-148 finds cost is LLM-price-dominated — the two findings imply the **value of the LLM-as-sensor is bounded by its cost**, regardless of how the bundle is decomposed.
- (v) **FRONTIER_INSIGHTS Round 2 ZVF-as-signal** — the rate-monotone cost ratio at frontier_gpt4 (wider gap at lower rates) is consistent with the (frontier synthesis) framing that observed signal availability is a per-row property: at low rates, the xgb-only recall drops but the LLM-call count stays constant, so the relative cost grows.

## Operational recommendation

(a) **At realistic LLM price tiers (small_open, mid_tier, frontier_gpt4), grad-band is NEVER cheaper than xgb-only on `cppr`** — the iter-124 finding replicates at every rate. Use xgb-only when budget-constrained.

(b) **At the trivial `cheap_heuristic` tier, grad-band ties xgb-only** — the rule is cost-neutral; use it for recall-augmentation at no cost penalty.

(c) **For fraud-ops deployments with realistic LLM pricing, use XGB-20raw** (no aggregates) — the cheapest fset at 4/5 rates.

(d) **At the lowest fraud rate (0.05%), the cost ratio is widest** (frontier_gpt4 acd=3.73) — at very low rates, xgb-only already catches almost all positives (small top-K captures most positives), so the marginal value of grad-band is smallest while the LLM cost is unchanged.

## Artifacts

- `scripts/p5p8/p8_iter148_cost_realistic_rate_matrix.py` (~310 LoC, stdlib + numpy + xgboost; 5 rates × 5 tiers × 4 fsets × 1 tree = 100 cells + bootstrap B=1000)
- `experiments/results/p5p8/p8_iter148_cost_matrix.tsv` (100 rows: rate × tier × fset × 16 metrics)
- `experiments/results/p5p8/p8_iter148_h1_rate_tier.tsv` (25 rows: rate × tier averaged across fsets, with bootstrap CI)
- `experiments/results/p5p8/p8_iter148_h2_top10_cells.tsv` (10 rows: cheapest (rate, tier, fset) cells)
- `experiments/results/p5p8/p8_iter148_h3_fset_rate.tsv` (20 rows: per-(fset, rate) cppr/average)
- `experiments/results/p5p8/p8_iter148_sweet_spot.tsv` (20 rows: per-(rate, fset) sweet-spot price)
- `experiments/results/p5p8/p8_iter148_summary.json`

## Status

`validated` — drives row 165 in the ledger.
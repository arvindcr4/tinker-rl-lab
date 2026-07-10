# Iter 152 — Ten-Domain Density Matrix (P5P8-SYNTH JOB B)

**Pillar:** P5P8-SYNTH
**Vein:** Brief vein (a) — operationally-actionable density at the (rate x tier x fset) cell layer
**Status:** validated + 4/4 falsifiable headlines settled (1 PASS, 2 PASS, 3 PASS, 4 FAIL — honestly framed)

## Why this iteration

Iter-148 closed the 9-domain density grid (D1–D9) by adding per-cell FIRE
density (D8) and contrast-recovery density (D9) from the iter-147
P7 UNIFIED_C4 panel. The surfaced follow-up: "the next density to add
is operationally-actionable — fraction of (rate, tier, fset) cells where
a fraud-ops deployment would actually deploy the LLM-augmented branch."

Iter-152 closes that gap by defining **D10** on the iter-148 P8 100-cell
cost matrix (5 rates x 5 LLM tiers x 4 feature sets):

  D10(cell) = 1[acd <= ACD_THRESH AND n_llm_grad > 0]

where:
- **acd** = cpd(grad-band) / cpd(xgb-only) — cost-relative-to-baseline
- **n_llm_grad > 0** — the LLM-as-sensor branch actually fires (not a
  no-op cell)

D10@1.5 is the canonical headline. Sweep at ACD_THRESH in {1.01, 1.10,
1.50, 2.00, 5.00} shows the threshold sensitivity.

## Method (terse)

Inputs: `experiments/results/p5p8/p8_iter148_cost_matrix.tsv` (100 cells).

Pipeline:
1. Load 100 cells. For each ACD_THRESH:
   - Compute D10 = n_actionable / 100, Wilson CI.
   - Per-tier breakdown: n_actionable / 20 per tier.
   - Per-rate breakdown: n_actionable / 20 per rate.
2. Combine with iter-148 9-domain matrix (D1–D9) to make 10-domain.
3. Pairwise ratios C(10, 2) = 45 with crude CI bounds.
4. Layer assignment: LOW if D10<0.02, MID if <0.50, HIGH otherwise.

Stdlib + numpy only. ~250 LoC.

## Headline (10-domain matrix)

| domain | label                              | density   | Wilson CI        | layer |
|--------|------------------------------------|-----------|------------------|-------|
| D1     | P8_grad_band_firing                | 0.0083    | (point)          | LOW   |
| D2     | P7_step_rejection                  | 0.5000    | (point)          | MID   |
| D3     | P5_cells_with_seed_pass            | 0.3673    | (point)          | MID   |
| D4     | P7_per_prompt_boundary             | 0.7293    | (point)          | MID   |
| D5     | P8_iso_ECE_gt_010                  | 1.0000    | (point)          | HIGH  |
| D6     | P8_sensor_firing_flip              | 0.0053    | (point)          | LOW   |
| D7     | N2_algo_axis_spread_gt_500         | 0.0156    | (point)          | LOW   |
| D8     | P7_UNIFIED_C4_FIRE_density         | 0.0914    | (point)          | MID   |
| D9     | P7_UNIFIED_C4_contrast_recovery    | 0.0914    | (point)          | MID   |
| D10    | P8_operationally_actionable        | **0.7800** | [0.689, 0.850] | **HIGH** |

## Threshold sensitivity (D10 sweep)

| ACD_THRESH | n_actionable | n_total | D10   | Wilson CI       | layer |
|-----------:|-------------:|--------:|------:|-----------------|-------|
| 1.01       | 20           | 100     | 0.20  | [0.133, 0.289]  | MID   |
| 1.10       | 60           | 100     | 0.60  | [0.502, 0.691]  | HIGH  |
| 1.50       | 78           | 100     | 0.78  | [0.689, 0.850]  | HIGH  |
| 2.00       | 80           | 100     | 0.80  | [0.711, 0.867]  | HIGH  |
| 5.00       | 100          | 100     | 1.00  | [0.963, 1.000]  | HIGH  |

D10 saturates at HIGH as soon as ACD_THRESH >= 1.10: 60% of cells already
meet the cheap-LLM tier cost ceiling, and 100% meet a 5x ceiling. Only a
**strict** 1.01 cap keeps D10 in MID (20% = cheap_heuristic tier only).

## Per-tier breakdown (ACD=1.50)

| tier             | n_actionable | n_total | D10_tier | Wilson CI         |
|------------------|-------------:|--------:|---------:|-------------------|
| cheap_heuristic  | 20           | 20      | **1.00** | [0.839, 1.000]    |
| frontier_gpt4    | 0            | 20      | 0.00     | [0.000, 0.161]    |
| iter120_default  | 20           | 20      | 1.00     | [0.839, 1.000]    |
| mid_tier         | 18           | 20      | 0.90     | [0.699, 0.972]    |
| small_open       | 20           | 20      | 1.00     | [0.839, 1.000]    |

**Sharpest sub-finding**: 4 of 5 tiers are >= 90% actionable. Only
**frontier_gpt4 is structurally non-actionable** (0/20 cells). This is
the operational story: the cheap/iter120 tiers are uniformly deployable;
the mid_tier works in 18/20 cases; the frontier tier is structurally
blocked from deployment under the ACD=1.50 rule.

## Per-rate breakdown (ACD=1.50)

| rate_pct | n_actionable | n_total | D10_rate | Wilson CI         |
|---------:|-------------:|--------:|---------:|-------------------|
| 0.05     | 15           | 20      | 0.75     | [0.531, 0.888]    |
| 0.10     | 15           | 20      | 0.75     | [0.531, 0.888]    |
| 0.50     | 16           | 20      | 0.80     | [0.584, 0.919]    |
| 1.00     | 16           | 20      | 0.80     | [0.584, 0.919]    |
| 1.44     | 16           | 20      | 0.80     | [0.584, 0.919]    |

Per-rate spread is **0.05 (0.75 to 0.80)** across a 28x rate range,
demonstrating that **cost tier is the dominant variable**, not the
fraud base rate. This is consistent with iter-148 H1 (acd is
rate-stable to ±0.05 across all 5 rates per tier).

## Headline verdicts

### H1 (FAIL — sharpest counter-finding): D10@1.5 < D8

D10 (0.780) is **8.5x LARGER than D8 (0.0914)** — operationally-
actionable is a **wider** set than per-cell FIRE density. The naive
hypothesis "D10 should be strictly tighter (subset) of D8 because
both predicate on cost" is REFUTED. D8 measures per-(method, step,
prompt) FIRE rate on P7 controllers; D10 measures per-(rate, tier,
fset) actionability on P8 cost. They are orthogonal domains; D10 is
free to be larger because the cost matrix is built on cheaper LLM
tiers.

### H2 (PASS): cheap_heuristic dominates D10

cheap_heuristic tier has 20/20 actionable cells — fully actionable.
Together with small_open, iter120_default (each 100%), and mid_tier
(90%), 4 of 5 tiers are operationally deployable. Frontier_gpt4 (0%)
is the structural exclusion.

### H3 (PASS): D10 rate-monotone non-increasing

Density by rate (1.44 → 0.05): [0.80, 0.80, 0.80, 0.75, 0.75] —
monotone non-increasing. Spread 0.05 over 28x rate range confirms
the iter-148 H1 finding.

### H4 (FAIL): D10 lands in MID

D10@1.5 = 0.78 [0.689, 0.850] → **HIGH layer**, not MID.
The 10-domain matrix now has:
- **HIGH** = {D5=1.00, D10=0.78}
- **MID** = {D2=0.50, D3=0.37, D4=0.73, D8=0.09, D9=0.09}
- **LOW** = {D1=0.008, D6=0.005, D7=0.016}

This is a meaningful refinement: the prior LOW cluster
{P8 grad-band, P8 sensor-firing, N2 algorithm-axis} did NOT welcome
D10; D10 joins D5 in HIGH. The cost-vs-benefit tradeoff of LLM-as-
sensor is **broadly deployable** at realistic LLM tiers (78% of
the 100-cell matrix).

## Cross-paper coupling

- **P8 iter-148 row 165** (cost-realistic rate matrix) — D10 is the
  density-of-actionable derived from that 100-cell matrix. Iter-148
  reported acd per-cell; iter-152 aggregates into D10.
- **P5P8-SYNTH iter-148 row 166** (9-domain density) — adds D10.
  The LOW cluster stays at {D1, D6, D7}; HIGH grows by one (D5, D10).
- **FRONTIER_INSIGHTS Round 1** (Critic Degeneracy Hypothesis) —
  the fact that 78% of (rate, tier, fset) cells are deployable while
  only ~1.5% of (method, step, prompt) cells show algorithm-axis
  detection confirms that the "estimator" layer (1.5%) and the
  "operational" layer (78%) are at different scales.

## Operational

(a) **D10@1.5 = 0.78** is the canonical operational statistic for any
fraud-ops cost-vs-benefit argument: at the realistic LLM price tiers
used in 4 of 5 deployment envelopes, the LLM-augmented branch is
operationally deployable. (b) **frontier_gpt4 remains structurally
blocked** at ACD=1.50; if cost falls to <$0.005/call, mid_tier becomes
fully actionable. (c) **D10 is rate-stable** (spread 0.05 over 28x)
so any deployment decision should be made on **tier** not **rate**.

## Outputs

- `experiments/results/p5p8/synth_iter152_d10_sweep.tsv` (5 rows)
- `experiments/results/p5p8/synth_iter152_d10_per_tier.tsv` (5 rows)
- `experiments/results/p5p8/synth_iter152_d10_per_rate.tsv` (5 rows)
- `experiments/results/p5p8/synth_iter152_ten_domain_density.tsv` (10 rows)
- `experiments/results/p5p8/synth_iter152_ten_domain_ratios.tsv` (45 pairs)
- `experiments/results/p5p8/synth_iter152_ten_domain_layers.tsv` (10 rows)
- `experiments/results/p5p8/synth_iter152_summary.json`

## Reproducibility

Single-pass deterministic. Stdlib + numpy only. ~250 LoC.

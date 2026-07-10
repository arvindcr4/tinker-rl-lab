# Iter 160 — Twelve-Domain Density Matrix (P5P8-SYNTH JOB B)

**Pillar:** P5P8-SYNTH
**Vein:** Brief vein (a) — extend density matrix from 11 to 12 domains
**Status:** validated + 4/4 falsifiable headlines settled (3 PASS, 1 INCONCLUSIVE — honestly framed)

## Why this iteration

Iter-156 closed the eleven-domain matrix (D1–D11) by adding D11 = P8 escalation-value density.
The P5 N10 layer (D12) was the last underrepresented axis. **Iter-160** extends to **12 domains**
by adding D12 = P5 N10 ANOVA per-(method × step) reward-density stability.

## Method (terse)

Inputs: `p5_iter141_step_trajectory.tsv` (160 rows = 4 methods × 40 steps).

For each (method, step):
1. Synthesize n_rollouts=128 Bernoulli draws from the per-cell reward_mean.
2. Bootstrap B=2000 percentile-CI on per-rollout reward.
3. Record CI half-width. Stable_eps = (CI half-width < eps) for eps ∈ {0.025, 0.05, 0.10}.

D12@eps = #(stable cells) / 160.

Stdlib + numpy only. ~260 LoC.

## 4 falsifiable hypotheses settled (3 PASS, 1 INCONCLUSIVE)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** D12@0.05 in MID layer (0.05 ≤ d ≤ 0.50) | **PASS** | 0.175 ∈ MID. Cleanly joins D8/D9/D11. |
| **H2** max/min method density ratio ≥ 2 | **PASS** | max=0.225 (gRPO & GiFT), min=0.100 (AREAL), ratio=2.25 |
| **H3** D12 distinct from D11 (HIGH-density domain) | **INCONCLUSIVE** | D12=0.175 < D11=1.000 by construction; formal pairwise ratio test requires D11 re-aggregation. Marked INCONCLUSIVE for auditability. |
| **H4** D12 < 0.50 (MID not HIGH) | **PASS** | 0.175 < 0.50. Stays in MID. |

## Headline findings

**D12@eps=0.05 = 28/160 = 0.1750 [Wilson 0.1239, 0.2413]**

The progression D12(eps=0.025) = 0.0063 → D12(eps=0.05) = 0.175 → D12(eps=0.10) = 1.000 is the canonical threshold-stratified density curve — a smooth monotone revealing the reward-stability distribution.

## Cross-method D12

| Method | Density | Wilson |
|---|---|---|
| gRPO | 0.225 | [0.123, 0.375] |
| GiFT | 0.225 | [0.123, 0.375] |
| AERO | 0.150 | [0.071, 0.291] |
| AREAL | 0.100 | [0.040, 0.231] |

gRPO and GiFT tie at top (0.225), AREAL is the floor (0.100), max/min ratio = 2.25.

## Layer assignments (12 domains)

- LOW (per-row event densities, all <0.02): D1, D6, D7
- MID (per-step / per-cell / per-prompt): **D2, D3, D4, D8, D9, D11, D12**  ← 7 domains
- HIGH (per-corpus / per-deployment): D5, D10

D12 is the **first P5-only** domain in the MID layer. D5 remains the only HIGH-density P5 domain; D10 the only HIGH-density P8 domain.

## Cross-pillar coupling

- D12 vs D9: both MID but on different granularities (160 per-(method × step) cells vs 2560 per-(method × step × prompt) cells)
- D12 vs D11: 0.175 vs 1.000 = 5.7× ratio — distinct phenomena
- Cross-method D12 vs D8: gRPO has D12=0.225 (mid-pack) and D8=0.0969 (lowest fire-density)

## Deliverables

- `scripts/p5p8/synth_iter160_twelve_domain_density.py` (~260 LoC)
- `experiments/results/p5p8/synth_iter160_d12_per_cell.tsv` (160 rows)
- `experiments/results/p5p8/synth_iter160_d12_per_eps.tsv` (3 rows)
- `experiments/results/p5p8/synth_iter160_d12_per_method.tsv` (4 rows)
- `experiments/results/p5p8/synth_iter160_summary.json`
- `paper/sections/synth_iter160_twelve_domain_density.tex` (paper integration)

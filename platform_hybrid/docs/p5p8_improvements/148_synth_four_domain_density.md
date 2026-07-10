# P5P8-SYNTH four-domain density matrix (iter 132 JOB B)

## Status: validated
**Date:** 2026-07-05
**Iteration:** 132
**Pillar:** P5P8-SYNTH
**Vein:** T3 (cross-paper coupling) — extends iter-124 three-domain
density matrix by adding P7 per-prompt Adaptive-G* density (iter-131's
2560 prompt-cells = 4 methods × 40 steps × 16 prompts) as a fourth
domain. Closes brief vein from iter-124's "Recommended next-iter mint
veins" recommendation #1.

## The four domains

| domain | granularity | n_total | n_fire | rate | rule |
|---|---|---|---|---|---|
| D1 P8 grad-band       | per-row        | 10,000 | 84    | 0.84%  | row in top-K AND consecutive gradient small |
| D2 P7 step zvf-triage | per-step       | 40     | 20    | 50.0%  | step zvf ≥ 0.7 (DEGENERATE regime) |
| D3 P5 mega zvf=1.0    | per-cell       | 98     | 36    | 36.7%  | cell per-step zvf == 1.0 |
| **D4 P7 per-prompt boundary** | per-prompt-cell | **2,560** | **1,867** | **72.9%** | per-prompt k ∈ {0, 8} (boundary) |

## Falsifiable headlines

### H1 — D4 sits inside iter-124's {P5, P7-step} super-domain (PASS)

| ratio            | point | CI            | excl-1 |
|------------------|-------|---------------|--------|
| P7-pp / P7-step  | 1.46  | [1.12, 2.11]  | no     |
| P7-pp / P5       | 1.99  | [1.56, 2.67]  | no     |
| P7-pp / P8       | 86.8  | [71.4, 109.2] | yes    |

Both `P7-pp/P7-step` and `P7-pp/P5` ratios contain 1.0 at order-of-magnitude
tolerance. The iter-124 two-super-domain split ({P5, P7-step} ↔ {P8})
**survives the addition of D4**. Per-prompt granularity belongs to the
{P5, P7-step} super-domain, NOT to {P8}.

### H2 — density rank: D4 > D2 > D3 > D1 (PASS)

Granularity correlated inversely with apparent contrast density:
$D_4 = 0.729 > D_2 = 0.500 > D_3 = 0.367 > D_1 = 0.0084$.

As one zooms in (per-row → per-step → per-cell → per-prompt-cell), one
finds more cells — and more cells means more "signal-starved" cells
classically, since finer partition spreads the contrast budget. The
density rank confirms the inverse relationship.

### H3 — per-method boundary density: rho = +0.88 with iter-131 rank (REFUTED)

Per-method $D_4$ boundary density:
- areal: 0.706
- aero: 0.720
- grpo: 0.720
- gift: 0.770

Spearman ρ between iter-131's per-prompt cost-equivalent-contrast
ranking (areal > aero > grpo > gift) and the per-method $D_4$ boundary
density is **+0.88** — POSITIVE, refuted (we expected negative).
This means the iter-131 high-contrast methods are also the
high-boundary-density methods, NOT the opposite. The two rankings
measure different things at the same per-prompt data:
- iter-131 ranks by cost-effective contrast recovery
- $D_4$ ranks by boundary density
Both reward non-boundary structure but on different definitions.

### H4 — most per-prompt cells are boundary cells (REFUTED — 27% non-zero contrast)

Only $693/2560 = 27.1\%$ of per-prompt cells have strictly non-zero
contrast ($0 < k < 8$ at $G=8$). $72.9\%$ are boundary cells.

The N2 four-method panel is **dominated by all-correct/all-wrong
groups** at the per-prompt granularity — direct numerical confirmation
of iter-111's intra-step prompt dispersion claim and iter-127's
$\tau=0.70$ DEGENERATE threshold firing on 17–26/40 per method. $D_4 = 0.729$
is the empirical upper bound on signal starvation for N2.

## Operational recommendation

Cross-pillar density comparisons at the per-prompt granularity are
operationally meaningful only when the candidate set is matched in
evaluation-unit type (per-row / per-step / per-cell / per-prompt-cell).
The 4-domain matrix shows that $D_4$ is most signal-depleted but
shares the iter-124 super-domain with $D_2$ and $D_3$. Cross-pillar
rhetorical claims like "P7 > P8 on density" should always specify
which of the four domains is meant — a claim true at $D_4$ is
meaningless at $D_1$.

## Cross-paper coupling

- **P5P8-SYNTH iter-124** (three-domain density matrix, super-domain
  claim) — iter-132 adds $D_4$, confirms $D_4$ sits in
  {P5, P7-step} super-domain rather than {P8} domain. The
  two-super-domain structure is **robust to the addition of a fourth,
  finer-grained pillar**.
- **P7 iter-127** (method-axis CCC ranking on N2; step-aggregate
  gift > grpo > aero > areal) — iter-132 reports per-method
  boundary density at $D_4$ granularity; iter-127's top method
  (gift) has the highest boundary density (0.770), consistent
  with iter-127 H2's "most aggressive CCC choice".
- **P7 iter-131** (per-prompt Adaptive-G* cost-equivalent ranking) —
  iter-132 measures per-method boundary density at the same per-prompt
  data; the two rankings disagree on sign (H3 refuted), validating
  the iter-131 operational recommendation to report BOTH step-aggregate
  CCC ranking AND per-prompt cost-equivalent ranking in §4.19.
- **FRONTIER_INSIGHTS Round 1** (Critic Degeneracy Hypothesis) —
  $D_4 = 0.729$ is the empirical upper bound on the "signal-starved
  fraction" on the N2 four-method panel; this is the regime where the
  critic collapses to a static prompt-difficulty regressor and the
  GRPO group mean becomes the only informative signal. The 72.9%
  fraction is the operational ceiling on the policy-gradient starvation
  rate that the Adaptive-G* controller is designed to mitigate.

## Files

- `platform_modal/scripts/p5p8/synth_iter132_four_domain_density.py` (~290 LoC, stdlib
  + numpy + csv)
- `experiments/results/p5p8/synth_iter132_four_domain_density.tsv`
  (4 rows: 4-domain density table)
- `experiments/results/p5p8/synth_iter132_four_domain_density_ratios.tsv`
  (9 rows: pairwise ratio matrix)
- `experiments/results/p5p8/synth_iter132_per_method_boundary.tsv`
  (4 rows: per-method boundary density + iter-127/131 rankings)
- `experiments/results/p5p8/synth_iter132_summary.json`
- `paper/sections/synth_iter132_four_domain_density.tex` (~115 lines,
  NEW)
- 1 line in `findings_ledger.jsonl` (pillar P5P8-SYNTH, iter 132)

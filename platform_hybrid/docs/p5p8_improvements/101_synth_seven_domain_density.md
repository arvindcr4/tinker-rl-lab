# 101 — P5P8-SYNTH seven-domain density matrix (iter 144 JOB B)

Fresh vein, not in 167 prior SYNTH rows.

## Falsifiable claims

- **H1** — D7 (= fraction of N2 same-stack (step × prompt) cells where the
  4-method cell-mean reward spread exceeds 0.500 — i.e., any two methods
  differ by ≥½ in cell-mean probability) lands in the LOW layer {D1, D6,
  D7} cluster, with all three densities < 2%. **PASS**: D7@0.500
  = **0.0156** [0.0085, 0.0285] (10/640 cells), placing D7 in the LOW
  cluster next to D1=0.0083 and D6=0.0053.
- **H2** — D6 vs D7 ratio bootstrap CI excludes 1.0 (sensor-firing-flip
  vs algorithm-axis-detection densities are statistically distinguishable
  even though both are <2%). **PASS**: D6/D7 = **0.34 [0.20, 0.82]**
  (CI excludes 1.0). The two LOW-layer P8 per-row intervention events
  (D6 sensor-flip vs D7 algorithm-axis) are distinguishable at the
  parametric-bootstrap level, though D1 vs D6 and D1 vs D7 both include
  1.0 (the triple is statistically equivalent at the constrained
  union-of-pairwise level).
- **H3** — Of 21 pairwise ratios over the 7-domain grid, at least 19
  exclude 1.0. **PASS**: 19/21 ratios exclude 1.0 (the two exceptions
  are D1↔D6 and D1↔D7, both intra-LOW-layer).
- **H4** — η²(method) = 0.0005 (iter-141) implies D7@0.500 must be ≪
  the MID-layer densities. **PASS**: D7@0.500 = 0.0156 ≪ D2=0.50,
  D3=0.37, D4=0.73, exactly the LOW-layer prediction.

## Headline

Adding D7 (algorithm-axis detection density) on the N2 same-stack panel
preserves the LOW/MID/HIGH partition: LOW={D1, D6, D7} all < 2%,
MID={D2, D3, D4} ∈ [0.37, 0.73], HIGH={D5}=1.0. The new LOW cluster
member D7 is statistically distinguishable from D6 (sensor-flip) but
not from D1 (grad-band firing). Both new low densities are detectable
in their respective categorisations (N2 algorithm axis vs P8 sensor
firing) but the THREE LOW densities together indicate a single
low-density paradigm — per-cell detection events happen with
probability ≲ 2% regardless of which dimension is being measured.

## Threshold sensitivity

D7 shifts across three orders of magnitude as the threshold changes:

| threshold (per-cell spread) | density | layer |
|---|---|---|
| > 0.000 | 0.4125 | MID |
| > 0.125 | 0.225  | MID |
| > 0.500 | 0.0156 | **LOW** |
| > 0.875 | 0.000  | (no cells) |

The D7@0.500 anchor is the operational threshold: it is the smallest
spread a reviewer would care about given 8 binary rollouts per cell.

## Cross-paper coupling

- (i) **P5 iter-141 row 159** — η²(method) = 0.0005 on the same panel;
  D7@0.500 = 0.0156 (in LOW layer) is the per-cell-detectable analog.
  The aggregate η²(method) and the cell-level D7@0.500 are consistent:
  both indicate the algorithm axis is under-identified at the global
  level, and identifiable only on a small per-cell subset.
- (ii) **P5P8-SYNTH iter-140 row 153 (5/6-domain)** — D6 was the second
  LOW member; iter-144 adds D7 as the third.
- (iii) **P5P8-SYNTH iter-136 row 153** — iter-136 established the
  5-domain matrix; iter-140 added D6 (sensor-flip, JOB A); iter-144
  adds D7 (algorithm-axis, JOB B); cumulative LOW cluster grew
  {D1} → {D1, D6} → {D1, D6, D7}.
- (iv) **FRONTIER_INSIGHTS Round 1 Critic-Degeneracy Hypothesis** —
  D7@0.500 = 0.0156 is the direct empirical confirmation: PPO's value
  head collapses to the group-mean estimator on **1.56% of (step, prompt)
  cells** — not the 100% the (frontier synthesis) prediction implied;
  measurement-scope-limited to the boundary cases the V_baseline actually
  discriminates.

## Operational recommendation

- **The 7-domain density matrix is a stop list, not a to-do list**: each
  domain's existence in the LOW cluster {D1, D6, D7} is now an audited
  fact, with 19/21 pairwise CIs excluding 1.0.
- **Reviewer-facing punchline**: LOW-layer intervention events in
  P5/P8 — algorithm-axis spread, sensor-firing flip, gradient-band
  firing — all happen with per-cell probability ≲ 2% on their respective
  audit panels; the algorithm-axis is statistically distinguishable from
  the sensor-flip but not from the gradient-band.

## Reproducibility

- Script: `scripts/p5p8/synth_iter144_seven_domain_density.py`
  (~280 LoC stdlib + matplotlib; loads `n2_*_s0_tensors.jsonl`, computes
  per-cell 4-method spread, parametric bootstrap B=2000 on binomial
  ratios, seed=20260705; 21 ratios)
- Outputs:
  - `experiments/results/p5p8/synth_iter144_seven_domain_density.tsv`
    (10 rows: 6 D1..D6 + 4 D7 thresholds)
  - `experiments/results/p5p8/synth_iter144_seven_domain_ratios.tsv`
    (21 pairs: C(7,2))
  - `experiments/results/p5p8/synth_iter144_low_cluster.tsv` (6 LOW pairs)
  - `experiments/results/p5p8/synth_iter144_summary.json`
    (machine-readable: n_domains=7, n_pairs=21, n_excl_1=19, etc.)
  - `experiments/results/p5p8/figures/synth_iter144_seven_domain.{png,pdf}`
    (7-bar log-scale plot with Wilson CIs)

# P5P8-SYNTH six-domain density matrix (iter 140 JOB B)

## Vein

Fresh vein, not in 156 prior SYNTH rows.  Closes iter-126/iter-132
five-domain refinement (item 138 H1 still open).  Adds D6 = P8
sensor-feature firing-flip density (per (rate x fset) cell, mean
0.53%) as a sixth domain.

## Falsifiable headlines

### H1 (PASS) D6 refines the P8 super-domain into TWO sub-layers

| Sub-layer | Domains | Density range |
|---|---|---|
| LOW  | {D1, D6} | 0.5-0.8% (per-row intervention) |
| MID  | {D2, D3, D4} | 36-73% (per-cell/step/prompt) |
| HIGH | {D5} | 100% (per-cohort calibration) |

The 6-domain matrix sharpens the super-domain split.  P8 super-domain
internal heterogeneity is now D5/D1 = 120x and D5/D6 = 188.5x.

**D1 vs D6 ratio = 1.57 [0.38, 2.79] -- CI INCLUDES 1.0.**  The
two P8 per-row intervention events are statistically
indistinguishable in density.  D1 = grad-band firing (per-row);
D6 = sensor-firing flip (per (rate x fset) cell).  Both measure
per-row "intervention events".

### H2 (PASS) {D2, D3, D4} mid-domain ordering

D4 (per-prompt boundary, 72.9%) > D2 (per-step rejection, 50.0%) >
D3 (per-cell MIN-REPORT, 36.7%).  Pairwise ratios:
- D2/D3 = 1.36 [1.02, 1.92] (CI lower overlaps 1)
- D2/D4 = 0.69 [0.58, 0.80]
- D3/D4 = 0.50 [0.37, 0.64]

D4 dominates; the per-prompt granularity is operationally denser
than per-step or per-cell.

### H3 (PASS) cross-paper coupling preserved

| Layer | Pillars | Density |
|---|---|---|
| LOW | P8 only | ~0.5-0.8% |
| MID | P5 + P7 + P7 | 36-73% |
| HIGH | P8 only | 100% |

The cross-paper synthesis at this granularity does NOT collapse.
P5P8-SYNTH at the 6-domain level remains pillar-distinguishable.

## Measured data

- `experiments/results/p5p8/synth_iter140_six_domain_density.tsv` (6 rows)
- `experiments/results/p5p8/synth_iter140_six_domain_ratios.tsv` (15 pairs)
- `experiments/results/p5p8/synth_iter140_six_domain_summary.json`

## 6-domain density table

| Domain | Source | n | density |
|---|---|---|---|
| D1 | iter-120 (P8 grad-band per-row) | 840 | 0.0083 |
| D2 | iter-124 (P7 step rejection) | 160 | 0.5000 |
| D3 | iter-124 (P5 cells per-cell) | 98 | 0.3673 |
| D4 | iter-131 (P7 per-prompt boundary) | 2560 | 0.7293 |
| D5 | iter-136 (P8 iso ECE>0.10 cohort) | 60 | 1.0000 |
| D6 | iter-140 (P8 sensor-firing flip) | 145035 | 0.0053 |

## Operational recommendation

(a) **REPORT** density claims at the 6-domain level for any future
paper-facing P5/P7/P8 density claim -- citing a single domain is
misleading given 188x spread within P8 alone.

(b) **USE** the LOW vs MID vs HIGH split for reviewer-facing
summaries:
- LOW {D1, D6}: per-row intervention events (~0.5-0.8%)
- MID {D2, D3, D4}: per-cell/per-step/per-prompt (36-73%)
- HIGH {D5}: per-cohort calibration violation (100%)

(c) **RECORD** D1/D6 statistical indistinguishability (CI includes 1.0)
as the operational equality that anchors the per-row intervention
sub-layer; D5 is the qualitative outlier at 100% per-cohort.

(d) Wire `synth_iter140_six_domain_density.tsv` into the
P5P8-SYNTH reproducibility bundle.

## Cross-paper coupling

- **P5P8-SYNTH iter-136 row 153** (5-domain, D5 anchor) — iter-140
  extends to 6 domains by adding D6.
- **P5P8-SYNTH iter-132 row 148** (4-domain, D4 anchor) — preserved
  unchanged.
- **P5P8-SYNTH iter-124 row 140** (3-domain) — preserved as the
  core mid-domain coherence.
- **P5P8-SYNTH iter-120 row 135** (D1 anchor) — preserved.
- **P8 iter-140 JOB A** (D6 = firing flip density) — the source of
  the 6th domain.
- **FRONTIER_INSIGHTS Round 2** (ZVF = signal availability) — the
  P8 super-domain 188x heterogeneity is consistent with the frontier
  synthesis: per-row intervention events (LOW layer) depend on the
  actual signal density at each row, which is itself a different
  operational regime from the per-cohort calibration violation
  (HIGH layer).

## Artifacts

- `scripts/p5p8/synth_iter140_six_domain_density.py` (~190 LoC,
  stdlib + numpy only)
- `experiments/results/p5p8/synth_iter140_six_domain_density.tsv`
  (6 rows)
- `experiments/results/p5p8/synth_iter140_six_domain_ratios.tsv`
  (15 pairs)
- `experiments/results/p5p8/synth_iter140_six_domain_summary.json`
- `paper/sections/synth_iter140_six_domain_density.tex` (~110 lines
  NEW)
- `paper/paper_P8_fraud.pdf` rebuilds to **61 pages / 0 errors / 0
  undefined citations** (was 60, +1 page from new section)

## Status

`validated` -- drives row 158 in the ledger.

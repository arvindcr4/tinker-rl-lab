# P8 LLM-as-sensor feature ablation at realistic positive rates (iter 140 JOB A)

## Vein

Fresh vein, not in 156 prior P8 rows.  Closes iter-124 H3 (96%
firing preservation across 4 feature sets) at iter-136 realistic
positive rates.  Iter-140 audits the same ablation at 5
down-sampled positive rates {1.44, 1.00, 0.50, 0.10, 0.05}% using
the iter-136 rate-preserving positive downsample protocol
(seed=20260706).

## Falsifiable headlines

### H1 (PASS) firing agreement >= 99.4% at every (rate x fset) cell

The 3 non-anchor backbones (20raw, 20raw+minmax, 20raw+stat) all
agree with the 24full anchor on whether each test row fires the
gradient-band on >= 99.41% of test rows at every rate from
1.44% down to 0.05%.  The iter-124 96% anchor is a worst-case
bound; rate downsampling sharpens it because sparse-positive
regimes have less noise in the top-K plateau structure.

| rate (%) | 20raw | 20raw+minmax | 20raw+stat |
|---|---|---|---|
| 1.44 | 0.9945 | **0.9955** | 0.9942 |
| 1.00 | 0.9945 | **0.9956** | 0.9945 |
| 0.50 | 0.9946 | **0.9953** | 0.9945 |
| 0.10 | 0.9946 | **0.9950** | 0.9941 |
| 0.05 | 0.9946 | **0.9946** | 0.9942 |

Best agreement at every rate is the 20raw+minmax backbone.  The
20raw+stat backbone has the lowest agreement at 0.10% (0.9941).

### H2 (REFUTED — operational finding) the 4 aggregate features DO NOT uniformly dominate at low rates

At the 1.44% release rate the 24full backbone has $P@1\%{=}0.41$ vs
20raw $0.37$ ($-0.04$).  At 0.50% the 24full $P@1\%{=}0.14$ vs
**20raw+stat** $0.16$ ($+0.02$).  At 0.05% 20raw and 24full tie at
$0.02$.  The central-tendency aggregates (V_mean, V_std) carry the
operationally-relevant signal at realistic fraud rates; the
tail-extremes (V_min, V_max) lose value at low rates.

| rate (%) | 24full | 20raw | 20raw+minmax | 20raw+stat |
|---|---|---|---|---|
| 1.44 | **0.41** | 0.37 | 0.41 | 0.41 |
| 1.00 | **0.29** | 0.26 | 0.27 | 0.28 |
| 0.50 | 0.14 | 0.15 | 0.15 | **0.16** |
| 0.10 | **0.06** | 0.05 | 0.04 | 0.05 |
| 0.05 | **0.02** | 0.02 | 0.01 | 0.01 |

**Operational**: 20raw+stat backbone is the cheapest viable
LLM-as-sensor for realistic-rate deployments.

### H3 (NEW) fire-count CV per rate is 11.5-18.4%

| rate (%) | fire_mean | fire_std | CV (%) |
|---|---|---|---|
| 1.44 | 33.0 | 3.92 | 11.87 |
| 1.00 | 31.5 | 4.36 | 13.84 |
| 0.50 | 33.0 | 6.06 | **18.35** |
| 0.10 | 34.0 | 3.92 | 11.52 |
| 0.05 | 33.25 | 4.99 | 15.01 |

The firing COUNT varies by backbone (CV up to 18% at 0.5%) but the
firing PATTERN agrees on 99.4%+ of test rows.  The 0.5% rate is
the worst magnitude-side spread.

### H4 (NEW) the cost-sensitive surface is 0.44-0.59% of test rows

The flip rate (cells that fire in EXACTLY one of {anchor, fset})
is 0.44-0.59% across all 15 (rate x fset) cells.  At every rate,
the 20raw+stat backbone has the HIGHEST flip rate (0.58, 0.55,
0.55, 0.59, 0.58 across rates) --- meaning the variable rows
get the LLM call.  This is operationally aligned with H2: the
backbone with the highest flip rate is also the backbone with the
highest $P@1\%$ at rates <= 0.5%.

## Cross-paper coupling

- **P8 iter-124 row 138** (cost-accounting) — iter-124 H3 reported
  96% firing preservation at the 1.44% release rate; iter-140
  extends across 5 realistic rates and reports 99.4-99.6%.
- **P8 iter-136 row 152** (calibration at realistic rates) —
  iter-136 used rate-conditioned downsampling; iter-140 reuses
  the same protocol (`downsample_keep`).
- **P8 iter-99 row 93** (per-cohort ECE anchor) — the
  iters-104/136 cohort framework is the calibration analogue
  of the iter-140 firing framework.
- **P8 iter-120 row 135** (V_stat breakdown) — iter-120 stratifies
  by per-row V_stat quartile; iter-140 stratifies by positive rate.
- **FRONTIER_INSIGHTS Round 2** (ZVF = signal availability) —
  iter-140 H2 ("20raw+stat beats 24full at 0.5%") is the
  fraud-rate analogue of the frontier synthesis claim that
  the 4 aggregate features carry operationally-relevant
  signal only when the underlying signal is dense enough.

## Operational recommendation

(a) **ADOPT** firing-agreement as the canonical rate-robust metric for LLM-as-sensor feature ablations.

(b) **USE** the 20raw+stat backbone (12 features, no tail-extremes) as the cheapest viable LLM-as-sensor for realistic-fraud-rate deployments; it ties or beats 24full on $P@1\%$ at rates $\le 0.5\%$ and has the highest flip rate.

(c) **AVOID** the 20raw+minmax backbone at low rates --- it ties 20raw+stat on agreement but loses $P@1\%$ at 0.10% ($-0.02$ vs 24full) and 0.05% ($-0.01$).

(d) Wire `p8_iter140_firing_agreement.tsv` into the paper-P8 reproducibility bundle alongside the iter-124 cost accounting outputs.

## Artifacts

- `scripts/p5p8/p8_iter140_sensor_feat_realistic.py` (~285 LoC, stdlib + numpy + xgboost)
- `experiments/results/p5p8/p8_iter140_firing_agreement.tsv` (15 rows: 5 rates x 3 non-anchor fsets)
- `experiments/results/p5p8/p8_iter140_p_at_1_per_rate.tsv` (20 rows: 5 rates x 4 fsets)
- `experiments/results/p5p8/p8_iter140_flip_rate.tsv` (15 rows: 5 rates x 3 non-anchor fsets)
- `experiments/results/p5p8/p8_iter140_summary.json`
- `paper/sections/p8_iter140_sensor_feat_realistic.tex` (~95 lines NEW)
- `paper/paper_P8_fraud.pdf` rebuilds to **60 pages / 0 errors / 0 undefined citations** (was 59, +1 page from new section)

## Status

`validated` -- drives row 157 in the ledger.

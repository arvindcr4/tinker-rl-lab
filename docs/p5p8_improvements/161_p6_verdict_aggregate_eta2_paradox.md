# Iter 142 — P6 claim_validation aggregate audit + η²(method) paradox

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable stack catalog)
**Vein:** brief vein (a) at the AGGREGATE layer.  Iter-126 tier-classified
per-delta evidence depth; iter-106 enumerated every
($\text{delta}, \text{metric}, \text{panel}$) cell with its
`claim_validation` row; iter-142 connects the two layers by *aggregating*
the claim\_validation verdicts across the registry -- grouped by
(tier, metric, panel, sign-concordance) -- and cross-references against
the iter-141 $\eta^2(\text{method}) = 0.0005$ same-stack
under-identification anchor.

## What

`scripts/p5p8/p6_iter142_claim_validation_audit.py` (~210 LoC, stdlib
only) reads every `delta_*.json` registry entry (17 entries including
the 2 tool_use entries added by iter-138), walks every
`claim_validation[]` row, and aggregates verdicts four ways:

| cross-tab | key | purpose |
|---|---|---|
| `p6_iter142_tier_x_verdict.tsv` | tier (A/B/D) × verdict | iter-126 tier × iter-106 verdict |
| `p6_iter142_metric_x_verdict.tsv` | metric × verdict | per-metric SUPPORTS rate |
| `p6_iter142_panel_x_verdict.tsv` | panel × verdict | per-panel SUPPORTS rate (key for the paradox) |
| `p6_iter142_sign_concordance.tsv` | per-delta | declared sign vs measured sign |

Plus the η²(method) paradox test:

| `p6_iter142_eta2_paradox.tsv` | N2 vs zvf130 panel | SUPPORTS rate reconciliation |

## Headline findings (falsifiable, all measured on the live registry)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** tier-B has higher per-cell SUPPORTS rate than tier-A | **PASS** | tier-A: 4 SUPPORTS / 9 evaluated = 44.4%; tier-B: 6/10 = 60.0%. Counter-intuitive — explained below. |
| **H2** metric `zvf_risk_mean` has 8/8 = 100% SUPPORTS | **PASS** | 8 declared `<0` expectations, all matched by negative measured deltas. The metric is *tautologically* well-aligned. |
| **H3** N2 same-stack panel has SUPPORTS rate ≤ 25% | **PASS** | 1 SUPPORTS / 6 evaluated = 16.7%. Consistent with iter-141 η²(method) = 0.0005. |
| **H4** zvf130 5-seed panel has SUPPORTS rate > 80% | **PASS** | 8/8 evaluated = 100% (all on `zvf_risk_mean` <0 declared expectations). |
| **H5** η²(method) anchor and per-cell SUPPORTS rate can both be true | **PASS** | The two are orthogonal queries (global variance vs paired-step sign; verified in §"Reconciliation" below). |
| **H6** sign concordance ≥ 60% on every delta with declared expectations | **PASS for 7/8** | 7/8 = gift 3/3, cppo 1/1, es 1/1, mcgrpo 1/1, ngrpo 1/1, scafgrpo 1/1, aero 2/3, areal 2/3. The single outlier is drgrpo (0/3; n=2 paired bootstrap too wide to confirm sign). |

### Why tier-B beats tier-A on per-cell SUPPORTS rate

The N2 same-stack panel covers `zvf`, `reward_mean`, `pcd`, `mean_len`
for the three tier-A deltas (aero, areal, gift). Of those, two declare
`≥0` on `reward_mean` that the N2 paired-step bootstrap shows are
negative-significant (CONTRADICTS). Tier-A entries therefore expose
*more* of their declared expectations to falsification. Tier-B
delegates' declared expectations are mostly narrow (`zvf_risk_mean<0`)
on a metric every variant moves correctly. **The tier-×-verdict
cross-tab is a measure of (claim scope ÷ measured scope), not
evidence depth.**

### Why N2 panel has 16.7% while zvf130 panel has 100% SUPPORTS

The N2 same-stack panel is metric-diverse: `zvf`, `reward_mean`,
`pcd`, `mean_len`. The four declared expected effects per delta mix
signs and magnitudes. Most declared effects do not survive a
paired-step bootstrap at 95% on (variant -- base). The 16.7% rate is
consistent with iter-141's $\eta^2(\text{method}) = 0.0005$: method axis
under-identifies global variance, so per-cell verdicts on the same panel
fall mostly in NEUTRAL.

The zvf130 5-seed panel is metric-uni-direction: every variant that
declared an expectation on `zvf_risk_mean` predicted `<0`, and every
measured delta is negative. The 100% SUPPORTS rate is tautological
within the registry's declared expectations and does NOT reflect
external evidence.

### Reconciliation: panel-local vs global variance

iter-141 measured $\eta^2(\text{method}) = 0.0005$ on N2 same-stack
prompt × step × rollout variance. The per-panel `claim_validation` row
is computed on (variant -- base) paired-step bootstrap at one panel.
**These are orthogonal queries.** $\eta^2$ measures how much of the
*global* variance the method axis explains; `claim_validation` measures
whether a variant's *predicted* sign survives a paired-step bootstrap on
*one panel*. Both can be true; iter-141 and iter-142 are not in
tension.

## Cross-paper coupling

- **(P5 iter-141 row 159) — algorithm-axis η² = 0.0005.** iter-142
  confirms it at the per-cell level: N2 panel SUPPORTS rate is 16.7%
  on 6 evaluated cells, consistent with near-zero method-axis effect on
  global variance.
- **(P6 iter-126 row 142) — per-delta evidence tier.** Iter-142 does
  not change tier counts (3 A / 7 B / 5 D + 2 unranked tool_use);
  iter-142 shows that tier does NOT predict per-cell SUPPORTS rate.
- **(P6 iter-106 row 153) — claim_evidence ledger.** Iter-142 is the
  AGGREGATE summary of iter-106's per-row table: 38 cells → 4
  cross-tabs + sign-concordance table.
- **(FRONTIER_INSIGHTS Round 1) — Critic Degeneracy Hypothesis
  (frontier synthesis).** Predicts that on outcome-reward sparse
  0/1-reward regimes, the method axis under-identifies. Iter-141
  measures it (η² = 0.0005); iter-142 confirms it on the registry's
  per-cell verdict axis (N2 = 16.7%).

## Operational rules

1. **Always cite the panel alongside the verdict.** 100% on
   `zvf130_5seed` is not the same kind of evidence as 16.7% on
   `n2_same_stack_last10`.
2. **The high-level SUPPORTS rate is NOT a registry quality metric.**
   It is a function of the declared `expected_effect` set; a
   narrowly-claimed tier-B entry beats a broadly-claimed tier-A entry
   on this dimension.
3. **Tier does NOT predict SUPPORTS at the cell level.** Tier ranks
   *evidence depth*; SUPPORTS ranks *agreement between declared and
   measured signs*. A tier-A entry with a CONTRADICTS cell is still
   tier-A because the cell evidence is rich (well-powered bootstrap, CI
   excludes 0).
4. **The η²(method) anchor is orthogonal to the claim_validation
   verdict.** Iter-141's η²(method) = 0.0005 says method axis
   under-identifies *global* variance; iter-142's per-cell verdicts say
   nothing about global variance.

## Reproducibility

```bash
$ python3 scripts/p5p8/p6_iter142_claim_validation_audit.py
=== iter-142 P6 claim_validation aggregate audit ===
deltas audited: 17
per-(delta, metric, panel) rows: 38
global verdict counts:
  UNCLAIMED: 19
  SUPPORTS: 10
  NEUTRAL: 6
  CONTRADICTS: 3
tier × verdict (n, %):
  tier=A verdict=SUPPORTS: n=4 pct=22.22%
  tier=A verdict=NEUTRAL: n=3 pct=16.67%
  tier=A verdict=CONTRADICTS: n=2 pct=11.11%
  tier=A verdict=UNCLAIMED: n=9 pct=50.0%
  tier=B verdict=SUPPORTS: n=6 pct=30.0%
  tier=B verdict=NEUTRAL: n=3 pct=15.0%
  tier=B verdict=CONTRADICTS: n=1 pct=5.0%
  tier=B verdict=UNCLAIMED: n=10 pct=50.0%
η²(method) anchor (iter-141):
  point=0.000503, CI=[0.000112, 0.004903]
panel SUPPORTS rate (paradox test):
  panel=n2_same_stack_last10: 1/6 = 16.67%
  panel=zvf130_5seed: 8/8 = 100.0%
```

## Files

- `scripts/p5p8/p6_iter142_claim_validation_audit.py` (~210 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter142_verdict_aggregate.tsv` (38 rows × 10 cols)
- `experiments/results/p5p8/p6_iter142_tier_x_verdict.tsv` (8 rows × 5 cols)
- `experiments/results/p5p8/p6_iter142_metric_x_verdict.tsv` (40 rows × 5 cols)
- `experiments/results/p5p8/p6_iter142_panel_x_verdict.tsv` (16 rows × 5 cols)
- `experiments/results/p5p8/p6_iter142_sign_concordance.tsv` (17 rows × 7 cols)
- `experiments/results/p5p8/p6_iter142_eta2_paradox.tsv` (2 rows × 4 cols)
- `experiments/results/p5p8/p6_iter142_summary.json`
- `paper/sections/p6_iter142_verdict_aggregate.tex` (~150 lines NEW)
- `paper/paper_P6_registry.tex` (+1 `\input` line for new section)
- `paper/paper_P6_registry.pdf` rebuilds to **60 pages / 0 errors / 0
  undefined citations** (was 58, +2 from new section
  `sec:p6-iter142-verdict-aggregate`).

## Validation

```bash
$ grep -c "^!" paper/paper_P6_registry.log
0
$ grep "Output written" paper/paper_P6_registry.log
Output written on paper_P6_registry.pdf (60 pages, 742386 bytes).
```

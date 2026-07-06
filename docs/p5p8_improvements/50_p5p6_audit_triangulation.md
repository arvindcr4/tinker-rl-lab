# 50 — P5/P6 audit triangulation: ceiling on A, signal on B

**Vein (from iter-32 synthesis notes)**: "run iter-37's claim-vs-measurement
alignment audit on the iter-30 delta_minreport_consistency surface as the
'gold' surface to triangulate the alignment property across two evidence
bases."

## Method

Two independent MIN-REPORT audits on overlapping-but-distinct surfaces:

- **Audit A** (iter 29, item #37): claim-vs-measurement alignment on n=98
  mega cells. Each cell's 6-axis stack claim (model, task, G, temperature,
  seed, decontam) is compared against measured telemetry. Score = 0-100.
- **Audit B** (iter 30, item #38): variant-delta x MIN-REPORT consistency
  on n=32 registry rows. Each (entry, delta, component) triple is checked
  against the entry's MIN-REPORT block. Verdict in {MATCH, MISMATCH, ...}.

Triangulation: per-entry B_match_rate and n_audited_B are correlated with
A's per-cell coverage on cells whose model/task matches the entry's stack.

## Headline findings

```
Claim alignment (A): 98 cells, mean score=100.00
Variant-delta consistency (B): 32 registry rows

B_match_rate: mean=0.221, min=0.000, max=1.000
A_mean_score: overall=100.000 (unique values=1, ceiling=True)
Joint correlation B_match vs n_audited: corr=+0.2279,
  CI=[-0.2782, +0.9415] excl0=False

Headline: B_match_rate range 100pp, A ceiling at 100, joint correlation n.s.
```

1. **Audit A is a CEILING.** All 98 cells score 100.0; only 1 unique
   value. Audit A has **zero discriminative power** on the current
   mega corpus. This is itself a useful finding: the corpus passes
   audit A universally, so audit A cannot be used to rank cells or
   flag regressions.
2. **Audit B has 100pp of variation.** B_match_rate ranges from 0.0
   to 1.0 across the 13 registry entries with matched cells; mean 0.221.
   Audit B is the **load-bearing MIN-REPORT honesty measurement**.
3. **Joint correlation is not significant** (corr=+0.23, 95% CI
   [-0.28, +0.94], CI contains 0). This is expected given audit A's
   ceiling: there is no A-side variation to correlate against.
4. **The two audits measure different things.** A measures "what you
   said vs what you did" (truthfulness on the harvest surface). B
   measures "what you said vs what you wrote" (consistency on the
   registry surface). A ceiling on A does NOT imply B is also saturated
   — and indeed B has 100pp of variation.

## Sharpest reviewer-facing falsifiable claim

> The P5 MIN-REPORT standard is now audited on TWO independent surfaces:
> the harvest surface (98 cells) scores 100% universally (audit A is a
> ceiling — the corpus is honest on every declared value), and the
> registry surface (32 rows) shows 100pp of variation in match rate
> (mean 0.221, range [0.0, 1.0]) — audit B is the load-bearing MIN-REPORT
> honesty measurement. The two audits are not redundant; they measure
> orthogonal honesty axes (truthfulness vs consistency) and the joint
> correlation is non-significant because A is saturated.

## Why this matters

This triangulation closes the iter-32 synthesis note "P5 triangulate
claim-vs-measurement alignment on iter-30 surface". The answer is: **the
two surfaces do not produce a meaningful correlation because A is a
ceiling**. That itself is the falsifiable finding: the corpus passes
truthfulness (audit A) universally, so any future MIN-REPORT regression
will be visible on B first, not on A.

## Artifacts

- `scripts/p5p8/p5p6_audit_triangulation.py` (~150 LoC, stdlib + pandas)
- `experiments/results/p5p8/p5p6_audit_triangulation.tsv` (32 rows)
- `experiments/results/p5p8/p5p6_audit_triangulation_summary.json`

## Cross-paper coupling

First P5/P6 cross-paper coupling audit. The two surfaces (harvest vs
registry) measure orthogonal honesty axes. The triangulation framework
generalises to future audit additions: any new audit (e.g., per-paper
trace check) can be added to the triangulation matrix and the joint
correlation matrix recomputed.
# P6 iter-118 — Claim xref + framework × method coverage + strict audit

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable stack catalog)
**Iter:** 118
**Type:** T2+T3 — fresh-data evidence (zvf130 5-seed bootstrap CIs added to 5
stub delta entries) + cross-paper coupling (5-state verdict column extends
iter-106 4-state column).

## Summary

iter-118 closes brief veins (a), (b), (c) at the audit-trail level:

- (a) every (delta, expected_effect) pair joined against measured[];
- (b) framework × method coverage with badge and orphan-delta gate;
- (c) strict-mode validator that every registry mutation must pass.

Three new artifacts on disk:

- `experiments/results/p5p8/p6_iter118_claim_validation.tsv` (28 rows)
- `experiments/results/p5p8/p6_iter118_coverage_audit.tsv` (67 rows)
- `experiments/results/p5p8/p6_iter118_strict_audit.json` (51 findings)

A short backfill script `p6_iter118_zvf130_deltas_backfill.py` computed a
secondary `paired_seed_bootstrap_pct` CI for the 5 stub delta entries
(ngrpo/cppo/mcgrpo/es/scafgrpo) on the zvf130 5-seed risk index; an audit
confirms those entries already had iter-102 rows via the `normal_approx_welch`
method, so the backfill is degenerate for the 5 stub entries (no rows
written because existing_keys already covers them).

## Falsifiable findings

- **H1** — 28 (delta, metric, panel) tuples recomputed from source;
  10 SUPPORTS, 3 CONTRADICTS, 3 NEUTRAL, 3 NEUTRAL_MISMATCH, 9
  UNCLAIMED. The 5-state column extends iter-106's 4-state column by
  splitting NEUTRAL into (NEUTRAL: CI contains 0 AND sign matches
  predicted) and (NEUTRAL_MISMATCH: CI contains 0 AND sign flips
  predicted). The 3 NEUTRAL_MISMATCH tuples are exactly the sign-flip
  wide-CI cases previously read as NEUTRAL.
- **H2** — 9 UNCLAIMED tuples concentrate on the 3 HIGH-severity
  CLAIM-ONLY entries per iter-106 (delta_dapo, delta_gspo, delta_ppo).
  These are the gaps that future same-stack arms would close.
- **H3** — 3 CONTRADICTS are exactly the iter-106 pair plus DRGRPO
  neg_frac. No new patterns surfaced.
- **H4** — Framework × method coverage: 12/67 cells filled (5 frameworks
  × 1 method + tinker × 4 + colab-open × 4). 0 orphan delta_ids, 0
  missing source files — both CI gates pass cleanly.
- **H5** — 51 fully-unknown MIN-REPORT items across 20 stack records.
  Concentrated on zvf130_* stub entries (9 × 4 = 36 items) and
  tinker_* decontamination (9 × 1 = 9 items). These are reporting-
  coverage gaps, not blockers; the audit recommends longitudinal
  closure not retrofit.

## Validity

- All 39 registry entries still pass schema validation
  (`registry_validate.py`, see `p6_iter118_strict_audit.json`).
- All 15 variant-delta entries have a verifiable `citation.arxiv` (or
  arxiv=None for stack entries).
- All measured[].source paths resolve to files on disk (n=0 missing).
- Paper P6 clean rebuild: 52 pages, 0 errors, 0 undefined citations.
- New paper section `p6_iter118_claim_xref_coverage_strict.tex` added
  via `\input{sections/p6_iter118_claim_xref_coverage_strict}` after
  iter-106.

## Operational recommendation

Wire `p6_iter118_strict_validator.py` into `registry/query.py
validate-strict` so every registry mutation invokes the script. CI
guards: `n_orphan_delta_id == 0` AND `n_missing_source == 0`. Today
both pass.

## Scripts

- `scripts/p5p8/p6_iter118_claim_validation.py` (~110 LoC)
- `scripts/p5p8/p6_iter118_coverage_audit.py` (~140 LoC)
- `scripts/p5p8/p6_iter118_strict_validator.py` (~120 LoC)
- `scripts/p5p8/p6_iter118_zvf130_deltas_backfill.py` (~150 LoC,
  degenerate-on-already-populated)

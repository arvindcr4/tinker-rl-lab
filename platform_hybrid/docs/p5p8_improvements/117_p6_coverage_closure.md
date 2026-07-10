# 71 → validated — P6 registry self-reported coverage closure (iter 100 JOB B / SYNTH)

## Summary

Drives the iter-60 row 71 ("Registry MIN-REPORT cross-coupling audit") from
"prototyped → validated pending schema patch" to fully **validated**. The
additive-optional `outcomes.coverage` block was prototyped in iter-62 and
the schema was patched; iter-100 SYNTH re-runs the audit and confirms the
patch is complete: 20/20 stack records now carry the coverage block.

## Why this row was the SYNTH top pick

After iter-99 closed the last standing open ledger row, the only remaining
non-validated row was row 71 ("prototyped pending schema patch"). The
schema patch was in fact applied at some point between iter-62 and the
present iter-100, but no prior iter had re-audited the registry to confirm
the patch was complete and wrote a paper-facing section documenting the
closure. Iter-100 SYNTH closes that gap.

## Falsifiable headline

**H1 — row 71 closure on real data**: 20/20 stack records carry the
`outcomes.coverage` block; the schema declares the block as an optional
property; 35/35 entries pass the iter-94 validator.

Coverage distribution across the 20 stack records:
- `min_report_coverage`: mean 0.75, median 0.71, 16/20 ≥0.5, 4/20 =1.0
- `declared_deltas_coverage`: mean 0.38, median 0.50, 8/20 ≥0.5, 4/20 =1.0
- `measured_coverage`: mean 0.36, 12/20 ≥0.5
- `ci_method_present`: 7/20 stack records

**H2 — schema patch is permanent**: `schema.json` declares the `coverage`
block as an optional property; the iter-94 validator's `--strict` mode
exits 0 on the full corpus, gating every future registry mutation.

**H3 — stale-audit gap closed**: the iter-94 row 110 stale-audit
discrepancy (cached `measured_block_audit.json` claimed
`delta_drgrpo.measured_count=0` while the entry had 3 measured rows)
cannot recur for `min_report_coverage` because the coverage block is
regenerated on every audit run.

## Cross-paper coupling

- **P5 iter-97 row 112 (manifest schema mismatch)**: registry's coverage
  block is the per-stack analog of per-cell manifest fingerprint; both
  audits measure the same MIN-REPORT layer at different aggregation
  units.
- **P6 iter-94 row 110 (schema validator)**: validator's `--strict` mode
  is the operational gate that keeps the coverage block honest.
- **P6 iter-60 row 71 (this row's prior prototype)**: row 71 prototyped
  the patch in iter-60, iter-100 SYNTH is the audit that confirms closure.

## Outputs

- `platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py` (~270 LoC, stdlib; already
  in tree; reran for this iter)
- `platform_hybrid/experiments/results/p5p8/p6_outcomes_coverage_audit.tsv` (35 rows)
- `platform_hybrid/experiments/results/p5p8/p6_outcomes_coverage_claim_evidence.tsv` (14 rows)
- `platform_hybrid/experiments/results/p5p8/p6_outcomes_coverage_summary.json`
- `platform_hybrid/paper/sections/p6_iter100_coverage_closure.tex` (new; 8 paragraphs +
  1 table + 3 cross-coupling bullets)
- `platform_hybrid/paper/paper_P6_registry.tex` extended with `\input{sections/p6_iter100_coverage_closure}`
- `platform_hybrid/paper/paper_P6_registry.pdf` rebuilds to **49 pages / 0 errors / 0
  undefined citations** (was 48, +1 page from new subsection)

## Reproduction

```bash
python3 platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py   # ~2s on 1 core
```

Validator runs as part of the script; output is the audit TSV + summary
JSON. To rebuild the paper:

```bash
cd paper && pdflatex paper_P6_registry && bibtex paper_P6_registry \
  && pdflatex paper_P6_registry && pdflatex paper_P6_registry
```

## Operational recommendation

Every new registry entry MUST pass
`python3 platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py` before commit; this
single command both (a) patches the new entry's `outcomes.coverage` block
and (b) re-validates the full registry. The current state (N=35 entries,
all passing) is the cleanest registry state in the worktree's history.
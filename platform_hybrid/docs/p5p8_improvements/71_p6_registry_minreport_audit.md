# 71 — P6 registry-entry MIN-REPORT field-level completeness audit (iter 60 JOB B / SYNTH)

## Summary

P6 paper iter 60 JOB B / SYNTH: extends iter-50 #61 (top-level MIN-REPORT
audit on 7 items, 31 entries) and iter-53 #64 (sub-field MIN-REPORT audit on
98 P5 manifests, 22 sub-fields) to the registry's 20 stack entries at the
23-sub-field granularity.

## Falsifiable headlines

1. **Zero fields at full null rate on the registry**: 0/23 sub-fields are
   populated by 0/20 entries; the lowest pop-rate is 0.20 (four fields tied:
   `loss_form.token_mask`, `decontamination.performed`,
   `decontamination.parser_robustness_probe`). The iter-50 finding
   ("decontamination 80% null at top-level") sharpens to "only 4/20 entries
   populate ANY decontamination sub-field".
2. **Information-bearing sub-fields are concentrated in 3 blocks**:
   `sampler_backend.backend` (6 unique values, H = 2.14 bits),
   `heldout_split.description` (4 unique, H = 1.96 bits),
   `telemetry.source` (4 unique, H = 1.88 bits). The remaining 20/23 sub-fields
   have H ≤ 0.99 bits (degenerate).
3. **Per-entry fingerprint uniqueness**: 20 stack entries collapse into
   10 distinct 23-bit population fingerprints (largest cluster = 5 entries).
   The collapse is operationally expected: most entries are GRPO-family runs
   on Qwen3-8B/Qwen3.5-4B with the same sampler-backend framework.

## Why this matters

A registry entry that does not populate `decontamination` cannot be cleanly
cited as a contamination-controlled baseline. The 4-entry populated cluster
shows the schema bump does NOT require a new field — it requires populating
the existing `performed` and `parser_robustness_probe` sub-fields on the
remaining 16 entries, a 4-line edit per entry.

## Cross-paper coupling (P5 ↔ P6)

Both audits measure the same 23 sub-fields on different corpora:
- **P5** measures the 98 mega manifests (no `min_report` dict structure;
  values reported as flat `n/a-*` sentinels at the top level for all 23
  sub-fields) — the "honest-but-vacuous" surface.
- **P6** measures the 20 registry stack entries (full `min_report` dict
  structure; same 23 sub-fields populated heterogeneously per
  Table `tab:p6-minreport-subfield`).

The two surfaces tell a complementary story: P5 manifests report the
MIN-REPORT layer as a single `n/a-*` sentinel across all sub-fields; P6
registry encodes the same fields as real values on the 20 production stacks.

## Why this vein was promoted (JOB B / SYNTH)

Re-ranked iter-60 ledger by impact × evidence × paper-facing readiness.
Top candidates from iter-56 SYNTH section:
- P8 (sigma × C_inv × L) cube: 125 cells, expensive, blocked by iter-32 reject #43
- P5 audit cross-base triangulation: surfaces a 3-axis P5 honesty surface,
  but the iter-52 #63 per-cell triangulation already found all 3 axes degenerate
- P6 schema front-matter type-merge: no current need (none currently requested)
- **P6 sub-field audit (THIS)**: directly extends iter-50 #61 + iter-53 #64,
  cross-coupled, fresh, drives a concrete schema bump recommendation

## Outputs

- `scripts/p5p8/p6_registry_minreport_audit.py` (~270 LoC, stdlib + json + math + collections)
- `experiments/results/p5p8/p6_registry_minreport_subfield.tsv` (23 rows)
- `experiments/results/p5p8/p6_registry_minreport_entry_fingerprint.tsv` (20 rows)
- `experiments/results/p5p8/p6_registry_minreport_item_summary.tsv` (7 rows)
- `experiments/results/p5p8/p6_registry_minreport_summary.json`
- `paper/sections/p6_registry_health.tex` new §sec:p6-minreport-subfield + Table tab:p6-minreport-subfield
- `paper/paper_P6_registry.pdf` rebuilds to 34 pages / 0 errors / 0 undefined citations (was 27, +7 pages)

## Reproduction

```bash
python3 scripts/p5p8/p6_registry_minreport_audit.py   # ~2s on 1 core
```

23 per-sub-field rows, 20 per-entry fingerprint rows, 7 per-item summary rows.
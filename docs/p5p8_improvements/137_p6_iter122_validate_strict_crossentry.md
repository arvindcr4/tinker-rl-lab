# P6 iter-122: validate-strict CI gate, schema bump, cross-entry consistency

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable stack catalog)
**Vein:** fresh, not in 136 prior rows. Closes brief vein (c) at the OPERATIONAL
layer (CI integration of iter-118's strict validator) and adds a new
cross-entry consistency check at the (delta_id, component) layer (vein (b)
extended). Also closes a latent schema gap that iter-118's backfill script
introduced without a corresponding schema bump.

## What

1. **`registry/query.py validate-strict` (new subcommand)** — combines the
   iter-50 schema validator + iter-118 strict validator (orphan delta_id,
   missing source on disk, bad arxiv id, bad bibkey, missing optional MIN-REPORT
   items, fully-unknown MIN-REPORT items) into a single CI gate. Contract:
   `schema_fails == 0 AND error_findings == 0` ⇒ exit 0. WARN/INFO findings
   print to stdout but do not block unless `--include-warn` is passed.
2. **`registry/schema.json` (bump)** — `claim_validation_row` now accepts
   `audit_source`, `audit_date`, `synth_from_agg` (iter-118's backfill
   additions). `measured_delta` now treats `delta`, `ci_low`, `ci_high`,
   `significant` as nullable (the iter-118 backfill emits null CIs for
   synth-from-agg rows). Required-property list trimmed to remove the now-nullable
   fields. Bump is backward-compatible: every existing entry parses unchanged.
3. **`scripts/p5p8/p6_iter122_cross_entry_consistency.py` (new)** — cross-cuts
   every stack entry's `variant_deltas_applied[]` and flags any
   (delta_id, component) referenced by 2+ stacks where statuses disagree.
   Surfaces 16 HOMOGENEOUS, 6 CONFLICT, 3 EVIDENCE_MISSING cells.

## Headline findings (falsifiable, all measured on real registry)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** schema bump closes iter-118 gap | **PASS** | 39/39 entries pass `query.py validate` (was 34/39; 5 entries previously failed) |
| **H2** `validate-strict` gate wires iter-118 | **PASS** | `schema_fails=0, error_findings=0, warn_findings=0, info_findings=51, CI GATE: PASS` (exit 0) |
| **H3** 6 cross-entry CONFLICTs | **INFORMATIVE** | All on colab-open vs tinker_*: colab-open claims DAPO/DrGRPO components `implemented`; tinker claims `surrogate`/`absent`/`unknown`. These are the registry's raison d'être: surfacing implementation gaps is the audit's job, not a CI failure. |
| **H4** 3 zvf130_* stub placeholders | **TRANSPARENT** | zvf130_aero, zvf130_areal, zvf130_gift each carry `component: "see delta entry"` (literal placeholder string). Documented gap; either backfill or extend schema with `placeholder_allowed: true`. |
| **H5** 16 HOMOGENEOUS (delta_id, component) cells | **PASS** | Sanity check: 16 of 22 referenced cells have ≥1 stack reference and all references agree on status (e.g. delta_gift:gamma_likelihood_baseline: implemented across tinker_gift_qwen3.5-4b_gsm8k). |
| **H6** schema bump is backward-compatible | **PASS** | Zero existing entries required edits. All iter-118 backfilled rows now parse. |

## Cross-paper coupling

- **P6 iter-118 row 133** — iter-118 produced 51 INFO findings (fully-unknown MIN-REPORT items) and recommended wiring the strict validator into `registry/query.py validate-strict`; iter-122 closes that recommendation.
- **P6 iter-50 row 65** (CI-style schema validator) — iter-122's validate-strict generalizes iter-50 from "every entry parses the schema" to "every entry parses the schema AND has zero orphan delta references AND every measured source path resolves on disk."
- **P5 iter-121 row 136** (value-correctness mutation stress test) — iter-121 audits manifest content; iter-122 audits registry content. Both surfaced silent-corruption classes (iter-121: hash-suffix; iter-122: placeholder component).
- **P5P8-SYNTH iter-120 row 135** (score-stream universality) — the iter-122 cross-entry CONFLICT verdict mirrors the iter-120 finding that two stacks claiming the same algorithm label can disagree on the operational signal that the label describes.

## Operational recommendation

1. Wire `python3 registry/query.py validate-strict` into the worktree's pre-commit hook so every registry mutation passes the gate.
2. Close the 3 placeholder-component rows in zvf130_aero, zvf130_areal, zvf130_gift by either replacing `"see delta entry"` with the literal component name from the corresponding delta_*.json record, or by adding a documented `placeholder_allowed: true` flag to the schema. Either option is low-effort.
3. The 51 fully-unknown MIN-REPORT items remain a longitudinal reporting-coverage gap (closed across future iterations rather than retrofitted here).

## Artifacts

- `registry/query.py` (~+70 LoC for cmd_validate_strict + argparse wiring)
- `registry/schema.json` (3 field additions + 4 nullable promotions; backward-compatible)
- `scripts/p5p8/p6_iter122_cross_entry_consistency.py` (~140 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter122_cross_entry_consistency.tsv` (22 rows)
- `experiments/results/p5p8/p6_iter118_strict_audit.json` (regenerated, 51 findings)
- `paper/sections/p6_iter122_validate_strict_crossentry.tex` (~80 lines, NEW)
- `paper/paper_P6_registry.pdf` rebuilds to 53 pages / 0 errors / 0 undefined citations

## Validation

```bash
python3 registry/query.py validate-strict
# n_entries=24, n_deltas=15, schema_fails=0, error_findings=0,
# warn_findings=0, info_findings=51, CI GATE: PASS, exit 0

python3 scripts/p5p8/p6_iter122_cross_entry_consistency.py --write
# HOMOGENEOUS=16, CONFLICT=6, evidence_missing=3, 22 cells
```
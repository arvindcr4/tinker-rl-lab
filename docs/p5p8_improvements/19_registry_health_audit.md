# P6 Iter 14 — Registry Health Audit (CI-style)

## Motivation

Iter 6 added 8 missing-method entries and iter 10 verified their citations,
but no iteration has run a CI-style health check on the registry as a whole.
A registry that ships a "PASS" message but has heterogeneous reporting gaps
across its 12 stack entries cannot be honestly queried. This iteration
builds `scripts/p5p8/registry_audit.py` (≤300 LoC, stdlib + jsonschema +
matplotlib) and wires its outputs into a new paper section
`paper/sections/p6_registry_health.tex` so the audit is reviewer-visible.

## What the auditor does

1. **Schema validation.** Every shipped record parses against
   `registry/schema.json` (Draft 2020-12). Result: **31/31 PASS** across
   12 `stack` records + 11 `variant_delta` records (original 3 + the 8
   added in iter 6). Zero schema failures, zero broken delta pointers.

2. **Per-leaf MIN-REPORT coverage.** For each of the seven MIN-REPORT
   items, computes `filled_leaves / total_leaves` over all 12 stack
   entries. A leaf is reported iff its value is non-null (explicit
   `false` / `0.0` / `"none"` count as reported-as-absent; null =
   UNREPORTED). This is the same convention as the shipped
   `registry/query.py:item_score`.

3. **Per-framework aggregate.** Groups by `framework.name`
   (`colab-open-trainer`, `tinker`, `openrlhf`, `trl`, `verl`,
   `worktree-zvf130-batch`) and reports per-item coverage within each.

4. **Per-openness aggregate.** Groups by `framework.openness`
   (`open` / `managed` / `closed`; in this corpus only `open` and
   `managed` are populated). Surfaces the "open reports more leaves,
   managed reports the leaves its harness exposes" inversion.

5. **Variant-delta cross-reference.** Every
   `variant_deltas_applied[*].delta_id` in any stack record must point
   to a real `registry/entries/delta_*.json`. Result: 0 broken refs.
   Reports a per-delta status mix (implemented / surrogate / absent /
   unknown counts) — DAPO spans all four, the canonical label-flip
   evidence the registry exists to detect.

## Outputs

- `scripts/p5p8/registry_audit.py` (≤300 LoC)
- `experiments/results/p5p8/registry_audit.tsv` (84 rows: 12 entries × 7 items)
- `experiments/results/p5p8/registry_audit_summary.json` (machine-readable)
- `experiments/results/p5p8/figures/registry_coverage.{png,pdf}` (heatmap)
- `paper/sections/p6_registry_health.tex` (new section, \secref{sec:p6-health})
- `paper/build/paper_P6_registry.pdf` rebuilt — **20 pages, 0 errors,
  0 undefined refs, 0 warnings**

## Headline findings (P6, iter 14)

1. **Schema-integrity claim verified.** 31/31 PASS; 0 broken delta
   pointers; 11/11 deltas claimed by stacks exist in the catalog.
   The DAPO row of the delta-status-mix table shows a four-way split
   (4 implemented / 1 surrogate / 3 absent / 2 unknown), which is
   exactly the label-flip pattern the registry exists to expose.

2. **`heldout_split` is universally reported (100%); `decontamination`
   is universally unreported on managed runtimes (0%).** The 100%
   vs. 20% spread on the two extremes of the seven-item list is the
   cleanest summary of where the registry's reporting gaps actually
   live. `sampler_backend` (95%) and `telemetry` (90%) are the next-best
   covered; `group_size_schedule` (68%), `reference_kl` (48%),
   `loss_form` (32%), and `decontamination` (20%) carry the gap.

3. **Per-openness inversion.** Managed runtime (`tinker`, 8 entries)
   reports 100% on `sampler_backend` and `telemetry` but only 22.9% on
   `loss_form` and 0% on `decontamination`. Open-source harnesses
   (10 entries across `colab-open`, `zvf130-batch`, and the framework
   config dumps) report 91.7% / 83.3% / 58.3% on those same items.
   The pattern inverts the conventional reading: the open backward-pass
   trainer is forced to declare every leaf, while the managed runtime
   declares the leaves its harness exposes.

4. **Per-framework maximum gap.** The five `zvf130-batch` entries
   (single-batch risk-index harness) report 0/30 leaves on
   `loss_form` and 0/15 on `reference_kl`. This is honest: the
   harness is a snapshot, not a full training run. Future entries
   that want to improve their badge should target `loss_form` first
   (the largest gap on managed stacks) and `decontamination` first
   (the largest gap on open stacks); the per-framework columns of
   `registry_audit_summary.json` are the machine-readable checklist.

5. **Backward compatibility.** The new `registry_audit.py` is
   independent of the existing `registry_validate.py` (iter 2). Both
   pass; `registry_validate.py` runs the schema check on every
   entry, and `registry_audit.py` adds the per-leaf coverage,
   cross-reference, and figure generation on top. The two scripts
   can run in either order in CI.

## Open questions for iter 15

- Should the per-item coverage thresholds be enforced (e.g. require
  `loss_form.frac_reported >= 0.5` for an entry to be published)?
- Should the registry schema add a `coverage_audit` block that
  every entry self-reports, so consumers can see the badge at a
  glance without running the script?
- Does adding `framework.evidence_tier` (full / batch / dry-run)
  let the badge aggregate correctly across tiers instead of being
  dominated by the four fully-instrumented `colab-open` arms?
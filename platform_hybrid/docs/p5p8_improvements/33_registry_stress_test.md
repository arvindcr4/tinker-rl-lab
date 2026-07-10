# Iter 26 Pillar 2 (P6) — Registry Schema Stress Test + Three Schema Bumps

**Status:** validated (iter 26)
**Class:** T1 (statistical rigor) + T3 (cross-paper structural coupling)
**Pick rationale:** Iter-14's registry health audit reported `31/31 schema
PASS` as a single point estimate. That estimate is incomplete — it never
probed whether the schema *catches* adversarial perturbations. Iter-26
adds a 13-category × 31-entry × 10-mutation stress suite (3,230 attempted
perturbations) that quantifies the schema's true-positive rate on
synthetic bad entries, then closes three latent gaps the stress test
surfaced. The bumped schema raises recovery from **86.4% [75.6%, 87.8%]**
to **100.0% [100.0%, 100.0%]**.

## Method

### 1. Stress-test harness (new in iter 26)

`platform_modal/scripts/p5p8/registry_stress_test.py` (~290 LoC, stdlib + jsonschema):

- For each of 13 mutation categories (drop a top-level required key; wrong
  type on `record_type`, `id`, `min_report`, `outcomes`; bad `id` regex;
  bad `framework.openness` enum; bad `variant_deltas_applied[].status`
  enum; bad `loss_form.advantage_normalization` enum; additional top-level
  property; drop a leaf inside a MIN-REPORT item; drop
  `provenance.source_artifacts`; flip a delta reference to a non-existent
  id) and each of the 31 entries, it runs 10 perturbations → 3,230
  total attempts.
- Each mutation: deep-copy the entry, apply the structural change, hand
  the perturbed record to `jsonschema.validate(perturbed, schema)`.
- The script records per-(entry, category, mutation_idx) whether the
  schema rejected the perturbation (caught = TP), accepted it (missed =
  FN), or skipped it because the entry shape didn't support the mutation.
- A paired over-attempt bootstrap CI (B=4000, seed=20260704) on the
  recovery rate is reported per category and overall.

### 2. Three schema bumps (iter 26)

The first stress-test run surfaced THREE latent schema bugs that the
iter-14 audit's per-leaf coverage check had silently allowed:

| Category                       | Pre-bump recovery | Post-bump recovery |
|--------------------------------|-------------------|---------------------|
| `bad_id_pattern`               | 64.5% (110/110 misses on `delta_*` records; the variant-delta `id` had no pattern check) | 100% |
| `drop_min_report_leaf`         | 0.0% (200/200 misses; min_report item leaves were optional) | 100% |
| `broken_delta_ref`             | 0.0% (130/130 misses; `delta_id` was a free-form string) | 100% |
| **All 13 categories combined** | **86.4% [75.6%, 87.8%]** | **100.0% [100.0%, 100.0%]** |

The patches (kept inside `platform_hybrid/registry/schema.json`, all current 31 entries
still PASS) are:

1. **Pattern on `variant_delta_record.id`** — copied from
   `stack_record.id` (`^[a-z0-9][a-z0-9_.-]*$`). The iter-15 citation
   audit had verified this pattern implicitly via brace-balanced titles,
   but the schema never enforced it.
2. **`required` on every MIN-REPORT item's leaves** — every property
   inside each of the 7 items is now in the item's `required` list.
   Every current entry already has every leaf (possibly null), so
   validation still passes for all 31 records; the new invariant is that
   a "leaf dropped" record now fails fast instead of silently passing.
3. **`enum` on `variant_deltas_applied[].delta_id`** — every valid
   `platform_hybrid/registry/entries/delta_*.json` stem is now an allowed value.
   Adds a small maintenance burden (run
   `platform_modal/scripts/p5p8/regenerate_schema_delta_enum.py` whenever a new delta is
   added), which is principled because broken refs were previously caught
   only by `registry_audit.py`'s post-hoc check (`variant_delta_xref`).

### 3. Schema-bump helper

`platform_modal/scripts/p5p8/regenerate_schema_delta_enum.py` (~60 LoC, stdlib +
optional jsonschema): regenerates the `delta_id` enum in
`platform_hybrid/registry/schema.json` from the current `platform_hybrid/registry/entries/delta_*.json`
set, then validates all entries still PASS. Exit code = 0 only if every
entry passes the bumped schema.

### 4. Re-validation

After the bumps, the existing `platform_modal/scripts/p5p8/registry_audit.py` still
reports `31/31 PASS` (the per-leaf coverage numbers are unchanged
because the underlying entries are unchanged; only their schema
constraint set is tighter).

## Headline findings

1. **The schema's structural recovery rate is now 100.0% on the
   N=3,230 perturbed-attempt stress test.** The pre-bump 86.4% rate
   included 3 silent failures in 13 categories. The iter-26 bump
   closes them all.
2. **None of the 31 current entries were broken by the bumps.**
   `31/31 PASS` after each bump; the patches are conservative (they
   tighten constraints; they don't add new properties).
3. **The `variant_deltas_applied[].delta_id` enum is the closest to a
   "living" schema constraint.** It must be re-generated each time a
   new delta is added; the helper script handles that automatically
   and exits non-zero if the enum regen would break any entry.
4. **The iter-14 audit's per-leaf coverage (e.g. `loss_form 38/120 =
   31.7%`) is unchanged by the bumps** because the underlying entries
   still report the same value distribution (nulls are still nulls).
   The new invariant is about **key presence**, not **value quality** —
   so a contributor can no longer accidentally ship an entry with a
   missing leaf.

## What this iteration does NOT change

- The 31 entries themselves are byte-for-byte identical.
- The 7-item MIN-REPORT field set is unchanged (no new required MIN-REPORT
  items; this iter strengthens the schema, not the manifest schema).
- The iter-14 audit's per-leaf coverage numbers carry forward unchanged.
- The stackdiff R0–R5 ladder and `query.py` semantics are unchanged.
- No NEW Tinker API calls were used; the stress test is a pure
  synthetic-data validator over the existing 31-entry registry.

## Reproducibility

```bash
python3 platform_modal/scripts/p5p8/registry_stress_test.py --n-mutations-per-category 10 --seed 20260704
python3 platform_modal/scripts/p5p8/regenerate_schema_delta_enum.py
python3 platform_modal/scripts/p5p8/registry_audit.py
```

Expected summary (post-bump):
- `n_attempts_total=3230, n_caught_total=3230, n_misses=0`
- `overall_recovery_rate=1.0`, `overall_recovery_rate_ci95=[1.0, 1.0]`

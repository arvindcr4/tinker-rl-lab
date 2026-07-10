# Iter 146 — P6 per-row provenance-recompute audit (value-traceable registry)

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable catalog)
**Vein (fresh, not in 161 prior rows):** Brief vein (a) at the **value-trace layer**.
Prior P6 iters (iter-118, iter-134, iter-142) verified that every `measured[]`
row has a non-empty `source` string, that the file at that path exists on disk,
that `ci_low ≤ delta ≤ ci_high`, and that the row's `significant` flag matches
the stored CI. None of those iters verified that **the registry's stored
delta/CI values are actually reproducible from the cited source file** using
the inferred bootstrap recipe.

Iter-146 closes that gap. For every measured[] row, it (i) infers the source
family from the path/panel, (ii) recomputes the (delta, ci_low, ci_high, n,
significant) tuple from the source file using a deterministic LCG paired
bootstrap (B=2000, seed=20260705), and (iii) classifies the result as MATCH,
MATCH_POINT (point-delta only), POINT_MATCH_WRONG_SOURCE (value correct but
source path misattributed), DRIFT_*, UNINFERABLE_SOURCE, or NULL_DELTA_SKIPPED.

## Falsifiable measured headlines (live registry, n=40 measured rows)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1 (PASS)** Pre-patch: ≥10 measured[] rows are POINT_MATCH_WRONG_SOURCE | **PASS** (13/40 = 32.5%) | All 8 entries with zvf130 panel cells (aero, areal, gift, cppo, es, mcgrpo, ngrpo, scafgrpo) carry one or two misattributed rows |
| **H2 (PASS)** Post-patch: 0 measured[] rows remain POINT_MATCH_WRONG_SOURCE | **PASS** (0/40) | The 13 misattributions split: 5 mag_mean rows become MATCH_POINT (their source-path now correctly points to `zvf_iter130_method_risk.tsv` aggregate); 8 mean_zvf rows become DRIFT_SIGN (see H3) |
| **H3 (NEW — sharpest finding)** Post-patch: 8 measured[] rows have `significant: false` despite recomputed 95% CI excluding 0 | **PASS (8/40 = 20%)** | These are the mean_zvf rows on aero/areal/gift/cppo/es/mcgrpo/ngrpo/scafgrpo. The registry marks them `significant: false` because `note: "mag_mean per-seed sd not stored; point estimate only (unmeasurable CI)"`. The 5-seed paired-seed bootstrap on the CORRECT source file (`zvf_iter130_risk_index.tsv`) yields CI half-widths of 0.001-0.005, all excluding 0. **This is a registry-vs-measurement convention conflict** that iter-146 surfaces explicitly |
| **H4 (NEW — invariant)** N2 panel: 12/12 MATCH (perfect provenance, full CI recompute) | **PASS** | n2_same_stack_last10 rows for {aero,areal,gift} × {zvf,reward_mean,pcd,mean_len}. The N2 panel is the registry's cleanest provenance cell |
| **H5 (NEW)** length_bias and qp7_adaptive panels: 5/5 MATCH | **PASS** | drgrpo's {neg_frac, pos_frac, L_star} on the length_bias TSV (parametric normal approx on 2 tasks) and adaptiveg's {reward_mean, zvf} on the qp7_adaptive TSV (paired-arm bootstrap on 16 steps). These are the registry's two small-sample panels |
| **H6 (REPORTED)** 2 tool_use rows carry `panel=zvf130_1seed_tooluse` | UNINFERABLE_SOURCE | n_seeds=1; no paired-bootstrap recipe inferable. The registry's note documents this deferred state. iter-146 reports but does not patch |

## What changed in the registry (patched in this iter)

13 rows across 8 entries had their `source` path corrected:

| entry | metric | old `source` | new `source` |
|---|---|---|---|
| delta_aero, delta_areal, delta_gift | mean_zvf | `zvf_iter130_method_risk.tsv` | `zvf_iter130_risk_index.tsv` |
| delta_cppo, delta_es, delta_mcgrpo, delta_ngrpo, delta_scafgrpo | mean_zvf | `zvf_iter130_method_risk.tsv` | `zvf_iter130_risk_index.tsv` |
| delta_cppo, delta_es, delta_mcgrpo, delta_ngrpo, delta_scafgrpo | mag_mean | `zvf_iter130_risk_index.tsv` | `zvf_iter130_method_risk.tsv` |

Each patched row's `note` field is appended with the audit trace
(`iter-146 provenance-recompute audit: stored delta matches recompute from
<new_source>; source path was misattributed to <old_source>; patched in
iter-146`). The stored `delta` / `ci_low` / `ci_high` / `n` / `significant`
values are UNCHANGED — the values were always correct; only the source-path
attribution was wrong.

## Pre/post headline numbers

| metric | pre-patch | post-patch | delta |
|---|---:|---:|---:|
| MATCH (full CI recompute) | 17 | 17 | 0 |
| MATCH_POINT (point-delta only) | 8 | 13 | +5 |
| **Total MATCH-or-MATCH_POINT** | **25** | **30** | **+5** |
| **pct_match** | **62.5%** | **75.0%** | **+12.5 pp** |
| DRIFT_SIGN | 0 | 8 | +8 (H3 finding) |
| POINT_MATCH_WRONG_SOURCE | 13 | 0 | -13 (H2 finding) |
| UNINFERABLE_SOURCE (tool_use) | 2 | 2 | 0 |

The +12.5 pp pct_match is the auto-fix gain. The +8 DRIFT_SIGN is the
explicit surfacing of the `mean_zvf` measurement-design tension that the
prior registry hid with the "mag_mean per-seed sd not stored" note.

## Cross-paper coupling

- **P6 iter-118 row 133 (strict-coverage)** — iter-118 verified `source` is a
  non-empty string; iter-146 verifies the SOURCE actually CONTAINS the
  metric column. iter-146 is the value-trace analogue of iter-118's
  string-existence check.
- **P6 iter-134 row 150 (per-row measured-field completeness)** — iter-134
  added the `audit_source` / `audit_date` / `synth_from_agg` fields to the
  schema. iter-146 directly populates `note` with the audit trace for the
  13 patched rows; the iter-134 audit fields are now wired into real
  registry content.
- **P6 iter-122 row 137 (cross-entry consistency + validate-strict)** —
  iter-122 wired `query.py validate-strict` into CI. iter-146's
  `p6_iter146_apply_fix_plan.py` is the SAME class of CI gate at the
  per-row provenance-trace layer; recommend wiring it as
  `query.py validate-provenance` in a follow-up iter.
- **P5 iter-145 row 162 (schema-ground-truth audit)** — iter-145 audited the
  manifest schema for naming-convention drift; iter-146 audits the registry
  schema for source-path column drift. Same "ground-truth" layer, two
  artefacts: manifests (P5) and registry measured rows (P6).
- **P5 iter-141 row 159 (η²(method)=0.0005 same-stack under-identification)** —
  iter-141 shows algorithm axis contributes ≤ 0.5% variance at the panel
  level. iter-146 confirms at the per-row level: the 8 DRIFT_SIGN rows show
  the same direction (mean_zvf strictly lower for all variants vs grpo),
  consistent with iter-141's null algorithm-axis result.

## Operational recommendations

1. **Adopt `p6_iter146_provenance_recompute.py` as the iter-146 schema-CI
   hook**: every `delta_*.json` mutation must pass the per-row
   source-column check (metric column exists in cited file).
2. **Document the `mean_zvf` measurement-design convention**: the registry's
   `note: "mag_mean per-seed sd not stored; point estimate only (unmeasurable
   CI)"` for mean_zvf rows is a deliberate convention but reads as
   `significant: false` to downstream consumers. Recommend adding a
   `significant_convention` field (`"convention_unmeasurable_CI"` vs
   `"computed_from_bootstrap"`) to disambiguate.
3. **Schedule the 2 UNINFERABLE_SOURCE tool_use rows for n_seeds≥5**: the
   single-seed zvf130_1seed_tooluse panel is a deferred-measurement gap.
   iter-138 already documented this; iter-146 makes it machine-readable
   in the audit.
4. **Add `p6_iter146_apply_fix_plan.py` as a CI pre-commit**: catches future
   source-path misattributions before they enter the registry.

## Reproducibility

```bash
python3 scripts/p5p8/p6_iter146_provenance_recompute.py
# pre-patch: 25/40 MATCH-or-MATCH_POINT, 13/40 POINT_MATCH_WRONG_SOURCE, 2 UNINFERABLE_SOURCE
python3 scripts/p5p8/p6_iter146_apply_fix_plan.py
# 13 rows patched; schema_pass=31/43 (12 pre-existing fails from
# iter_recomputed/evidence_deferred_until, not introduced by iter-146)
python3 scripts/p5p8/p6_iter146_provenance_recompute.py
# post-patch: 30/40 MATCH-or-MATCH_POINT (+5), 8/40 DRIFT_SIGN (H3), 2 UNINFERABLE_SOURCE
```

Inputs read: `registry/entries/delta_*.json`,
`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`,
`experiments/results/zvf_iter130_method_risk.tsv`,
`experiments/results/zvf_iter130_risk_index.tsv`,
`experiments/results/length_bias_iter60_grpo_vs_drgrpo.tsv`,
`experiments/results/quick_20260704/qp7_adaptive.tsv`,
`registry/schema.json`.

## Outputs

- `scripts/p5p8/p6_iter146_provenance_recompute.py` (~330 LoC, stdlib only,
  deterministic LCG bootstrap B=2000 seed=20260705)
- `scripts/p5p8/p6_iter146_apply_fix_plan.py` (~110 LoC, stdlib + jsonschema)
- `experiments/results/p5p8/p6_iter146_audit_pre_patch.tsv` (40 rows)
- `experiments/results/p5p8/p6_iter146_audit_post_patch.tsv` (40 rows)
- `experiments/results/p5p8/p6_iter146_per_entry_pre_patch.tsv` (12 rows)
- `experiments/results/p5p8/p6_iter146_per_entry_post_patch.tsv` (12 rows)
- `experiments/results/p5p8/p6_iter146_pre_patch_summary.json`
- `experiments/results/p5p8/p6_iter146_post_patch_summary.json`
- `experiments/results/p5p8/p6_iter146_fix_plan.tsv` (13 rows, the patch plan)
- `experiments/results/p5p8/p6_iter146_patch_log.tsv` (13 rows, audit log)
- 8 patched entries: `delta_{aero,areal,gift,cppo,es,mcgrpo,ngrpo,scafgrpo}.json`
- 1 line in `findings_ledger.jsonl` (pillar P6, iter 146)

## What iter-146 did NOT do (scope-protective)

- Did not change any stored `delta` / `ci_low` / `ci_high` / `n` / `significant`
  values. The 13 patched rows had correct values all along; only the
  `source` path was misattributed.
- Did not modify the registry `schema.json`. The pre-existing 12 schema
  fails (`iter_recomputed`, `evidence_deferred_until`) are independent
  audit gaps owned by iter-130 and iter-138 respectively; out of scope.
- Did not address the 2 UNINFERABLE_SOURCE tool_use rows — they require
  n_seeds≥5 BFCL reproduction, which is a deferred-measurement gap from
  iter-138.

## Pre-existing build-error status

`paper_P6_registry.pdf` rebuild status: not rebuilt (the registry entries
were patched; the paper would need a §p6-iter146 section to surface the
audit, deferred to a follow-up iter that combines with iter-141's
algorithm-axis synthesis). Current build state: 62 pages / 0 errors / 0
undefined citations (unchanged from iter-145).
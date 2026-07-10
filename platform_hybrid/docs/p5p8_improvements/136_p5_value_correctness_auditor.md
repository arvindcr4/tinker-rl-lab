# 136 — P5 MIN-REPORT v2.2 value-correctness mutation stress test (iter 121)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Type:** T1 + T3 — auditor robustness + cross-paper coupling (auditor-on-auditor)
**Status:** proposed → **validated** (iter 121)
**Vein (fresh, not in 135 prior rows):** Closes brief vein (a) at the
**AUDITOR-ROBUSTNESS layer**. Iter-97 audited schema-level presence;
iter-105 audited per-value-class presence; iter-113 audited
declared-vs-emitted-vs-derivable content; iter-117 audited per-location
encoding mode. **None of them stress-tested the AUDITORS THEMSELVES
against controlled perturbations.** Iter-121 audits the auditor.

## Problem statement

A presence-only audit ("is field X present in the manifest?") can pass
on a manifest whose values are silently wrong. The value-correctness
auditor (iter-121) must be tested against controlled mutations to
quantify: (a) **detection rate** for each mutation class, (b) **blind
spots** (mutations that pass the auditor unchanged), and (c) the
**check-fanout** structure (which check fires on which mutation).

## Method

1. Load n=98 live mega manifests from `experiments/results/mega_20260704/manifests/`.
2. Define 8 controlled mutations, each corrupting a known MIN-REPORT v2.2
   item by flipping the value to a known-wrong one (preserving format
   but breaking semantics):
   - **M1** cell_id_swap_hash — replace last 8 hex chars with `'0' * 8`
   - **M2** model_family_swap — swap `meta-llama-Llama-3-2-3B` ↔ `Qwen-Qwen3-5-4B`
   - **M3** task_slice_swap — swap `gsm8k_easy` ↔ `humaneval_subset`
   - **M4** G_swap — flip `G=2` ↔ `G=32` in cell_id filename
   - **M5** temperature_swap — flip `_t0.6_` ↔ `_t1_` in cell_id filename
   - **M6** seed_swap — flip `_s0_` ↔ `_s1_` in cell_id filename
   - **M7** heldout_split_swap — flip JSON-body `heldout_split` `gsm8k_easy` ↔ `humaneval_subset`
   - **M8** per_step_zvf_path_break — append `__broken.json` to JSON-body `per_step_zvf_path`
3. Apply each mutation to each cell; re-run the 10-check value-correctness
   audit (C01..C10). A mutation is "caught" if AT LEAST ONE check flips
   from PASS (baseline) to FAIL (mutated).
4. Aggregate detection rate per mutation × per check.

## Hypotheses tested

- **H1 — every mutation is caught on ≥1 check** — falsifiable: the
  auditor is complete (no silent corruption).
- **H2 — every mutation has a stable dedicated check** — falsifiable:
  each mutation class is owned by exactly one check (one-to-one mapping
  M_i ↔ C_j with detection_rate ≥ 0.95).
- **H3 — cell_id mutations are caught by cell_id-only checks** — falsifiable:
  mutations that change cell_id (M1, M2, M3, M4, M5, M6) flip C01;
  mutations that change JSON body (M7, M8) flip C07/C10.
- **H4 — silent-corruption mutations exist** — falsifiable: there is at
  least one mutation class with detection_rate = 0.0 (auditor has blind spots).

## Measured results

| mutation | name | n_cells | n_caught | detection_rate | top_check | top_count |
|----------|------|---------|----------|----------------|-----------|-----------|
| M1 | cell_id_swap_hash | 98 | **0** | **0.000** | C01 | 0 |
| M2 | model_family_swap | 98 | **0** | **0.000** | C01 | 0 |
| M3 | task_slice_swap | 98 | 98 | **1.000** | C03 | 98 |
| M4 | G_swap | 98 | 98 | **1.000** | C04 | 98 |
| M5 | temperature_swap | 98 | 98 | **1.000** | C05 | 98 |
| M6 | seed_swap | 98 | 98 | **1.000** | C06 | 98 |
| M7 | heldout_split_swap | 98 | 98 | **1.000** | C07 | 98 |
| M8 | per_step_zvf_path_break | 98 | 98 | **1.000** | C10 | 98 |

**Per-check fanout** (across all 8 mutations × 98 cells = 784 evaluations):

| check | name | n_catches | fraction |
|-------|------|-----------|----------|
| C01 | cell_id_json_eq_filename | **0** | 0.0% |
| C02 | model_family_filename_canonical | **0** | 0.0% |
| C03 | task_slice_filename_eq_cells | 98 | 12.5% |
| C04 | G_filename_eq_cells | 98 | 12.5% |
| C05 | temperature_filename_eq_cells | 98 | 12.5% |
| C06 | seed_filename_eq_cells | 98 | 12.5% |
| C07 | heldout_split_json_eq_task_slice | 98 | 12.5% |
| C08 | group_size_schedule_contains_G | **0** | 0.0% |
| C09 | decontamination_contains_task_prefix | **0** | 0.0% |
| C10 | per_step_zvf_path_exists_on_disk | 98 | 12.5% |

## Verdicts

- **H1 — REFUTED.** Two mutation classes (M1 cell_id hash swap, M2
  model_family swap) are caught on 0/98 cells. The auditor has blind
  spots.
- **H2 — PARTIALLY PASS.** 6/8 mutations have a stable dedicated check
  with detection_rate = 1.0 (M3→C03, M4→C04, M5→C05, M6→C06, M7→C07,
  M8→C10). The remaining 2 (M1, M2) have NO check.
- **H3 — REFUTED for M1.** H3 expected M1..M6 (cell_id mutations) to
  flip C01. Instead M1 flips NO check (cell_id JSON-basename match
  passes because the new cell_id matches the new filename basename;
  no check validates the hash suffix against a registry). M2 also flips
  NO check (C02 only validates the model token against a canonical set,
  not against cells.tsv `model_family`).
- **H4 — CONFIRMED.** M1 and M2 are silent-corruption blind spots with
  detection_rate = 0.0 across 98 cells (0/196 = 0.0% combined).

## Why this matters (the cross-paper fingerprint)

The iter-121 value-correctness audit (passes 100% on the unmutated
corpus, see summary `p5_iter121_summary.json`) and the iter-121
mutation stress test (REFUTES H1, CONFIRMS H4) jointly establish:

1. **The unmutated corpus is internally consistent** (10/10 checks
   pass on 98/98 cells, 100.0% strict-pass rate).
2. **The auditor has 2 specific silent-corruption blind spots**:
   - **C01 catch-zero**: the cell_id hash suffix (last 8 hex chars)
     is never validated. A manifest whose cell_id hash does not match
     a known-registry entry slips through silently. Operational risk:
     a duplicate-cell_id collision (two cells with the same
     `<model>_<task>_G<g>_t<t>_s<seed>_<hash>` prefix but different
     hashes) is undetectable by the audit.
   - **C02 catch-zero**: the model_family filename token is validated
     against a 2-element canonical set, not against the cells.tsv
     ground-truth. A manifest that claims `meta-llama-Llama-3-2-3B` but
     whose cells.tsv row has model_family `Qwen/Qwen3.5-4B` slips
     through. Operational risk: a model_family mislabel in the filename
     that disagrees with cells.tsv is undetectable.
3. **The detection fanout is mostly one-to-one (M_i ↔ C_j)**. This is
   a positive structural property: each mutation class has a stable
   dedicated check, which means future extension of the auditor (new
   checks for new items) can be added without breaking existing
   detection.

## Cross-paper coupling

- **P5 iter-105 row 121** (live-manifest per-value coverage) — iter-105
  proved per-value-class presence; iter-121 extends this to
  per-value-correctness with controlled-mutation stress testing.
- **P5 iter-113 row 127a** (MIN-REPORT v2.2 declared-vs-emitted-vs-derivable)
  — iter-113 closed the CONTENT gap; iter-121 closes the
  AUDITOR-ROBUSTNESS gap.
- **P5 iter-117 row 132** (structural-encoding layer) — iter-117
  audited per-location encoding; iter-121 audits per-value correctness
  with respect to cells.tsv ground-truth.
- **P6 iter-94 row 110** (registry schema validator) + iter-102 row 122
  (crossref-integrity guard) — P6's validator runs `jsonschema.check_schema`
  on registry entries; iter-121's mutation stress test is the same
  pattern at the manifest layer with controlled perturbations instead
  of schema-level checks.

## Operational recommendation

**(a)** The iter-121 value-correctness audit currently passes 100% on
the unmutated corpus. **No CI gate action is needed yet.**
**(b)** Add 2 new checks to close the H4 blind spots:
  - `C11` — cell_id_hash matches a known-registry hash (requires
    maintaining a registry of valid hashes per `<model>_<task>_G<g>_t<t>_s<seed>` prefix).
  - `C12` — model_family filename token matches cells.tsv model_family
    (after normalizing `/` ↔ `-`).
**(c)** Document in §p5_evidence that the iter-121 audit is "presence+correctness
for 7 explicit-json-key items + filename-vs-cells.tsv for 5 stack axes,
NOT including hash-suffix or model-family cross-validation". This honest
scoping is the iter-121 deliverable; it sets the work-list for iter-122.

## Deliverables

- `scripts/p5p8/p5_iter121_value_correctness.py` (~210 LoC, stdlib only)
- `scripts/p5p8/p5_iter121_mutation_stress.py` (~210 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter121_value_correctness.tsv` (98 rows)
- `experiments/results/p5p8/p5_iter121_value_correctness_per_item.tsv` (10 rows)
- `experiments/results/p5p8/p5_iter121_summary.json` (machine-readable)
- `experiments/results/p5p8/p5_iter121_mutation_stress.tsv` (784 rows: 8 mutations × 98 cells)
- `experiments/results/p5p8/p5_iter121_mutation_summary.json` (per-mutation + per-check fanout + H1-H4 verdicts)
- 1 line in `findings_ledger.jsonl` (pillar P5, iter 121)
- `paper/sections/p5_iter121_value_correctness.tex` (~75 lines, NEW)
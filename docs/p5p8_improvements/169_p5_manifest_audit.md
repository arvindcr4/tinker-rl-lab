# Iter 169 — P5 MIN-REPORT manifest ground-truth audit

## Vein
Brief vein (a) at the **actual JSON manifest files on disk** layer (not the
cells.tsv aggregate layer, not a curated subset). Distinct from prior P5
audits (iter-1 row 01 schema-coverage, iter-14 row 14 cells.tsv leaf-coverage,
iter-18 row 17 MIN-REPORT-RL auditor, iter-73 row 11 stack-axis MIN-REPORT,
iter-145 row 32 schema-ground-truth).

## What it audits

For every JSON file in `experiments/results/mega_20260704/manifests/`:

1. **Per-leaf presence + type correctness**: each of the v1 seven MIN-REPORT
   items (`loss_form`, `ref_policy_kl`, `sampler_backend_precision`,
   `per_step_zvf_path`, `group_size_schedule`, `heldout_split`,
   `decontamination_notes`) is present as a `str` on the manifest.
2. **Value-plausibility** (per-key Shanon entropy + n_unique).
3. **`per_step_zvf_path` truthiness** — does the path resolve to a real JSON
   file? Does its basename equal the `cell_id`?
4. **Schema/data coherence** between manifest declarations and `cell_id` axis
   encoding — group_size_schedule vs parsed G, heldout_split vs parsed task.
5. **Cells.tsv cross-coherence** — the `(model_family, task_slice, G,
   temperature, seed)` axes parsed out of cell_id agree with the
   corresponding cells.tsv ground-truth columns after an encoding
   normalization (`Qwen/Qwen3.5-4B` cells.tsv vendor prefix vs
   `Qwen-Qwen3-5-4B` cell_id encoding; floats for temperature to handle
   `t1` vs `1.0`).
6. **v2 expansion potential** — how many fresh entropy bits would the v2
   schema (model_family/task_slice/G/temperature/seed) carry beyond what v1
   already carries in stack-descriptor fields.

## 5 falsifiable hypotheses (5/5 PASS)

| # | hypothesis | bar | measured | verdict |
|---|------------|-----|----------|---------|
| **H1** | every v1 item has leaf-presence >=98/98 | >= 98/98 | 98/98 on all 7 | **PASS** (decisive) |
| **H2** | >=3 v1 items are PLACEBO at the manifest layer (zero stack-discriminative bits) | >= 3 placebo | exactly 3 placebo: `loss_form` ("n/a-sampling"), `ref_policy_kl` ("n/a"), `sampler_backend_precision` ("tinker-closed") | **PASS** (decisive) |
| **H3** | `per_step_zvf_path` resolves to an existing JSON file on >=98% of manifests; basename = cell_id | >= 98% | 98/98 (100%) path-exists, 98/98 (100%) basename-match | **PASS** (decisive) |
| **H4** | manifest group_size_schedule matches G-from-cell_id on >=98%; heldout_split matches task-from-cell_id on >=98% | >= 98% each | 98/98 (100%) on both | **PASS** (decisive) |
| **H5** | every parsed (model_family, task_slice, G, temperature, seed) axis matches cells.tsv ground truth on >=98% of cells | >= 98% per axis | match-model-vendor 98/98, match-task 98/98, match-G 98/98, match-temp 98/98, match-seed 98/98, match-zvf-path-basename 98/98 | **PASS** (decisive) |

## Sharpest structural findings

* **Placebo-triple + 3-discriminator split**: 3 v1 items carry zero
  stack-discriminative entropy at the manifest layer; 3 carry
  $\{2.31, 1.55, 0.93\}$ bits of entropy, jointly $H{=}4.80$ bits.
* **`per_step_zvf_path` is a verifiable on-disk claim, not a label**: at the
  file-pointer layer, every manifest's declared path resolves to an existing
  JSON file whose basename equals `cell_id`. This is the only MIN-REPORT item
  that is verifiable against `os.path.exists`, and the audit confirms the
  declarative intent is honest.
* **v2 expansion potential = +2.99 fresh bits**: of the v2 schema's $H{=}6.86$
  bits of stack-axis entropy, two axes (`task_slice`, `G`) are already
  carried by v1 stack descriptors (`heldout_split`, `group_size_schedule`).
  The truly fresh contributions are `model_family` ($H{=}1.00$), `temperature`
  ($H{=}0.995$), `seed` ($H{=}1.00$), jointly $H{=}2.99$ bits — a +62%
  boost over the v1 stack-discriminator subset.
* **Manifest ↔ cells.tsv coherence is exact on 6 axes across all 98 cells**.
  The cell_id is the **single source of truth** that the manifest and
  cells.tsv agree on. Every future mega harvest can use this audit as a
  pre-commit gate to prevent silent manifest/cells drift.

## Cross-paper coupling

* **P5 iter-1 row 01** (1008-line-curated MIN-REPORT schema-coverage):
  iter-1 measured the 7-item MIN-REPORT schema against a curated subset;
  iter-169 measures the 7 items at the file-level on the live corpus.
* **P5 iter-14 row 14** (cells.tsv leaf-coverage): iter-14 audited the
  cells.tsv aggregate layer for 11 measured-telemetry fields; iter-169
  audits the *source* layer (manifest JSON files).
* **P5 iter-73 row 11** (stack-axis MIN-REPORT): iter-73 measured
  v1_stack_discriminative_h = 4.798 bits across the same 98 manifests;
  iter-169 reproduces 4.798 exactly.
* **P5 iter-145 row 32** (schema-ground-truth): iter-145 cross-referenced
  cells.tsv schema field names; iter-169 cross-references the manifest
  JSON declarations against both cell_id axis parsing AND cells.tsv ground
  truth.
* **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability)**: the placebo
  triple (3/7 items constant on this corpus) is the structural evidence
  that the v2 schema is overdue — exactly what the iter-73 row 11
  analysis foreshadowed. 7-item MIN-REPORT has a structural underuse rate
  of 3/7 = 0.43 at the present run-mix.

## Operational recommendations

1. **DEPLOY** `python3 scripts/p5p8/p5_iter169_manifest_audit.py` as a
   pre-commit CI gate that fails any new manifest whose 7-item leaf-presence
   drops below 100% (H1 expectation).
2. **EXPAND** the manifest emitter to add a `model_family` / `temperature`
   / `seed` triplet alongside the v1 7-item schema, so v2 backward
   compatibility is preserved and the +2.99 fresh bits become
   standardly-captured.
3. **DOCUMENT** the placebo-triple fact in `paper_P5_minreport.tex`
   §sec:p5-stack as the structural justification for v2 schema expansion.
4. **VALIDATE** the cells.tsv cross-coherence (H5) on every future mega
   harvest as the canonical manifest/cells.tsv agreement gate.

## Artifacts emitted

* `scripts/p5p8/p5_iter169_manifest_audit.py` (~290 LoC, stdlib only).
* `experiments/results/p5p8/p5_iter169_manifest_audit_per_cell.tsv` (98 rows ×
  35 cols: per-cell leaf-presence + path truthiness + parsing).
* `experiments/results/p5p8/p5_iter169_manifest_audit_per_key.tsv`
  (8 rows × 7 cols: per-key n_populated, n_unique, h_bits, is_placebo,
  top_value, top_value_freq).
* `experiments/results/p5p8/p5_iter169_manifest_audit_cells_join.tsv`
  (98 rows × 17 cols: per-cell parse + cells.tsv join with 6 match flags).
* `experiments/results/p5p8/p5_iter169_summary.json` (H1-H5 verdicts +
  per-key entropy + v2 expansion potential).
* `paper/sections/p5_iter169_manifest_audit.tex` (NEW §sec:p5-iter169-manifest-audit
  with 5 H sub-sections + placebo-triple + v2 expansion potential + 4
  cross-paper coupling bullets + 4 operational recommendations).
* `paper/build/paper_P5_minreport.pdf` rebuilds to 69 pages, 0 errors,
  0 undefined citations (was 65 before iter-169).
* `docs/p5p8_improvements/169_p5_manifest_audit.md` (this file).
* 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P5, iter 169).

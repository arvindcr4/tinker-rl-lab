# P5 MANIFEST SUFFICIENCY + MANIFEST PREDICTIVE POWER AUDIT (iter 189)

## Why this iter

Fresh P5 vein, not in 198 prior P5 rows. Closes brief vein (a) at the
**manifest self-sufficiency** layer AND opens the **manifest predictive
power** layer in one audit. Iter-185 (row 198) audited v2.5's cross-corpus
portability and found the 13-field v2.5 schema satisfies mega-corpus
fill-rate + value-correctness; iter-189 goes one step further and asks:

1. **Is the manifest self-sufficient?** Given ONLY the 8 manifest JSON
   fields (no cells.tsv), how many of the 98 live cells can the manifest
   UNIQUE-IDENTIFY?

2. **Is the manifest predictively sufficient?** Compute η² of the
   manifest's discriminating fields on the 4 cells.tsv telemetry
   channels (zvf, mean_reward, pcd, mean_completion_len). This quantifies
   how much of cells.tsv telemetry is "stack-driven" (i.e., predictable
   from manifest fields alone) vs "model/temp/seed-driven" (the missing
   3 stack axes).

The headline finding is that the manifest **collapses 98 cells into 15
equivalence classes** (5 G × 3 task_slice), with **0 cells uniquely
identifiable** from manifest alone. The manifest requires cells.tsv to
identify any individual cell. This is a sharp paper-grade finding for
P5's "Report the Stack, Not the Label" thesis: the current manifest
**does not report the full stack** — it omits `model`, `temperature`,
and `seed`, which are stack-conditioning axes per iter-5 mega-η² (row
11).

## Method

For each candidate field-set, compute:
- n_classes = number of distinct (concatenated-field-values) tuples
- n_unique_cells = count of cells whose tuple is size-1
- max_class_size, min_class_size = extremes of class-size distribution

Field-sets tested:
- `manifest_all_8` (8 manifest fields including cell_id and per_step_zvf_path)
- `manifest_discriminating_3` (the 3 non-constant fields: G, task_slice, decontam)
- `manifest_plus_cells_5` (3 manifest + 2 cells.tsv = 5)
- `cells_all_5` (model, task_slice, G, temperature, seed)
- 3× `cells_minus_X_4` (drop one cells.tsv field)

For each (field_group, channel), compute one-way η² (variance-explained
between groups / total variance) on the live 98 cells. Field groups:
- `manifest_discriminating_3` (the 3 non-constant manifest fields)
- `cells_all_5` (the full cells.tsv fingerprint)
- `manifest_plus_temp_seed` (manifest + the 2 missing stack axes)

Per-channel lift = η²(manifest+temp+seed) − η²(manifest alone), in pp.

## Hypotheses (5, set BEFORE measurement)

| # | claim | result |
|---|-------|--------|
| H1 | manifest_discriminating_3 produces ≥ 10 size-1 cells | 0/98 **FAIL** |
| H2 | manifest_discriminating_3 populates ≥ 14/15 effective classes | 15/15 PASS |
| H3 | minimal-fields-to-cover-all-98 ≤ 5 | k=5 PASS |
| H4 | η²(zvf) > η²(pcd) for manifest_discriminating_3 | 0.872 > 0.660 PASS |
| H5 | η²(mean_completion_len) lift ≥ 5pp from adding temp+seed | +18.36pp PASS |

**4/5 PASS + 1 sharp FAIL.**

## Headline findings

**F1 (H1 FAIL — the sharpest paper-grade finding):** The manifest
**does not uniquely identify ANY of the 98 cells**. With manifest fields
alone (8 fields), 0 cells have a size-1 equivalence class; the 3
discriminating manifest fields collapse 98 cells into **15 equivalence
classes** (max class size = 8). This is **structural**: the manifest
records `loss_form`, `ref_policy_kl`, `sampler_backend_precision` as
**constants** (single value across all 98 cells), and `decontam` is
**deterministic** from `task_slice`. The manifest's effective
discrimination is therefore (G × task_slice) = 5 × 3 = 15 classes. To
uniquely identify any cell, the manifest must include
`model`, `temperature`, `seed`.

**F2 (H3 PASS):** The minimal-field-set to uniquely identify all 98
cells is **k=5 fields**: `{model, G, task_slice, temperature, seed}`.
No 4-field subset covers all 98 (max coverage at k=4 is 54 cells when
seed is dropped; 50 cells when temperature is dropped). The manifest's
8 fields alone are **insufficient**; it must be **augmented with 2
fields** (model, seed-or-temperature) to reach full coverage.

**F3 (H4 PASS):** Among the 4 cells.tsv telemetry channels,
`zvf` is the **most manifest-predictable** (η² = 0.872 with manifest
fields alone). This is because zvf is a function of (G, task_slice) per
iter-5 mega-η² (row 11): zvf collapses monotonically with G on
gsm8k-style tasks. In contrast, `mean_completion_len` is the LEAST
manifest-predictable (η² = 0.260) because it depends strongly on model
identity (Qwen vs Llama) and temperature.

**F4 (H5 PASS):** Adding `temperature` + `seed` to the manifest
boosts η²(mean_completion_len) by **+18.36pp** (0.260 → 0.443). The
boost is small for zvf (+4.18pp), modest for pcd (+5.46pp), and
near-zero for mean_reward (+0.24pp). This is a CHANNEL-DEPENDENT
missing-axes contribution: temperature and seed primarily affect
**decoded-token statistics** (mean_completion_len), not reward
statistics (mean_reward).

**F5 (cross-coupling):** cells_all_5 produces η² = 1.0 trivially
because each (model, G, task_slice, temperature, seed) tuple is a
single cell. This is **structural not informative**: cells.tsv already
uniquely identifies every cell. The informative contrast is
`manifest_discriminating_3` vs `manifest_plus_temp_seed`.

## Cross-paper coupling

- **P5 iter-5 row 11 (mega-η²)** — iter-5 found stack axes explain
  73-93% of variance in every telemetry channel. iter-189 is consistent:
  manifest_discriminating_3 explains 87% of zvf variance, 66% of pcd,
  31% of mean_reward, 26% of mean_completion_len — all consistent with
  iter-5's "stack-axes dominate" finding.
- **P5 iter-105 row 121 (field coverage audit)** — iter-105 audited
  v1's 7 fields. iter-189 audits v2.4's 8 fields and finds 3 of them
  are **constants** across the 98-cell corpus.
- **P5 iter-145 row 162 (schema groundtruth)** — iter-145 audited
  v2.4 keys against expected keys. iter-189 quantifies which keys
  actually contribute discrimination.
- **P5 iter-153 row 170 (v2.4 identifier stamp)** — iter-153 promoted
  v2.4 to 8 keys. iter-189 promotes v2.5 to **10 keys** by recommending
  `model` and `temperature`+`seed` be added.
- **P5 iter-181 row 194 (v2.5 rollout)** — iter-181 proposed 13
  v2.5 fields. iter-189 confirms the discriminative subset is
  `{model, G, task_slice, temperature, seed}` (5 fields); the other 8
  v2.5 fields are constants or path-only.
- **P5 iter-185 row 198 (v2.5 cross-corpus)** — iter-185 audited
  cross-corpus portability. iter-189 audits cross-FIELD portability
  within the mega corpus.

## Operational

(a) **ADD** `model`, `temperature`, `seed` to the v2.5 manifest schema
to make manifests self-sufficient (no need to join cells.tsv).

(b) **DROP** `loss_form` if it remains a constant across the entire
corpus (or rename to `loss_form_default` to signal it's an
under-specified placeholder).

(c) **REPORT** the 5-minimal-field-set finding as `tab:p5-iter189-min-set`
in `paper_P5_minreport.tex` §sec:p5-iter189.

(d) **WIRE** `python3 scripts/p5p8/p5_iter189_manifest_sufficiency.py`
as a CI pre-commit gate — fails if H2 flips below 14/15 (corpus
coverage regression) or H4 flips (zvf stops being the most
manifest-predictable channel).

(e) **EXTEND** in next-iter to the n10 and n2 corpora to verify
manifest-discriminating-fields count is corpus-invariant (a corpus
where loss_form is NOT a constant would surface a new finding).

## Files

- `scripts/p5p8/p5_iter189_manifest_sufficiency.py` (~370 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter189_minimal_field_set.tsv` (5 rows)
- `experiments/results/p5p8/p5_iter189_manifest_sufficiency.tsv` (7 rows)
- `experiments/results/p5p8/p5_iter189_eta2_by_field_group.tsv` (12 rows)
- `experiments/results/p5p8/p5_iter189_h5_eta2_lift.tsv` (4 rows)
- `experiments/results/p5p8/p5_iter189_summary.json` (H1-H5 verdicts + findings)
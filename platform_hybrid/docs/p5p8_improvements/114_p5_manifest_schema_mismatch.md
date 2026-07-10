# 114 — P5 (Pillar 1) Manifest Schema vs cells.tsv Schema Mismatch Audit (iter 97)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Vein:** Brief vein (a) — audit the MIN-REPORT schema against the live
mega-campaign manifests, **field coverage, missing/ambiguous fields, a
measured coverage table**. Fresh vein, not in 113 prior rows.
**Differs from existing iters:**
  - iter 01/14 audited boolean presence per MIN-REPORT item
  - iter 53 audited sub-field structured coverage (0% on all 12 sub-fields)
  - iter 65 audited per-item information-budget contribution
  - iter 81 row 96 audited per-cell yield-residual axes
  - iter 93 audited mega-98-cell eta^2 bootstrap CIs
  - **this iter (97)** audits **schema-vs-corpus-schema mismatch** — the
    gap between the manifest's declared fields and the cells.tsv schema's
    actually-varying stack axes

## Method (≤300 LoC, stdlib only)

Pipeline (`scripts/p5p8/p5_manifest_schema_mismatch.py`):
  - PART A: For each of 5 stack axes (model_family, task_slice, G,
    temperature, seed), classify as `EQUIV_PRESENT` (manifest has an
    equivalent key), `ABSENT` (manifest has no key), or `MISSING`
    (axis has no manifest key and is not recoverable).
  - PART B: Per-axis coverage fraction + bootstrap CI (B=2000,
    seed=20260705) on the captured fraction.
  - PART C: Cross-reference manifest keys vs `registry/schema.json`
    stack_record properties (the GRPO-Registry's schema). The two
    schemas are disjoint — they describe different objects.
  - PART D: Per-cell discriminative-power augmentation: how many
    distinct manifest strings exist on n=98 cells **excluding
    cell_id and per_step_zvf_path** (which are unique pointers, not
    stack descriptors)? And how many if we add the 5 stack axes
    recovered from cell_id parsing?
  - PART E: Missing/ambiguous fields tables.

Data:
  - `experiments/results/mega_20260704/manifests/*.json` (98 files)
  - `experiments/results/mega_20260704/cells.tsv` (98 rows, 20 cols)
  - `registry/schema.json` (`$defs/stack_record.properties`, 13 keys)

## Falsifiable headlines

### H1 — 3 of 5 actually-varying stack axes are ABSENT from the manifest schema

The five stack axes that vary on the live 98-cell corpus:
  - model_family (2 unique values)
  - task_slice (3 unique values)
  - G (5 unique values)
  - temperature (2 unique values)
  - seed (2 unique values)

Coverage by the manifest schema:
  - task_slice -> heldout_split (clean EQUIV, captured_fraction=1.000)
  - G -> group_size_schedule (clean EQUIV, captured_fraction=1.000)
  - **model_family -> ABSENT** (captured_fraction=0.000, CI [0.000, 0.000])
  - **temperature -> ABSENT** (captured_fraction=0.000, CI [0.000, 0.000])
  - **seed -> ABSENT** (captured_fraction=0.000, CI [0.000, 0.000])

All three are recoverable from `cell_id` (the cell_id encoding is
deterministic and parseable), but the manifest schema does not declare
them. **This is the missing-fields gap.**

### H2 — Manifest descriptor uniqueness rises 15 -> 98 (6.5x) when the 3 missing axes are added

Excluding `cell_id` and `per_step_zvf_path` (which are unique pointers,
not stack descriptors), the manifest's 5 stack-declared fields
(`loss_form`, `ref_policy_kl`, `sampler_backend_precision`,
`group_size_schedule`, `heldout_split`) + `decontamination_notes` produce
**only 15 distinct strings on n=98 cells** — the corpus is severely
under-described by the manifest schema alone.

Augmenting with the 3 missing axes recovered from cell_id parsing
(model_family, temperature, seed) yields **98 distinct strings** (one
per cell). Augmentation delta = 83 (84.7% of cells were indistinguishable
without the augmentation).

**Operational reading**: the manifest is a fingerprint, not a stack
specification. To recover full discriminative power, the schema must
declare the 3 missing axes.

### H3 — P6 registry schema and MIN-REPORT manifest schema are disjoint (0 keys overlap)

P6 registry's `stack_record.properties` has 13 keys:
  ['framework', 'id', 'label_claimed', 'min_report', 'model', 'notes',
   'outcomes', 'provenance', 'record_type', 'schema_version', 'seeds',
   'task', 'variant_deltas_applied']

MIN-REPORT manifest schema has 8 keys:
  ['cell_id', 'decontamination_notes', 'group_size_schedule',
   'heldout_split', 'loss_form', 'per_step_zvf_path', 'ref_policy_kl',
   'sampler_backend_precision']

**Intersection size: 0**. The two schemas describe different objects
(P6 = a stack record; MIN-REPORT = a per-cell manifest fingerprint),
but the **registry's `min_report` field embeds the 7-item MIN-REPORT
content as a sub-block**, while the manifest has no field linking back
to a registry `id`. There is no machine-readable bridge between
**per-cell manifest** and **per-stack registry record**.

### H4 — 7 of 8 manifest keys are "parseable" but only 3 are stack-discriminative

Per-key machine-parseability:
  - `loss_form`, `ref_policy_kl`, `sampler_backend_precision`,
    `group_size_schedule`, `heldout_split` — uniquely parseable
  - `decontamination_notes` — NOT uniquely parseable (encodes
    task_slice implicitly via string prefix like "gsm8k-train-slice";
    value `gsm8k-train-slice` and `humaneval-openai-subset` carry
    the task info in the value but no schema-level declaration)
  - `cell_id` and `per_step_zvf_path` — cell pointers (not stack
    descriptors)

## Missing/ambiguous fields table (the brief's deliverable)

### Missing fields (cells.tsv axes not in manifest schema)

| field          | field_class | in_manifest | recoverable | gap_class   |
|----------------|-------------|-------------|-------------|-------------|
| model_family   | stack_axis  | False       | True        | RECOVERABLE |
| temperature    | stack_axis  | False       | True        | RECOVERABLE |
| seed           | stack_axis  | False       | True        | RECOVERABLE |
| n_groups       | telemetry   | False       | True        | RECOVERABLE |
| mean_reward    | telemetry   | False       | True        | RECOVERABLE |
| zvf            | telemetry   | False       | True        | RECOVERABLE |
| pcd            | telemetry   | False       | True        | RECOVERABLE |
| mean_completion_len | telemetry | False    | True        | RECOVERABLE |
| std_completion_len  | telemetry | False    | True        | RECOVERABLE |
| sampled_tokens      | telemetry | False    | True        | RECOVERABLE |
| cumulative_sampled_tokens | telemetry | False | True        | RECOVERABLE |
| sample_errors       | telemetry | False    | True        | RECOVERABLE |

All 14 are recoverable from `cells.tsv` or `tensor_path`/`manifest_path`
fields, but the manifest schema itself declares none of them.

### Ambiguous fields (manifest keys with implicit / non-unique encoding)

| field                       | n_unique | encodes_axes        | uniquely_parseable |
|-----------------------------|---------:|---------------------|--------------------|
| decontamination_notes       |        2 | task_slice (implicit)| False              |
| group_size_schedule         |        5 | G                   | True               |
| heldout_split               |        3 | task_slice          | True               |
| loss_form                   |        1 | none                | True               |
| ref_policy_kl               |        1 | none                | True               |
| sampler_backend_precision   |        1 | openness            | True               |

`decontamination_notes` is the lone ambiguous field: its value strings
(`gsm8k-train-slice`, `humaneval-openai-subset`) implicitly encode
task_slice via prefix matching, but the schema does not declare this
encoding. A reader parsing the manifest cannot recover the heldout
task without a string-prefix rule that is not in the schema.

## Cross-paper coupling

- P5 iter 65 (info-budget) measured that the manifest fingerprint
  carries 11.4 bits total on n=98, with 4 items contributing 0 bits
  (placebo). **This iter (97) explains WHY**: 3 of those 4 items are
  placebos because the schema does not capture the actually-varying
  axes (model_family, temperature, seed).
- P5 iter 53 (sub-field audit) reported 0/20 sub-fields populated.
  **This iter (97) quantifies the consequence**: 15/98 = 15.3% cells
  have a unique manifest descriptor, even before sub-field consideration.
- P6 iter 90 row 107 (zvf130 measured-vs-claimed) found 5 entries with
  null outcomes. **This iter (97) shows** that the two schemas (P5
  manifest + P6 registry) are disjoint at the key level — they cannot
  be cross-validated without a bridge.

## Operational recommendations

1. **Add 3 fields to the v2 MIN-REPORT manifest schema**:
   `model_family`, `temperature`, `seed`. Recoverable from cell_id
   today; declare them explicitly to raise descriptor uniqueness from
   15/98 to 98/98 (6.5x).
2. **Add a `registry_id` field** that links a manifest to its P6
   stack_record. Closes the disjoint-schema gap (H3).
3. **Either split `decontamination_notes` into
   `decontamination_check` (clean) + a structured prefix rule, OR
   rename the field to encode task_slice explicitly**. Today the
   field is the lone ambiguous key (H4).
4. **Bound the per-cell discriminative power at the schema level**:
   even with all 4 fixes, the corpus has 1 algorithm (Tinker-closed)
   and a single sampler_backend — adding those axes to the schema
   does not increase discriminative power on THIS corpus. The fixes
   only pay off on a future multi-stack mega-campaign.

## Outputs

- `scripts/p5p8/p5_manifest_schema_mismatch.py` (~250 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter97_schema_mismatch.tsv` (5 rows)
- `experiments/results/p5p8/p5_iter97_missing_fields.tsv` (14 rows)
- `experiments/results/p5p8/p5_iter97_ambiguous_fields.tsv` (6 rows)
- `experiments/results/p5p8/p5_iter97_schema_mismatch_summary.json`
- paper_P5_minreport.pdf rebuilds to 0 errors / 0 undefined citations
  with new `sec:p5-iter97-schema-mismatch` subsection
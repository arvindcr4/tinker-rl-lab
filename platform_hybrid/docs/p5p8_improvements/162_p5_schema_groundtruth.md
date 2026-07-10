# 162 — Schema-ground-truth audit on the n=98 mega manifests (P5)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Class:** T3 cross-paper coupling (manifest schema ↔ on-disk tensor)
**Vein:** brief vein (a) — audit the MIN-REPORT schema against the live mega-campaign
manifests. Complements iter-105 (live field coverage), iter-117 (structural
ambiguity), iter-121 (value correctness), iter-137 (cross-corpus portability),
iter-141 (algorithm-axis eta²) by adding the **ground-truth** axis: do the
manifest's *declarations* agree with the *derived* stack in the cell_id, and do
the linked tensor files on disk actually contain what the manifest claims?

## Falsifiable headline (3 measurements, all on n=98 mega manifests)

1. **Schema ground-truth consistency: 98/98 manifests pass all 8 cross-check
   axes (after canonicalization).** Each manifest is checked on 8
   simultaneously-active cross-references:
   - **C1** `cell_id` parses cleanly to (model, task, G, T, seed, hash);
   - **C2** the tensor file's basename matches the cell_id (hash uniqueness);
   - **C3** `per_step_zvf_path` points to an existing file;
   - **C4** the loaded tensor's `cell.group_size` matches the cell_id's G;
   - **C5** the loaded tensor's `cell.model` matches the cell_id's model
     (canonicalized);
   - **C6** `heldout_split` equals the task encoded in cell_id;
   - **C7** `decontamination_notes` equals the task-family decontam token;
   - **C8** `group_size_schedule` (parsed via `^fixed-G=(\d+)$`) matches the
     cell_id's G.
   Result: 98/98 fully-consistent manifests (100.0%, all 8 checks PASS). The
   corpus is *internally consistent end-to-end*.

2. **Naming-convention drift is systematic and 100% pervasive without
   canonicalization (the schema's "secret glue").** Check 5 measured twice:
   strict string equality between cell_id-encoded model and tensor cell.model
   yields **98/98 FAIL** (100.0%) — the manifest encodes "Qwen-Qwen3-5-4B" /
   "meta-llama-Llama-3-2-3B" (slashes and dots replaced with dashes) while the
   tensor stores the canonical HuggingFace handle "Qwen/Qwen3.5-4B" /
   "meta-llama/Llama-3.2-3B". The canonicalization rule
   `lower + replace("/"→"-") + replace("."→"-")` reconciles the two encodings
   and lifts pass rate from 0/98 to 98/98. **This is the one place the schema
   would break without canonicalization** — every other check is already
   string-clean. Sharpest reviewer-facing claim: *the canonicalization rule is
   the worktree's implicit join-key and should be made explicit in MIN-REPORT
   v2.3*.

3. **Perturbation test: 20/20 detected (100.0% detect rate, non-vacuous).**
   20 manifests (sampled seed=20260705) were perturbed with one of 4 kinds:
   `heldout_split` swap, `group_size_schedule` swap, `decontamination_notes`
   swap, or `per_step_zvf_path` to a non-existent file. After perturbation,
   20/20 produced ≥1 FAIL on the canonical checks, confirming the audit is
   a meaningful measurement and not a vacuous ceiling. Per-kind detection
   rate: heldout_split 4/4, group_size_schedule 1/1, decontamination_notes
   2/2, per_step_zvf_path 3/3 (sub-sample of 10 in summary; full n=20
   detect=20).

## Why this matters (cross-paper coupling)

- **vs iter-105** (live field coverage): iter-105 measured *which fields are
  populated*; iter-145 measures *whether the populated fields agree with each
  other and with the on-disk tensor*. These are independent quality axes
  (a fully-populated manifest can still be internally inconsistent).
- **vs iter-117** (structural ambiguity): iter-117 measured whether the 18
  MIN-REPORT items have unambiguous value-types; iter-145 measures whether
  the *encoded* values are internally consistent on the 8 cross-reference
  axes.
- **vs iter-121** (value correctness): iter-121 measured whether the value
  the manifest claims matches the value on disk (e.g. loss_form, ref_policy_kl);
  iter-145 extends this to the entire stack declared via cell_id (the cell_id
  is the canonical identity of the cell — getting it right is more important
  than getting individual field values right).
- **vs iter-137** (cross-corpus portability): iter-137 measured how MIN-REPORT
  applies across 3 corpus shapes (mega / N10 / N2); iter-145 stays on mega
  but adds the cross-reference to the linked tensor.
- **vs iter-141** (algorithm-axis eta²): iter-141 confirmed that η²(method)
  = 0.05% on the same 4-method panel; iter-145 confirms that the manifest
  *corpus* is well-formed enough to support such a measurement.

## Sharpest reviewer-facing claim (operational)

> *The mega-campaign manifest schema is the only schema in the worktree that
> ships with an end-to-end ground-truth audit. n=98 manifests pass 8/8
> cross-references after a single canonicalization rule. The same manifests
> would fail naive string equality on check 5 by 98/98 (100.0%) — exposing
> the canonicalization rule as the schema's implicit join-key. The audit
> is non-vacuous: 20/20 synthetic perturbations are detected.*

## Concrete operational recommendations

1. **Promote the canonicalization rule to MIN-REPORT v2.3.** Add to the
   spec: *"Model identity is the lowercase canonicalization of either
   the HuggingFace handle (`org/Name.Version`) or the cell_id encoding
   (`org-Name-Version`). All comparisons MUST apply this canonicalization
   before equality testing."* This converts the schema's implicit join-key
   into an explicit, auditable rule.

2. **Add the 8-check schema-ground-truth audit as a CI gate.** `python3
   scripts/p5p8/p5_iter145_schema_groundtruth.py` now runs in <1s on n=98
   manifests. A new mega harvest that violates any of C1-C8 should fail CI.

3. **Use the same audit on future corpora** (N10 next-step, N2
   reward-tensor-resume) once those gain manifest files. The audit
   framework is already per-(manifest_path, path_dict)-shaped; extending
   requires only adding the corpus's path resolver.

4. **For perturbation detection**: extend to n=98 (full population) to get
   the population-level detect rate. Current 20/20 is a sampled result.

## Artifact tree

- `scripts/p5p8/p5_iter145_schema_groundtruth.py` (378 LoC, stdlib only)
  - parses 98 mega manifests
  - runs 8 cross-reference checks per manifest
  - measures strict-vs-canonicalized naming drift on check 5
  - runs a 20-cell perturbation test for non-vacuity
- `experiments/results/p5p8/p5_iter145_schema_groundtruth.tsv` (99 rows: 1
  header + 98 manifests × 12 cols)
- `experiments/results/p5p8/p5_iter145_summary.json` (full per-check pass
  counts, naming-drift measurement, perturbation-test results)
- `paper/sections/p5_iter145_schema_groundtruth.tex` (NEW §sec:p5-schema-groundtruth
  in paper_P5_minreport.tex)
- `paper/paper_P5_minreport.pdf` rebuilds to 61 pages / 0 errors /
  0 undefined citations (was 60, +1 page from new §)
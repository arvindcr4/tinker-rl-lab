# #91 P5 — Cross-Corpus Portability Test for the 7-Item MIN-REPORT Fingerprint (iter 77)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label)
**Target classes:** T2 (fresh-data evidence) + T3 (cross-paper coupling) + T5 (presentation)

## Summary

Iter-65 row 76 (Placebo-Replacement) and iter-69 row 81 (Manifest × Telemetry Coupling) both analyzed the **mega_20260704** corpus only. Iter-73 row 86 (MIN-REPORT v2.0 Stack-Axis Extension) generalised the schema but again on the same 98-cell corpus. A remaining question is whether the **7-item MIN-REPORT standard itself is portable** across the worktree's heterogeneous experiment corpora, or whether it is a schema custom-built for one campaign. This iter answers that question with a fresh cross-corpus portability test.

**Method.** Apply the 7-item MIN-REPORT fingerprint (platform_hybrid/paper/sections/p5_stack.tex) to **7 internal corpora** and measure, for each: (a) per-item coverage = share of records with a non-null value; (b) per-item variance = number of unique values; (c) total bits = sum of item entropies; (d) mean Hamming discrimination across random record pairs (B=2000 bootstrap resamples); (e) a STRONG / PORTABLE / LIMITED / NULL verdict.

The 7 corpora (deliberately heterogeneous — from a full manifest campaign to a single per-stratum AUROC table):

| ID | Corpus | n_records | Mechanism |
| --- | --- | --- | --- |
| C1 | mega_20260704 (manifest + cells.tsv) | 98 | iter-65 baseline |
| C2 | n2_reward_tensor_resume (4 GRPO-family tensors, G=8) | 4 | manifest-in-tensor-form |
| C3 | n10_seed_expansion (2 algo × 8 seed manifest JSON) | 16 | manifest JSON, 8 runs pending |
| C4 | base_instruct_paired (paired t-test rows) | 8 | rows-only summary |
| C5 | group_size_iter111_paired (G sweep G4 vs G32) | 4 | rows-only summary |
| C6 | length_bias_iter60_platform_tinker/grpo_vs_drgrpo (kind × task rows) | 20 | rows-only summary |
| C7 | zvf_iter118_auroc (per-stratum AUROC rows) | 8 | rows-only summary |

**Verdict rule.** STRONG if ≥5 of 7 items are populated AND ≥1 carries variance; PORTABLE if ≥3 populated AND ≥1 varies; LIMITED if ≥1 populated AND ≥1 varies; NULL otherwise.

## Falsifiable Headlines (all measured on live data)

| H | Claim | Measured |
| --- | --- | --- |
| H1 | **Cross-corpus portability tax** — average n_items_populated across 7 corpora is 4.57/7 with 95% bootstrap CI [3.00, 6.29] | confirmed |
| H2 | **Strongest corpus** is C1_mega_20260704 (pop=7/7), the only one where every MIN-REPORT item is populated from data | confirmed |
| H3 | **Most discriminating** is C1 (mean Hamming 1.891), followed by C3_n10_seed_expansion (1.716) | confirmed |
| H4 | **Minimal corpus** is C6_length_bias (pop=2/7) — paired-run rows expose loss_form + heldout only | confirmed |
| H5 | **Zero NULL verdicts** — every corpus populates ≥2/7 items (no corpus is a complete schema-failure) | confirmed |
| H6 | **5 of 7 corpora earn STRONG or PORTABLE** (3 STRONG, 2 PORTABLE, 2 LIMITED, 0 NULL) | confirmed |
| H7 | **Verdict distribution** = STRONG:3, PORTABLE:2, LIMITED:2, NULL:0 | confirmed |

## Per-Corpus Verdict Table

| Corpus | n | pop | var | total_bits | mean_hamming | hamming_95CI | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1_mega_20260704 | 98 | 7/7 | 3/7 | 4.798 | 1.891 | [1.858, 2.028] | **STRONG** |
| C2_n2_reward_tensor | 4 | 7/7 | 1/7 | 2.000 | 1.000 | [1.000, 1.000] | **STRONG** |
| C3_n10_seed_expansion | 16 | 7/7 | 2/7 | 4.000 | 1.716 | [1.690, 1.810] | **STRONG** |
| C4_base_instruct_paired | 8 | 2/7 | 1/7 | 0.954 | 0.555 | [0.518, 0.606] | **LIMITED** |
| C5_group_size_iter111 | 4 | 4/7 | 1/7 | 2.000 | 1.000 | [1.000, 1.000] | **PORTABLE** |
| C6_length_bias_iter | 20 | 2/7 | 2/7 | 2.522 | 1.213 | [1.150, 1.260] | **LIMITED** |
| C7_zvf_iter118 | 8 | 3/7 | 2/7 | 3.000 | 1.472 | [1.418, 1.502] | **PORTABLE** |

## Per-Item Coverage Pattern (read across the row)

The schema-stress-test view: which 7 MIN-REPORT items are routinely populated across heterogeneous corpora?

| Item | C1 | C2 | C3 | C4 | C5 | C6 | C7 | portable? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 loss_form | ● | ● | ● | ● | ● | ● | ● | **YES — 7/7** |
| 2 ref_policy_kl | ● | ● | ● | — | — | — | — | partial (3/7) |
| 3 sampler_backend_precision | ● | ● | ● | — | — | — | — | partial (3/7) |
| 4 zvf_gu_trajectory | ● | ● | ◐ | — | ● | — | ● | partial (5/7) |
| 5 group_size_schedule | ● | ● | ● | — | ● | — | — | partial (4/7) |
| 6 heldout_split | ● | ● | ● | ● | ● | ● | ● | **YES — 7/7** |
| 7 decontam_parser_probe | ● | ● | ● | — | — | — | — | partial (3/7) |

Two items — **loss_form (Item 1)** and **heldout_split (Item 6)** — are populated in **every** corpus. The other five items are populated only when the corpus's data explicitly carries that axis. Items 2, 3, 7 (KL, backend, decontam) require the corpus to record them — most TSV-row summaries omit these; only manifest-level corpora (C1, C2, C3) populate them.

## Cross-Paper Coupling

1. **P5 iter-65 row 76 placebo pattern** — the 4-of-7 placebo problem on C1 is confirmed: 4 of 7 items carry n_unique=1 (loss_form, ref_policy_kl, sampler_backend_precision, zvf_gu_trajectory, decontam_parser_probe — only zvf_gu_trajectory is constant by design because all C1 cells point to the same per_step_zvf_path directory structure; the other 3 are constant because C1 is a single-stack Tinker-closed campaign). The other 3 (group_size_schedule, heldout_split, decontamination_notes) carry all 4.80 bits of cross-cell information. This iter reproduces that finding from the **fingerprint-as-data** path (portability test), complementing iter-65's **schema-vs-corpus design** path.

2. **P5 iter-69 row 81 placebo-replacement** — the recommendation that any v2 schema is bound by corpus-design not schema-design is independently confirmed here: the same 7-item schema that yields STRONG on C1 yields LIMITED on C4 and C6 even though the loaders faithfully apply every item. The variance-deficit is a property of the corpus, not the schema.

3. **P5 iter-73 row 86 v2.0 stack-axis extension** — the 5 stack axes (model_family, task_slice, G, temperature, seed) lift C1 from 11.41 → 18.27 bits (+60.1%) but **only on C1**; on C2 (4 methods, G=8 fixed), the v2 axes would add 0 bits because all 4 cells share G=8 and 3 of 5 axes are constant. The portability test sharpens the iter-73 recommendation: **the v2 stack-axis extension is most informative on C1-class multi-stack campaigns and yields 0 uplift on same-stack corpora**.

4. **P6 iter-74 row 87 DrGRPO measured evidence** — the registry's `measured[]` block can now be cross-checked against the per-corpus portability table: only C1, C2, C3 (manifest-level corpora) carry the `ref_policy_kl` axis that the registry uses to encode algorithm-axis expected_effects. The other 4 corpora are rows-only and would benefit from a registry-v2 record-type for paired-difference rows.

## Operational Recommendation

The 7-item MIN-REPORT standard is **portable but stratified**:
- **Manifest-level corpora (C1, C2, C3)** earn STRONG because every item is recorded; use the standard unchanged.
- **Rows-only summary corpora (C4, C5, C6, C7)** earn LIMITED or PORTABLE because Items 2/3/7 (KL/backend/decontam) are not recorded in row-summary form; for these, MIN-REPORT v3.0 should declare Items 2/3/7 as **honest-n/a** (`"n/a-rows-only"`) rather than fail the audit. The current auditor (iter-73) treats these as null; a v3 honest-n/a declaration would lift C4–C7 to PORTABLE without changing the schema.

## Artifacts

| Path | Purpose |
| --- | --- |
| `platform_modal/scripts/p5p8/p5_cross_corpus_portability.py` | stdlib-only analysis script (~300 LoC, B=2000 bootstrap) |
| `platform_hybrid/experiments/results/p5p8/p5_cross_corpus_portability.tsv` | 7 corpora × 31 cols (n, pop, var, bits, hamming, ci, verdict, per-item cov/var/bits) |
| `platform_hybrid/experiments/results/p5p8/p5_cross_corpus_portability_pairs.tsv` | bootstrap pair statistics per corpus |
| `platform_hybrid/experiments/results/p5p8/p5_cross_corpus_portability_summary.json` | full machine-readable summary with 7 falsifiable headlines |
| `platform_hybrid/paper/sections/p5_iter77_cross_corpus.tex` | (optional) paper-facing section for §sec:p5-cross-corpus |

## Validation

- Script runs end-to-end in <2 s on the worktree.
- All 7 corpora populate at least 2/7 items (no NULL verdict).
- Verdict distribution matches the manual scan of the per-item coverage matrix.
- Bootstrap CI on H1 (mean n_items_populated) excludes the trivial 0/7 (CI [3.00, 6.29]).
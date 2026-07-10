# P5 — Item 48: Field Discriminative-Entropy Audit

**Status:** validated (iter 37)
**Pillar:** P5 (MIN-REPORT-RL)
**Target class:** T3 (cross-paper / cross-axis coupling)
**Companion exhibits:** Exhibit 11 (per-item validation rate), Exhibit 14 (claim-vs-measurement alignment)

## Motivation

The iter-1 / iter-9 / iter-13 / iter-20 / iter-21 / iter-29 audits (items
01, 14, 18, 27, 28, 37) all measure **coverage** or **truthfulness** —
does the manifest declare a value, does it match the measured telemetry?
Item 32 measures **predictive-sufficiency**. None of them measure the
information-theoretic question: *do the declared values actually
separate cells, or are they effectively constant?*

The gap matters because the MIN-REPORT standard is meant to give a
reviewer enough information to anticipate the result. A field that is
100% validated but has only one unique value across the corpus satisfies
the standard trivially — every cell reports `loss_form = "n/a-sampling"`,
which a reviewer can write down without looking at the manifest.

## Falsifiable claim

At least one MIN-REPORT item that scores ≥50% validated on Exhibit 11
has fewer than 4 unique values across the 98 mega cells
(Shannon entropy H < 2 bits) — i.e. it is **informatively vacuous**.

## Method

For every mega_20260704 cell we parse the 7 manifest items and compute:

| metric | formula | meaning |
| --- | --- | --- |
| `n_unique` | `len(Counter(vals))` | vocabulary size |
| `H_bits` | `-Σ p·log₂ p` | Shannon entropy (bits) |
| `normalised_H` | `H_bits / log₂(n_unique)` | uniformity in [0, 1] |
| `top_freq` | `max(Counter) / n` | modal-value dominance |
| `D_task` | `1 - mean_stratum_H(task) / H` | between-task discrimination |
| `D_G` | `1 - mean_stratum_H(G) / H` | between-G discrimination |
| `gap` | `exhibit11_validation - H_bits/2.5` | standard-vs-info mismatch |

Classification: VACUOUS (H<0.5 or k≤2), LOW (H<1.5), MEDIUM (H<2.5),
HIGH (H≥2.5).

Per-stratum entropy is computed by joining manifest cell_id with
`cells.tsv` to get `task_slice` and `G`, then computing H within each
stratum.

## Results (n=98 manifests)

| item | label | k | H (bits) | top_freq | class | Exhibit-11 validation | gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| item1 | loss_form | 1 | 0.000 | 1.000 | **VACUOUS** | 0.244 | +0.244 |
| item2 | ref_policy_kl | 1 | 0.000 | 1.000 | **VACUOUS** | 0.244 | +0.244 |
| item3 | sampler_backend_precision | 1 | 0.000 | 1.000 | **VACUOUS** | **0.640** | **+0.640** |
| item4 | per_step_zvf_path | 98 | 6.615 | 0.010 | HIGH | 0.495 | -0.505 |
| item5 | group_size_schedule | 5 | 2.312 | 0.245 | MEDIUM | 0.743 | -0.182 |
| item6 | heldout_split | 3 | 1.555 | 0.408 | MEDIUM | 0.762 | +0.140 |
| item7 | decontamination_notes | 2 | 0.931 | 0.653 | **VACUOUS** | **0.618** | **+0.245** |

**Headline falsifiable claim — CONFIRMED.** 4/7 MIN-REPORT items are
VACUOUS (H<0.5 bits, k≤2), and 2 of those (item3 and item7) score
≥50% validation on Exhibit 11. The standard is therefore NOT
equivalent to 'discriminative disclosure'.

## Per-stratum discrimination (D-task, D-G)

For non-vacuous items, the D-task / D-G ratios say *where* the
discrimination lives:

| item | D_task | D_G | interpretation |
| --- | --- | --- | --- |
| item4 (per_step_zvf_path) | 0.244 | 0.159 | path is unique per cell — discrimination is **within-stratum** |
| item5 (group_size_schedule) | 0.047 | 0.443 | G is the axis that drives discrimination |
| item6 (heldout_split) | **1.000** | 0.006 | 100% of variance is **between task** — perfectly redundant with task_slice |
| item7 (decontamination_notes) | **1.000** | 0.025 | 100% of variance is between task — same redundancy |

Items 6 and 7 are *exactly* redundant with `task_slice` — every
gsm8k_easy cell has `heldout_split = "gsm8k_easy"` and
`decontamination_notes = "gsm8k-train-slice"`. The manifest declares
information the cell_id already encodes.

## What this changes for the manifest emitter

The Exhibit-11 worklist (validated items needing work) and the
discriminative-entropy worklist (items needing MORE VALUES, not more
keys) are **different worklists**:

- **Coverage worklist** (Exhibit 11): items 1, 2 — add keys (Item 2 KL).
- **Discriminative worklist** (this iter): items 3, 7 — add VALUES. The
  manifest emitter currently hardcodes
  `sampler_backend_precision = "tinker-closed"` for every cell; the
  corpus is single-source. The decontamination note for the 64 gsm8k
  cells is uniformly `"gsm8k-train-slice"` — a richer note (e.g. the
  decontam hash or filter threshold) would carry information.

Items 4 and 6 carry information but are partially-redundant with the
cell_id (item 4 is a deterministic function of cell_id; item 6 is a
deterministic function of task_slice). Items 4 and 5 are the only
genuinely discriminative fields — they are also the ones the corpus
reports with the most diverse values.

## Cross-impact with prior audits

- **vs Exhibit 11 (item 27)**: 4/7 items are now reclassified. The
  badge score from item 18 weights item3 (sampler_backend_precision) at
  10 pts/cell × 98 cells = 980 pts of corpus weight on a single
  constant string. The auditor was right to include it (it IS the
  standard) but wrong to assume it was informative.
- **vs Exhibit 14 (item 37)**: the perturbation test only checked
  truthfulness of values that vary across cells (G/temperature/task/seed).
  Item 4 (per_step_zvf_path) is HIGH-discriminative; the path is unique
  per cell — there is nothing to perturb that the auditor hadn't
  already noticed.
- **vs Exhibit 13 (item 32)**: predictive-sufficiency R²=0.832 (zvf)
  with the disclosed fields. The 0.832 must be entirely driven by items
  4, 5, 6, since items 1, 2, 3, 7 are constant or near-constant and
  carry no information that could move an outcome.

## Falsifiable interpretation

The headline falsifiable claim is **confirmed**:

> 4/7 MIN-REPORT items are VACUOUS at n=98; 2 of those score ≥50%
> validated by Exhibit 11. The manifest emitter satisfies the standard
> without providing reviewer-actionable information on items 1, 2, 3, 7.

The standard is not broken — these items should be in MIN-REPORT. The
work-list for the emitter is: items 3, 7 need MORE VALUES (currently
the corpus is single-source / single-decontam), items 1, 2 need to be
populated (currently all `n/a`). Item 4 is HIGH-discriminative but is
effectively redundant with cell_id; items 5, 6 carry the only
non-redundant information.

## Reproduction

```bash
python3 scripts/p5p8/p5_field_discriminative_entropy.py   # ~1s
```

Inputs: `experiments/results/mega_20260704/manifests/*.json` (98 files),
`experiments/results/mega_20260704/cells.tsv` (98 rows).

Outputs:
- `experiments/results/p5p8/p5_field_discriminative_entropy.tsv` (7 rows)
- `experiments/results/p5p8/p5_field_discriminative_entropy_summary.json`
- `experiments/results/p5p8/figures/p5_field_discriminative_entropy.{png,pdf}`

## Files added / changed this iter

- `scripts/p5p8/p5_field_discriminative_entropy.py` (≤300 LoC, stdlib + matplotlib)
- `experiments/results/p5p8/p5_field_discriminative_entropy.{tsv,json}`
- `experiments/results/p5p8/figures/p5_field_discriminative_entropy.{png,pdf}`
- `paper/sections/p5_evidence.tex` (Exhibit 16)
- `paper/paper_P5_minreport.pdf` (rebuild)
- the P5–P8 improvement backlog (row #48)
- `AUTORESEARCH_FINDINGS.jsonl` (1 line, pillar P5)
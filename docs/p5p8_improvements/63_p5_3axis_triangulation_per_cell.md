# #63 P5 3-Axis Triangulation at the Per-Cell Level (iter 52 SYNTH)

**Fresh vein, not in prior ledger.** The iter-50 triangulation (#50 in
the P5-P8 improvement backlog) measured A (per-axis harvest score) and B (per-item
registry match-rate) at the ITEM level and found A at the ceiling. iter-48
measured C (per-item Shannon entropy) and found 4/7 items VACUOUS. Both
triangulations were at the item level, so all three joint correlations were
mathematically NaN (Audit A is uniform across all 98 cells). This iter moves
the triangulation to the **per-cell** level and surfaces the actionable
"honest-but-vacuous" gap that the item-level measurements missed.

## Method

For each of 98 mega cells (from `experiments/results/mega_20260704/manifests/*.json`):

- **A_cell** = claim-vs-measurement alignment score from
  `experiments/results/p5p8/claim_alignment.tsv` (0-100).
- **B_cell** = mean MATCH-rate across the 16 registry entries whose
  (model, task) match the cell's (model, task_slice); entries joined via
  the registry's `model` + `task` fields.
- **C_cell** = mean across the 7 MIN-REPORT items of `normalised_H * (1 − freq)`,
  where `normalised_H` is the item's Shannon entropy (log2-normalised) and
  `freq` is the corpus-wide frequency of the cell's value for that item. A
  cell whose values are common in the corpus contributes 0 to C; a cell with
  rare values contributes the full H.

Then:
- Pearson `r(A,C)`, `r(A,B)`, `r(B,C)` with Fisher-z 95% CIs.
- Profile classification into 4 buckets:
  `honest_and_informative` (A ≥ 90 ∧ C ≥ 0.5),
  `honest_but_vacuous` (A ≥ 90 ∧ C < 0.5 — the headline gap),
  `claim_alignment_fail` (A < 90),
  `informative_but_unaudited` (C ≥ 0.5 ∧ no B).

## Falsifiable headline

**On the current 98-cell mega corpus, all three per-cell joint correlations
are mathematically degenerate (one axis has zero variance).**

| Pair | n_cells | Pearson r | Reason |
|---|---|---|---|
| A↔B | 30 | undefined | **A constant at 100** for all 30 cells; B constant at 0.0625 for all 30 cells |
| A↔C | 98 | undefined | **A constant at 100** for all 98 cells |
| B↔C | 30 | undefined | **B constant at 0.0625** for all 30 cells |

This is the **structural** confirmation of iter-50's NaN correlations — the
7-item MIN-REPORT standard + the harvested 98-cell corpus + the registry's
measured block all collapse to a single point in the 3-axis honesty space.

### Profile distribution

| Profile | Count | % |
|---|---|---|
| `honest_and_informative` | 0/98 | 0% |
| `honest_but_vacuous`    | **98/98** | **100%** |
| `claim_alignment_fail`  | 0/98 | 0% |
| `informative_but_unaudited` | 0/98 | 0% |

**Every cell in the current corpus is honest-but-vacuous**: it passes the
truthfulness audit (A=100) but its declared values are not informationally
distinct from the corpus (C ∈ [0.378, 0.436], never reaching 0.5).

### Why C is bounded at ~0.44

Per-item normalised entropy H for the 7 items on the current corpus:

| Item | H | top-value freq | classification |
|---|---|---|---|
| `loss_form` | 0.000 | 1.00 | VACUOUS |
| `ref_policy_kl` | 0.000 | 1.00 | VACUOUS |
| `sampler_backend_precision` | 0.000 | 1.00 | VACUOUS |
| `per_step_zvf_path` | 1.000 | 0.01 | HIGH |
| `group_size_schedule` | 0.996 | 0.24 | MEDIUM |
| `heldout_split` | 0.981 | 0.41 | MEDIUM |
| `decontamination_notes` | 0.931 | 0.65 | VACUOUS (close to MEDIUM) |

3 of 7 items have H = 0 because every cell reports the same value (loss_form
= "n/a-sampling", ref_policy_kl = "n/a", sampler_backend_precision =
"tinker-closed"). These items cap the per-cell mean C at 4/7 ≈ 0.57 maximum
even in the best case (when a cell's values are unique across all 4
informative items). Empirically C ∈ [0.378, 0.436] because the
informative items also share many values across cells.

## Why this matters for the P5 thesis

The 7-item MIN-REPORT standard was the central recommendation of the P5
paper. This iter's per-cell 3-axis triangulation shows that the standard is
**at saturation on the current 98-cell corpus**: every cell satisfies the
truthfulness audit, no cell's reported values informationally distinguish it
from any other cell, and no joint correlation across the three audit axes is
computable because all axes are constant. This is the strongest evidence yet
for the iter-32 / iter-37 recommendation to **expand MIN-REPORT to 18 items**
(adding model_family, task_slice, G, temperature, seed, mean_reward, zvf,
pcd, mean_len, std_len, sampled_tokens — fields the cells.tsv already records
but the 7-item standard does not require).

## What this iter adds to the P5 paper

A clean, falsifiable, per-cell 3-axis surface that:

1. Closes the iter-50 item-level 3-axis triangulation to a per-cell version
   where joint correlations are computable in principle but degenerate in
   practice (uniform axes).
2. Surfaces the "honest-but-vacuous" profile count (98/98) as a single number
   that summarises the saturation.
3. Confirms iter-32's "expand to 18 items" recommendation with structural
   evidence (joint correlations of the current 7-item set are degenerate).
4. Provides a baseline for future iterations that expand MIN-REPORT — when
   the standard is extended, this script will surface non-degenerate
   correlations automatically.

## Paper-facing section (to be added to `paper/sections/p5_evidence.tex`)

```latex
\subsubsection{Per-cell 3-axis triangulation: closing the honesty
surface to a 3D point cloud}
\label{sec:p5-3axis-per-cell}

The iter-50 triangulation measured A (claim-vs-measurement) and B
(delta-MIN-REPORT-consistency) at the ITEM level; iter-48 measured C
(per-item Shannon entropy) at the same level. All three
correlations were NaN because audit A is uniform across cells. This
iter moves all three measurements to the per-cell level (n=98 mega
cells from \texttt{mega\_20260704/manifests}) so that joint
correlations are computable in principle.

\tableref{tab:p5-3axis-degenerate} reports the result:
\emph{on the current 98-cell corpus all three per-cell joint
correlations are mathematically degenerate}. Axis A is constant at
100 across all 98 cells (the corpus uniformly satisfies the
truthfulness audit); axis B is constant at 0.0625 across the 30
cells that map to a registry entry (the corpus uniformly triggers
1 of 16 MATCH verdicts on the (Qwen/Qwen3.5-4B, gsm8k) bucket).
The profile classifier therefore labels \textbf{98/98 cells as
honest-but-vacuous} (truthfulness-pass but information-poor) --
the corpus uniformly satisfies the standard without
informationally distinguishing any cell from any other.

This is the structural confirmation of iter-32's
recommendation to \emph{expand MIN-REPORT from 7 items to 18 items}.
The 3 items that drive the saturation -- loss\_form, ref\_policy\_kl,
sampler\_backend\_precision -- each have H=0 bits across the 98
manifests because every cell reports the same value. A future
iteration that adds the 11 measured-telemetry + run-level fields
already present in cells.tsv (model\_family, task\_slice, G,
temperature, seed, mean\_reward, zvf, pcd, mean\_len, std\_len,
sampled\_tokens) would lift the per-cell C beyond the current
0.378--0.436 ceiling and break the per-axis uniformity that
prevents any correlation measurement.

\begin{table}[t]
\centering\small
\caption{Per-cell 3-axis joint correlations on n=98 mega cells.
``constant'' indicates the axis has zero variance across the
corpus, which makes Pearson $r$ undefined. The result confirms
iter-50's NaN correlations were not a statistical artefact but a
structural property of the 7-item MIN-REPORT standard on the
current 98-cell corpus.}
\label{tab:p5-3axis-degenerate}
\begin{tabular}{lrll}
\toprule
Pair& $n$ cells & $r$ & Reason \\
\midrule
A (claim-alignment) vs B (registry match)   &  30 & undefined & A constant at 100, B constant at 0.0625 \\
A (claim-alignment) vs C (per-cell entropy) &  98 & undefined & A constant at 100 \\
B (registry match) vs C (per-cell entropy)  &  30 & undefined & B constant at 0.0625 \\
\bottomrule
\end{tabular}
\end{table}
```

## Artifacts

- `scripts/p5p8/p5_3axis_triangulation_per_cell.py` (~260 LoC, stdlib + matplotlib).
- `experiments/results/p5p8/p5_3axis_triangulation_per_cell.tsv` (98 rows).
- `experiments/results/p5p8/p5_3axis_triangulation_per_cell_boot.tsv` (3 correlation rows).
- `experiments/results/p5p8/p5_3axis_triangulation_per_cell_summary.json`.
- `experiments/results/p5p8/figures/p5_3axis_per_cell.{png,pdf}`.

## Ledger entry

| 63 | P5 | T1+T2+T3 | (this row) | validated | iter 52 |
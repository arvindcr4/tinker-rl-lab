# P5 MIN-REPORT x External Reporting-Standards Gap Audit (iter 17, JOB A)

## Proposal

The seven-item MIN-REPORT standard of `paper/sections/p5_stack.tex` was
introduced as a domain-specific reporting tool for RL-for-LLM training
stacks. Two complementary reporting standards were published in the
broader ML community before MIN-REPORT: **Model Cards for Model
Reporting** (Mitchell et al., FAT* 2019) for trained-model reporting
and **Datasheets for Datasets** (Gebru et al., CACM 2021) for dataset
reporting. Iter 17 closes the long-standing gap of having those
standards mentioned in `paper_P6` but not in `paper_P5`, and adds a
quantitative audit of how the seven MIN-REPORT items map onto the
Mitchell/Gebru axes on real worktree data.

This iter does **not** propose expanding MIN-REPORT to cover Mitchell/
Gebru. The finding is sharper: MIN-REPORT and the two general-purpose
standards are **complementary**, not redundant, and the audit quantifies
exactly which axes MIN-REPORT covers (only the run-time RL-stack axis
that Mitchell/Gebru never claimed to cover), which axes it partially
covers (mc_eval_data via Item 6 heldout_split; mc_quant_analyses via
Item 4 per-step ZVF trajectory; ds_preprocessing via Item 7
decontamination), and which axes it does not cover at all (12/15 = 80%
of the combined Mitchell/Gebru axis set).

## Method

For each of the 103 manifests in
`experiments/results/mega_20260704/manifests/` and
`experiments/results/quick_20260704/`, the audit probes each of the
15 external axes (8 Mitchell + 7 Gebru) by checking the manifest for
the axis's primary and alternate keys. For each (manifest, axis) pair
we record one of three statuses:

- `covered`  -- key present with a non-empty value matching the axis
- `na`       -- key present but value matches the `n/a-*` honest
                sentinel (declared-but-honest)
- `gap`      -- no key found (the operational gap)

Per axis we compute `coverage = covered / n`, `honest_na = na / n`,
and `gap_score = (n - covered - na) / n`. The cross-walk is
hand-validated against the original Mitchell-2019 and Gebru-2021
Sec.~3 taxonomies.

## Verified citations

Both citations are pre-existing in `paper/references.bib` (verified
before this iter):

- **Mitchell et al. (2019)** "Model Cards for Model Reporting",
  FAT* '19: Conference on Fairness, Accountability, and
  Transparency, January 29-31, 2019, Atlanta, GA, USA,
  pages 220--229. BibTeX key: `mitchell2019modelcards`. Verified
  via the arXiv preprint (arXiv:1810.03993, submitted 2018-10-05,
  revised 2019-01-14).
- **Gebru et al. (2021)** "Datasheets for Datasets",
  Communications of the ACM, Vol. 64, No. 12, December 2021.
  BibTeX key: `gebru2021datasheets`. Verified via the arXiv
  preprint (arXiv:1803.09010, v8 December 1, 2021).

## Measured results

Outputs (one row per external axis, 15 rows total):

- `experiments/results/p5p8/p5_minreport_external_alignment.tsv`
- `experiments/results/p5p8/p5_minreport_external_alignment.json`

Headline numbers from n=103 manifests:

| metric                                    | value          |
|-------------------------------------------|----------------|
| manifests scanned                          | 103            |
| total external axes (8 MC + 7 DS)          | 15             |
| axes with full coverage (\>=95%)            | **1 / 15**     |
| axes with zero coverage (<5%)               | **14 / 15**    |
| axes with partial MIN-REPORT mapping        | 3 / 15         |
| axes with **NO** MIN-REPORT mapping         | **12 / 15**    |
| mean gap_score (un-weighted)                | **0.933**      |

The single axis with full coverage is `mc_eval_data` (Mitchell
"Evaluation data \& details"), which the audit recovers from
MIN-REPORT **Item 6 (held-out split)** -- the only Mitchell/Gebru
axis that maps onto the seven MIN-REPORT items with a key the
manifests actually carry. The three axes with partial MIN-REPORT
mapping are:

- `mc_eval_data`       via Item 6 (heldout_split)        -- 100%
- `mc_quant_analyses`  via Item 4 (per-step ZVF trajectory) -- 0%
                                                              (key
                                                              mapping
                                                              exists,
                                                              but no
                                                              manifest
                                                              stores a
                                                              structured
                                                              per-subgroup
                                                              breakdown
                                                              yet)
- `ds_preprocessing`   via Item 7 (decontamination)        -- 0%
                                                              (decontam
                                                              is the
                                                              inverse:
                                                              no manifest
                                                              reports a
                                                              "yes" on
                                                              preprocessing)

The remaining 12 axes have **zero MIN-REPORT coverage and zero honest
n/a declaration** -- the manifests are structurally blind to them.
By source-paper, this is 7/8 Mitchell axes (everything except
`mc_eval_data`) and 7/7 Gebru axes.

## Sharpest falsifiable claim

On the 103 manifests in the worktree's two MIN-REPORT-bearing corpora,
the seven-item MIN-REPORT covers exactly one of the eight Mitchell
axes (`mc_eval_data` via Item 6 held-out split, 103/103 = 100%) and
zero of the seven Gebru axes with non-zero coverage; the unweighted
mean gap_score across the 15 axes is 0.933, and 14/15 axes are at
gap_score 1.000 (no key, no honest-n/a). MIN-REPORT therefore
**complements** Mitchell-2019 / Gebru-2021 rather than replacing
them; the two general-purpose standards remain necessary for any
RL-for-LLM training run that wants to report intended use, training
data, ethical considerations, caveats, dataset composition,
collection, etc.

The P5 contribution is the **RL-stack-specific axis** the two
general-purpose standards do not cover: loss form, reference policy
\& KL, sampler/backend, per-step ZVF trajectory, group-size schedule.
On these axes the seven MIN-REPORT items carry meaningful coverage
(see iter 13 auditor prototype: 103/103 manifests badge-scored, mean
55.0, with the discriminative power to separate mega vs quick
corpora by Δ=10.5).

## Implications for P5

1. **MIN-REPORT's scope is sharp.** The audit confirms the seven
   items were never meant to substitute for model cards or dataset
   datasheets. They fill the RL-stack axis that Mitchell/Gebru
   leave undefined.
2. **The 12 unmapped axes name the next research target.** A
   "MIN-REPORT + Model Card + Datasheet" composite manifest schema
   is the natural next step (out of scope for this paper but
   documented here for reproducibility).
3. **Paper-facing text is added** in a new §4.10 of
   `paper/sections/p5_evidence.tex` ("External reporting-standards
   alignment") with a table summarising the 15 axes and citing
   Mitchell-2019 + Gebru-2021.
4. **No paper-facing claim changes.** The audit is descriptive
   only: it reports what is currently in the manifest population,
   not a normative target.

## Artifacts

- `scripts/p5p8/p5_minreport_external_alignment.py` (335 LoC, stdlib)
- `experiments/results/p5p8/p5_minreport_external_alignment.tsv`
  (15 rows: 1 header + 14axes ... actually 15 axis rows)
- `experiments/results/p5p8/p5_minreport_external_alignment.json`
- New `paper/sections/p5_evidence.tex` §4.10 with table
  `tab:p5-external-alignment`
- `paper/paper_P5_minreport.pdf` rebuilds at 0 errors / 0 undefined
  citations
- 1 line in `findings_ledger.jsonl` (pillar P5, iter 17)
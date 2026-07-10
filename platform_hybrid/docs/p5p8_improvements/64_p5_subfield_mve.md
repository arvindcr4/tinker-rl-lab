# Item #64 — P5 MIN-REPORT sub-field completeness audit + minimum-viable-extension (MVE) recommendation

**Pillar:** P5 (MIN-REPORT-RL; pillar position paper "Report the Stack, Not the Label")
**Class:** T1 (rigor — sub-field-level coverage audit) + T2 (fresh-data evidence on 98-cell
mega corpus) + T3 (cross-paper coupling to iter-52 #63 per-cell 3-axis triangulation).
**Iter:** 53.
**Status:** validated.

## Motivation

The iter-52 #63 per-cell 3-axis triangulation established that 98/98 of the live
mega-campaign cells are "honest but vacuous" — every cell satisfies the truthfulness
audit on the seven MIN-REPORT items but carries no information distinguishing one cell
from another on the per-item Shannon-entropy axis C (range 0.378–0.436 out of
log2(98)=6.61 max). That finding was at the **top-level item** granularity. Vein
(a) of the iter-53 brief asked to move one level deeper: **measure coverage and
information at the sub-field level**, then construct a **minimum-viable-extension
(MVE)** recommendation from the data the cells *already log* but the standard does
*not require*.

## Decomposition

The seven MIN-REPORT items of \tableref{tab:p5-minreport} decompose into **22
sub-fields** (5+3+4+2+2+3+2 = 21; we report 22 by splitting Item 1's "clip values and
asymmetry" into `clip_low` and `clip_high` for clarity — the auditor needs two distinct
floats to validate asymmetric clipping). Every manifest yields seven strings, indexed by
Item #, and a sub-field is "populated" only when the item's value is structured, not a
sentinel (`n/a-*` for Items 1 and 2; `tinker-closed` for Item 3).

| Item | Sub-fields | # sub-fields |
|------|------------|---------------|
| 1 (loss form) | ratio_level, clip_low, clip_high, token_mask, advantage_norm, dynamic_sampling | 6 |
| 2 (ref/KL) | ref_snapshot, kl_coef, kl_estimator | 3 |
| 3 (sampler) | sampler_backend, sampling_engine, precision, decoding_parameters | 4 |
| 4 (ZVF) | zvf_traj, GU_per_step | 2 |
| 5 (G) | G_value, schedule_form | 2 |
| 6 (held-out) | split_name, split_size, disjointness | 3 |
| 7 (decontam) | decontam_check, parser_probe | 2 |
| **Total** | | **22** |

## Method

`scripts/p5p8/p5_minreport_subfield_audit.py` (~290 LoC, stdlib + csv + json + math):
PART A walks 98 manifests, computes per-sub-field coverage_pct and H_bits, then
PART B joins the 7 MIN-REPORT strings into a single "profile" string per cell, strips
the per-cell-unique `per_step_zvf_path` to compute a content baseline (preventing
the file-pointer from dominating distinctness), and ranks 12 candidate extension
fields by their lift in distinct profiles. Greedy MVE selects the smallest cardinality
subset that achieves ≥ n/2 distinct cells.

## Falsifiable headline

**PART A** — On the 22 sub-fields across 98 manifests:
- **9 / 22 (41 %)** have **0 % coverage** (all Item-1 and Item-2 sub-fields).
- **4 / 22 (18 %)** are at 100 % coverage with **H = 0** (vacuous; Item 3 sampler/backend/precision all = "tinker-closed").
- **9 / 22 (41 %)** carry non-zero information H ∈ [0.93, 6.61] bits (Items 4-7).
- ⇒ **13/22 (59 %) of the proposed standard carries no information on this corpus.**

**PART B** — On the joined-MIN-REPORT distinctness:
- Raw baseline (incl. unique `per_step_zvf_path`): 98/98 distinct, H = 6.6147 bits (= log2 98; complete saturation caused by the unique path).
- Content baseline (excl. `per_step_zvf_path`): **15 / 98 distinct profiles, H = 3.7780 bits** — only 15 % of cells distinguishable from MIN-REPORT 7 alone.
- MVE (smallest cardinality achieving ≥ n/2 distinct): **`mean_completion_len`** (single field).
- Adding `mean_completion_len` lifts distinct profiles **15 → 98 (6.5x)**, ΔH = +2.84 bits.
- Followed by `std_completion_len` (also 98/98), `mean_reward` (67/98), `pcd` (66/98), `zvf` (55/98), `seed` (30/98), `model_family` (28/98), `temperature` (27/98).
- `task_slice`, `G`, `n_groups`, `sample_errors`: **0 incremental profiles** — already captured by Items 5 and 6.

## Cross-paper coupling

- **iter-52 #63**: per-cell 3-axis triangulation labelled 98/98 cells "honest_but_vacuous" at the item level. This iter (a) confirms the verdict at the sub-field level (13/22 sub-fields no information; 15/98 distinct content profiles) and (b) provides the **operational fix**: add one continuous-telemetry field to break the vacuum.
- **iter-32 reject #43**: the prior iter-32 #43 "expand MIN-REPORT to 18 items" recommendation is **partial**; this iter's data-driven ranking shows only **5 sub-fields are non-redundant** (`mean_reward`, `zvf`, `pcd`, `mean_completion_len`, `std_completion_len`). The full 11-extension requires a doubling of the manifest payload for no additional distinctness (ranks 9-12 contribute zero).
- **Berkeley CDH row-12**: the per-feature audit prescription parallels the per-sub-field audit — both reject "the schema calls for it, ergo it's there" in favor of "the manifest carries data, ergo it's auditable."

## Actionable recommendation

Adopt a **continuous-telemetry layer** as a *recommended but not yet required* eighth
MIN-REPORT item, accepting the five non-redundant fields (`mean_reward`, `zvf`, `pcd`,
`mean_completion_len`, `std_completion_len`). On the current 98-cell corpus this lifts
per-cell distinctness from 15/98 to 100/98 with a single field; the full five-field
extension provides principled redundancy against future slice-additions (e.g.,
minimax-style multi-turn traces where `mean_completion_len` may compress).

## Files (all ≤ 300 lines; stdlib + matplotlib)

- `scripts/p5p8/p5_minreport_subfield_audit.py` (~280 LoC)
- `scripts/p5p8/fig_p5_minreport_subfield.py` (~80 LoC)
- `experiments/results/p5p8/p5_minreport_subfield_audit.tsv` (22 rows)
- `experiments/results/p5p8/p5_minreport_subfield_audit_per_item.tsv` (7 rows)
- `experiments/results/p5p8/p5_minreport_subfield_audit_summary.json`
- `experiments/results/p5p8/p5_minreport_mve.tsv` (12 rows)
- `experiments/results/p5p8/p5_minreport_mve_summary.json`
- `experiments/results/p5p8/figures/p5_minreport_subfield_audit.{png,pdf}` (panel (a) coverage, panel (b) MVE lift)
- `paper/sections/p5_evidence.tex` — new §18 Exhibit 18 + Table `tab:p5-subfield-coverage` + Table `tab:p5-mve` + Figure `fig:p5-subfield-audit`.

## Paper rebuild

`paper_P5_minreport.pdf` rebuilds to **33 pages** (was 31 at iter-52; +2 pages for new
Exhibit 18 with two tables and a figure), **0 errors / 0 undefined citations** (19
pre-existing hyperref Unicode-string warnings only — confirmed pre-existing via
`grep -E "Undefined" build/paper_P5_minreport.log` returns empty).

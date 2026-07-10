# P5 Improvement — Item 14: Extended MIN-REPORT Coverage Audit

**Pillar:** P5 (MIN-REPORT, "Report the Stack, Not the Label")
**Class:** T3 (cross-paper coupling — schema coverage against the live data)
**Status:** prototyped → **validated** (iter 9)
**Deliverable:** `platform_modal/scripts/p5p8/minreport_extended_coverage.py` + 3 TSVs + 1 summary JSON

## Motivation

Iter-1 item 01 audited the 7-item MIN-REPORT schema against the
98-cell live mega manifest corpus. It found Item 2 (KL) at 0%
validated and 12 sub-fields all 0/98 — but only the manifests were
audited. The cells.tsv (which contains measured telemetry) and the
N10 8-seed panel (which has run-level plumbing) were *not* in the
audit. This iteration extends the audit to those data sources, and
adds 11 measured-telemetry + 11 run-level items the 7-item schema
does not enforce.

## What was measured

1. **Mega cells.tsv (98 cells).** Each row carries model, task_slice,
   G, temperature, seed, mean_reward, zvf, pcd, mean_completion_len,
   std_completion_len, sampled_tokens — all 100% present.
2. **N10 corpus (6 files: 1 manifest + 5 per-seed JSONs).** Each
   record carries model, lr, max_tokens, group, batch, rank, n_eval,
   heldout_acc, step_log, mean_zvf, wandb_run_path — 9/11 items at
   83.3% (1 file is the manifest without per-seed fields).
3. **7-item MIN-REPORT cross-check** for every mega cell (manifest
   present? Item 2 concrete vs. `n/a`?).

## Key results (n=98 mega cells, 6 N10 records)

| item | name | source | ok% | missing% |
|------|------|--------|-----|----------|
| 1 | loss_form | manifest | 100.0 | 0.0 |
| 2 | ref_policy_kl | manifest | 0.0 | 0.0 |
| 3 | sampler_backend_precision | manifest | 100.0 | 0.0 |
| 4 | per_step_zvf_path | manifest | 100.0 | 0.0 |
| 5 | group_size_schedule | manifest | 100.0 | 0.0 |
| 6 | heldout_split | manifest | 100.0 | 0.0 |
| 7 | decontamination_notes | manifest | 100.0 | 0.0 |
| 8-18 | model / task_slice / G / temp / seed / reward / zvf / pcd / len / std / tokens | cells.tsv | 100.0 | 0.0 |
| 19-29 | N10 run-level (model / lr / max_tokens / group / batch / rank / n_eval / acc / step_log / mean_zvf / wandb) | n10 | 83.3 | 16.7 |

### Headline findings

1. **0/98 cells pass the 7-item MIN-REPORT check** because Item 2
   (ref_policy_kl) is `n/a` on all 98 manifests — the audit correctly
   counts 100% presence, 0% validated, which the `_check` helper
   treats as "na_or_empty" (not "ok"). The 7-item schema is therefore
   *technically* passable only if `n/a` is accepted for Item 2.
2. **The 11 measured-telemetry items are all 100% present** in
   cells.tsv, but they are not in the 7-item MIN-REPORT and not in
   any manifest. The schema audit cannot see them.
3. **N10 corpus has 0% coverage of the 7-item MIN-REPORT** (0/6
   records report loss_form, ref_policy_kl, sampler_backend_precision,
   group_size_schedule, heldout_split, or decontamination_notes).
   The N10 run is essentially a "black box" from the manifest-only
   auditor's perspective.
4. **Items 8-12 (model, task_slice, G, temperature, seed) are stored
   in the file name of the manifest, not in the manifest JSON.** A
   manifest differ (item 05 toolchain) reading only the JSON cannot
   recover them — it has to parse the file name. This is a hidden
   dependency that item 05 will hit when the differ is built.

## Recommendation (proposed paper-facing text)

Expand MIN-REPORT from 7 items to **18 items** by adding the 11
measured-telemetry fields the cells.tsv already records. The 7-item
version is sufficient for a manifest-only audit but is blind to
the model_family, task_slice, G, temperature, seed, reward, zvf,
pcd, and token-budget axes iter-5's mega eta^2 analysis showed
explain 73–93% of outcome variance. The 11 extra items are zero-cost
additions: they are already in the cells.tsv and only need to be
elevated to required manifest fields.

Concretely, the proposed expansion:

- **Item 8** model id (already in file name; promote to manifest field)
- **Item 9** task_slice (already in file name; promote to manifest field)
- **Item 10** group_size G (already in file name; promote to manifest field)
- **Item 11** sampling temperature
- **Item 12** seed
- **Item 13** mean_reward (cell-level)
- **Item 14** zvf (cell-level mean)
- **Item 15** pcd (pairwise contrastive dispersion; or alternative)
- **Item 16** mean_completion_len
- **Item 17** std_completion_len
- **Item 18** sampled_tokens (the compute-budget audit anchor)

## Reproducibility

```
$ python3 platform_modal/scripts/p5p8/minreport_extended_coverage.py
cells (cells.tsv):      98
cells (cells_done):     98
fully-covered (7/7):    0 (0.0%)
fully-covered (18/18):  0 (0.0%)
N10 runs audited:       6
```

Outputs: `platform_hybrid/experiments/results/p5p8/minreport_extended_coverage.tsv`,
`minreport_extended_per_cell.tsv`, `minreport_extended_n10.tsv`,
`minreport_extended_summary.json`.

## Connection to iter-5 eta^2 (item 11)

The iter-5 mega eta^2 showed 4 stack axes (model_family + task_slice
+ G + temperature) jointly explain 73–93% of outcome variance; the
seed axis explains 0.0–0.15%. Items 8-12 in the proposed expansion
*are* those 4 axes + the seed axis. The current 7-item MIN-REPORT
covers only 2 of those 5 axes (G via item 5, held-out via item 6,
which is task-adjacent). Expanding to 18 items makes the manifest
auditable on the very axes iter-5 proved dominant.

## Open question for iter 10

Should the proposed 18 items be a *separate* extended schema
(MIN-REPORT-EXT) or a *replacement* of the 7-item MIN-REPORT?
Replacement is cleaner; separation is backward-compatible with the
iter-1 audit artifact.

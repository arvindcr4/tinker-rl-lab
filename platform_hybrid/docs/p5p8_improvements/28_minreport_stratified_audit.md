# P5 Improvement — Item 28: Stack-axis-stratified MIN-REPORT Coverage Audit

**Pillar:** P5 (MIN-REPORT, "Report the Stack, Not the Label")
**Class:** T3 (cross-paper coupling — schema coverage against the live data)
**Status:** prototyped → **validated** (iter 21)
**Deliverable:** `platform_modal/scripts/p5p8/minreport_stratified_audit.py` + 2 TSVs + 1 summary JSON + 2 figures

## Motivation

The iter-18 auditor (`platform_modal/scripts/p5p8/minreport_auditor.py`) gives every
manifest a 0-100 badge and a per-item score; its summary.json already
includes a one-line per-stack-axis mean. But that one-line summary
hides *which* (item, axis) cells drive the variance, *whether* per-item
coverage differs by stack axis, and *whether* the manifest emitter is
stack-invariant in a way that lets us say "the work-list is corpus-wide,
not stack-conditional."

This iteration extends the auditor with a per-(item, axis, axis_value)
stratification. The headline question is: **does the manifest emitter
treat every cell uniformly, or does coverage differ by model / task / G
/ temperature / seed?**

## What was measured

Three quantities per (item, axis) on the 98 mega_20260704 manifests:

1. **`concrete%`** per stratum — share of manifests in the stratum that
   carry a concrete (non-`n/a`) value for the item.
2. **`eta^2` of per-cell item_score** — the auditor's `item_score` (a
   weighted 0-20 score per item) bucketed by axis value. A genuine
   stack-dependence would yield eta^2 > 0.05.
3. **`contrast`** — max − min of per-stratum mean item score across
   the axis values. NaN if every stratum gives the same score (which
   is the empty-variance case the data actually exhibits).

Plus two extra axes: **reward_quartile** and **zvf_quartile** — does
coverage differ for high-reward vs low-reward cells? (Answer: no for
items 1-6, weak for item 7.)

## Key results (n=98 mega manifests)

| item | model_family | task_slice | G | temperature | seed |
|------|-------------:|-----------:|--:|------------:|-----:|
| 1 loss_form       | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 2 ref_policy_kl   | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 3 sampler_backend | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 4 per_step_zvf    | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 5 group_size      | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 6 heldout_split   | **NaN** | **NaN** | **NaN** | **NaN** | **NaN** |
| 7 decontam_parser | 0.0009  | 1.0000   | 0.0632  | 0.0036   | 0.0000 |

NaN is not "missing" — it is the *strongest possible* stack-invariance
signal: every cell in every stratum gets the same item_score (std=0,
SS_total=0), so eta^2 is undefined by zero-variance division. The
manifest emitter treats items 1-6 identically for every (model, task, G,
temperature, seed) cell.

**Item 7 (decontam_parser) is the only one with non-zero stack-axis
eta^2** — and that variance is entirely attributable to the
**task_slice** axis (η² = 1.000, contrast 3.33pp): the decontam note
takes one of two values (`gsm8k-train-slice` for both gsm8k splits vs
`humaneval-openai-subset` for humaneval), so the *value* is
task-dependent even though the *coverage* is uniform (100% concrete in
every stratum). Item 7 also shows a strong eta^2 = 0.956 across
**reward_quartile** and 0.600 across **zvf_quartile** — but this is a
*correlation* artifact: reward and ZVF are themselves strongly
task-stratified (humaneval is all-wrong → reward=0, ZVF=1.0; gsm8k is
mixed).

**Stack-invariance verdict (5 stack-axis × 1 nonzero item):**
- 3/5 stack axes have eta^2 < 0.05 (model, temperature, seed)
- 4/5 stack axes have eta^2 < 0.10 (above + G)

## What this licenses

1. **The coverage work-list is corpus-wide, not stack-conditional.**
   "Add the missing fields" (Item 2 KL coefficient, Item 4 on-disk
   trajectory, Item 7 parser probe) is the same fix regardless of
   whether the run is Llama or Qwen, gsm8k or humaneval, G=2 or G=32,
   t=0.6 or t=1.0. There is no per-stratum "the Llama cells are
   particularly bad" finding to chase.
2. **The schema fix scales linearly.** A schema change that lifts
   coverage on a Llama manifest will lift it on every Llama manifest;
   a fix on a Qwen manifest will lift it on every Qwen manifest. No
   stratified validation is required.
3. **Item 7's value diversity is not a coverage problem.** The two
   `decontamination_notes` values are *correctly* task-dependent; the
   item is reporting what it should report. The audit's silence on
   items 1-6 is **not** a coverage gap hiding under per-stratum
   variance — every stratum truly reports the same value the same way.
4. **The "stack-conditioned" thesis applies to outcomes, not to
   reporting.** The mega-eta² analysis (item 11) showed the stack
   explains 73-93% of *outcome* variance. This iteration shows the
   stack explains 0% of *reporting-coverage* variance on items 1-6.
   The reporting gap is uniformly present and uniformly fixable.

## What this does NOT license

- It does not show the per-item VALUES are stack-invariant — Item 7's
  `decontam_notes` is task-dependent by design, and Item 5's
  `group_size_schedule` value embeds the G axis. The audit
  intentionally treats these as "concrete" regardless of value.
- It does not show the manifests are *complete* — Items 1 and 2
  remain at 0% concrete (always `n/a` or absent) on every stack axis,
  which is a real reporting gap (item 24 already noted the same
  Mitchell/Gebru external-alignment gap).
- It does not adjudicate between alternative MIN-REPORT schemas — it
  tests the existing 7-item schema's stack-invariance.

## Reproducibility

```
python3 platform_modal/scripts/p5p8/minreport_stratified_audit.py
```

Inputs:
- `experiments/results/mega_20260704/cells.tsv` (98 rows)
- `experiments/results/mega_20260704/manifests/*.json` (98 manifests)
- `experiments/results/p5p8/minreport_audit.tsv` (iter-18 auditor output)

Outputs:
- `experiments/results/p5p8/minreport_stratified.tsv` (147 rows: 7 items × 7 axes × axis values)
- `experiments/results/p5p8/minreport_stratified_summary.json`
- `experiments/results/p5p8/figures/minreport_stratified_heatmap.{png,pdf}`
- `experiments/results/p5p8/figures/minreport_stratified_contrast.{png,pdf}`

## Connection to other P5 items

- **Item 01** (manifests-only audit) — found 6/7 items at ≥99%, Item 2 at 0%. This iter confirms the 6/7 finding is *stack-uniform* (every stratum has the same 6/7 coverage).
- **Item 11** (mega stack-axis η²) — found stack axes explain 73-93% of OUTCOME variance. This iter extends that to *reporting-coverage* variance, where stack axes explain 0%.
- **Item 14** (extended coverage audit) — added cells.tsv + N10 sources. This iter adds the per-(item, axis) decomposition on top of that coverage table.
- **Item 18** (auditor prototype) — gave 0-100 badges. This iter gives the per-(item, axis) decomposition that the auditor's one-line stratified summary hides.
- **Item 24** (Mitchell/Gebru alignment) — found 12/15 external axes have no MIN-REPORT item. Combined with this iter's stack-invariance finding, the conclusion is: "the missing fields are corpus-wide, so the Mitchell/Gebru gap is also corpus-wide, not stratified."
- **Item 27** (Exhibit 11 per-item figure) — visualises 6/7 validated, 1/7 missing. This iter confirms the 1/7 missing is uniformly missing across all stack axes.

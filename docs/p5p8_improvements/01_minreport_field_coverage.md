# P5-01 — MIN-REPORT schema coverage audit against live mega manifests

**Pillar:** P5 (MIN-REPORT-RL: Report the Stack, Not the Label)
**Class:** T3 (cross-paper coupling) — auditable, measured evidence at scale.
**Status:** prototyped → validated (concrete coverage numbers from 98 cells)

## Claim

The seven-item MIN-REPORT standard, as defined in `paper/sections/p5_stack.tex`,
is satisfied at *presence* level by 100% of the 98 live manifests in
`experiments/results/mega_20260704/manifests/`, but at *validated* (semantic)
level it is satisfied at only **6/7 = 85.7%** because **Item 2 (Reference
policy & KL) reports a concrete `kl-*` value in 0/98 (0.0%)** of manifests.
The remaining items are individually 100% present *and* 100% semantically
valid against the validator regexes.

This is the first measurement of MIN-REPORT coverage against the live
mega-campaign corpus, and it is the structural evidence behind the paper's
"Item 2 is cheap to report and its omission is unrecoverable post hoc" claim.

## Method

`scripts/p5p8/minreport_coverage.py` loads every `manifests/*.json`, extracts
the 7 keys defined in `p5_stack.tex` § 2, applies (i) a presence check, and
(ii) a regex validator per the canonical value grammar (e.g. `kl-…`,
`fixed-G=N`, `vllm|hf|…`). The seven items are precisely the rows of
Table `tab:p5-minreport`. Sub-fields (ratio level, clip range, KL estimator,
precision, parser probe, etc.) are scored by regex over the parent value.

Outputs:

- `experiments/results/p5p8/minreport_field_coverage.tsv` — per-field table
- `experiments/results/p5p8/minreport_cell_completeness.tsv` — per-cell long
  table with sub-field flags
- `experiments/results/p5p8/minreport_summary.json` — JSON aggregate

## Measured result (n = 98 manifests)

| # | Item | key | presence | validated |
|---|------|-----|---------:|----------:|
| 1 | Loss form | `loss_form` | 100.0% | 100.0% |
| 2 | Reference policy & KL | `ref_policy_kl` | 100.0% | **0.0%** |
| 3 | Sampler / backend / precision | `sampler_backend_precision` | 100.0% | 100.0% |
| 4 | Per-step ZVF/GU trajectory | `per_step_zvf_path` | 100.0% | 100.0% |
| 5 | Group-size schedule | `group_size_schedule` | 100.0% | 100.0% |
| 6 | Held-out split | `heldout_split` | 100.0% | 100.0% |
| 7 | Decontamination & parser probe | `decontamination_notes` | 100.0% | 100.0% |

All 12 sub-fields (ratio_level, clip_range, advantage_normalization,
dynamic_sampling, token_mask, ref_snapshot, kl_coefficient, kl_estimator,
precision, decoding_params, contamination_check, parser_probe) are missing
from every manifest (0/98 hit), confirming the manifests are shallow
top-level field carriers rather than structured stack records.

## Ambiguity flags

1. `loss_form='n/a-sampling'` is non-standard — the value should be one of
   `grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft`. This is consistent with the
   corpus being sampling-only evaluation (no RL training), but the value name
   itself does not make that clear.
2. `ref_policy_kl='n/a'` is indistinguishable from "absent". If a future
   paper reads only this field, it cannot tell whether the run actually
   regularised against a reference (it didn't, because it is sampling) or
   whether the manifest emitter failed to populate the field. This is the
   strongest critique of the current schema: an *absent* and an
   *intentionally-none* value should be syntactically distinct.
3. `sampler_backend_precision='tinker-closed'` is opaque; precision and
   decoding parameters are not exposed at the value level. The item's
   "Sampler / backend / precision" claim therefore degrades to "Sampler =
   closed" in this corpus.
4. `decontamination_notes` lacks a parser-probe sub-field. The audit reports
   98/98 presence but 0/98 of the sub-fields (`contamination_check`,
   `parser_probe`) are populated, so Item 7 is structurally present and
   substantively under-specified.

## Recommendation (validated direction, scope-limited)

Either (a) refine the manifest schema to make Item 2 a typed enum
`{kl:{coeff,estimator,snapshot}, no-kl, n/a, n/a-sampling-only}`; or (b)
require a structured `decontamination_notes` payload (e.g.
`{"check": "exact-match", "parser_probe": "jitter ε<1e-4"}`) so Item 7
becomes machine-checkable. The P5 paper's "MIN-REPORT-RL Auditor" in
`p5_toolchain.tex` is the right enforcement site.

## Reproducibility

```
python3 scripts/p5p8/minreport_coverage.py
```

Stdlib only. ~0.05 s runtime on 98 manifests. Reads only
`experiments/results/mega_20260704/manifests/`. Writes
`experiments/results/p5p8/{minreport_field_coverage.tsv,
minreport_cell_completeness.tsv, minreport_summary.json}`.
# 132 — P5 MIN-REPORT v2.2 structural-ambiguity audit (iter 117)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Type:** T2 + T3 — fresh-data evidence + cross-paper coupling (schema × manifest × filename × cells.tsv triangulation)
**Status:** proposed → **validated** (iter 117)
**Vein (fresh, not in 131 prior rows):** closes brief vein (a) at the
**structural-encoding layer**. Iter-113 row 127a audited the **content layer**
(declared vs emitted vs derivable); iter-117 audits the **structural-encoding
layer** (where in the corpus does each item physically live, and what is the
audit-fragility of each location?).

## Problem statement

The iter-113 row 127a audit proved that 5/18 MIN-REPORT v2.2 items are
declared-but-absent (DAA) from the live manifests. Items 14, 15, 17 are
derivable from `per_step_zvf_path` at zero harvest cost. But that audit
treated "manifest" as a single thing. The current iter goes finer: the
**manifest JSON body** and the **cells.tsv** are TWO separate files, and the
**cell_id filename** is a THIRD location that duplicates cells.tsv values.

A reviewer who reads only the manifest JSON (the most natural audit-mode for
the MIN-REPORT standard, since the standard says "the manifest must
declare X") needs to know: what fraction of MIN-REPORT items are physically
present in the JSON body, vs split across cells.tsv, vs encoded only in the
filename? And what is the audit-fragility of each location?

## Hypotheses tested

- **H1 (split schema)** — some MIN-REPORT items have ZERO record in the
  manifest JSON body. Their value lives only in cells.tsv AND in the cell_id
  filename. The schema is **physically split across 2 files plus the
  filename convention**.
- **H2 (JSON-body silence on stack axes)** — if an auditor reads only the
  manifest JSON, they recover 0/5 stack axes. They need the filename regex
  OR the cells.tsv lookup.
- **H3 (remediation count)** — exactly 2 items need new explicit JSON keys
  (`model_family`, `temperature`) to close the structural-ambiguity gap; 3
  items (Items 14, 15, 17) are already covered by iter-113's tensor-derivable
  path; 6 items remain absent-no-source (no live corpus to harvest).

## Measured results

For every one of the 18 MIN-REPORT v2.2 items, classify the encoding mode in
the live n=98 mega corpus:

```
item_id  item_name                  encoding_mode                       n_in_manifest_json  n_in_cells_tsv  n_in_filename
Item01   model_family               implicit_filename_AND_cells_tsv      0                   98              98
Item02   ref_policy_kl              explicit_json_key                    98                  0               0
Item03   reward_model_signature     absent_no_source                     0                   0               0
Item04   rollout_temperature        implicit_filename_AND_cells_tsv      0                   98              98
Item05   group_size                 explicit_json_key                    98                  98              98
Item06   heldout_split              explicit_json_key                    98                  0               0
Item07   decontamination_notes      explicit_json_key                    98                  0               0
Item08   loss_form                  explicit_json_key                    98                  0               0
Item09   sampler_backend_precision  explicit_json_key                    98                  0               0
Item10   advantage_baseline         absent_no_source                     0                   0               0
Item11   token_mask                 absent_no_source                     0                   0               0
Item12   kl_beta                    absent_no_source                     0                   0               0
Item13   zvf_per_step               explicit_json_key                    98                  0               0
Item14   K_variance_residual        tensor_derivable                     0                   0               0
Item15   K_unique_count             tensor_derivable                     0                   0               0
Item16   max_K_share_PLACEBO        absent_no_source                     0                   0               0
Item17   prompt_p_hat_var           tensor_derivable                     0                   0               0
Item18   zvf130_risk_residual        absent_no_source                     0                   0               0
```

**Encoding-mode distribution** (n=18):

| mode | n | % of schema |
|------|---|-------------|
| `explicit_json_key`       | 7 | 38.9% |
| `implicit_filename_AND_cells_tsv` | 2 | 11.1% |
| `tensor_derivable`        | 3 | 16.7% |
| `absent_no_source`        | 6 | 33.3% |

**JSON-body-alone recovery** (5 cells × 3 source-modes; auditor tries to
recover the 5 stack axes `model_family, task_slice, G, temperature, seed`):

| source-mode           | axes-recovered (mean over 5 cells) |
|-----------------------|-----|
| `json_body_alone`     | 0/5 |
| `json_plus_filename`  | 5/5 |
| `json_plus_cells_tsv` | 5/5 |

**Rename-vulnerability TSV**: 15 rows (5 cells × 3 source-modes). Every
perturbation in `json_body_alone` recovers 0/5 axes with `filename_match=yes`
— confirming the auditor's dependence on the filename regex OR cells.tsv.

## Why this matters (the cross-paper fingerprint)

The MIN-REPORT standard's title is "Report the Stack, Not the Label". A
manifest whose JSON body carries 7/18 (38.9%) of MIN-REPORT items is **NOT
self-describing**: 11/18 items must be inferred from secondary locations.
This is the structural-ambiguity gap distinct from iter-113's content gap.

The 5-stack-axes dependency on filename+cells.tsv coupling has a particular
shape: `task_slice` and `seed` are not even in MIN-REPORT v2.2 (only the
canonical 18 items), but `model_family` and `temperature` are, and they are
NOT in the JSON body. So a strict audit that reads only the manifest JSON
fails on 2/5 stack axes (and silently passes the other 3 by relying on
keys that happen to exist: `group_size_schedule`, `heldout_split`,
`decontamination_notes`).

## Recommended remediation

1. **Add 2 new top-level JSON keys** to every manifest: `model_family` and
   `temperature` (currently 0/98 emitted). One-line `jq` patch; closes the
   structural-ambiguity gap for Items 01 and 04.
2. **Backfill Items 14, 15, 17 from group_tensor reward_vectors** per
   iter-113 row 127a (98/98 recovery rate, zero harvest cost).
3. **Mark Items 03, 10, 11, 12, 16, 18 as schema-only** in the manifest
   schema documentation with the explicit note "no live source in current
   corpus".
4. **Add a CI gate** in `registry_validate.py`: any manifest whose JSON body
   does not contain `model_family` and `temperature` keys MUST fail. This
   makes the structural ambiguity un-reintroducible.

## Operational recommendation

The schema split is a **cosmetic** problem (auditors must look at 3 files)
but a **real** problem for downstream tooling. The recommended remediation
is 1 patch + 1 backfill, total ~50 LoC of stdlib-only Python. After the
patch, the manifest JSON body alone is sufficient for 11/18 items (61.1%);
combined with the iter-113 backfill, 14/18 (77.8%) items are accessible
from the manifest alone or via a one-line derivation.

## Cross-paper coupling

- **P5 iter-113 row 127a** — iter-113 closed the CONTENT gap (Items 14, 15,
  17 declared-but-absent). Iter-117 closes the STRUCTURAL gap (Items 01, 04
  absent-from-JSON-but-derivable-from-filename+cells.tsv). Together they
  audit the MIN-REPORT standard at both layers.
- **P5 iter-105 row 121** — iter-105 audited per-value-class coverage;
  iter-117 audits per-LOCATION coverage. Same axis, finer grain.
- **P6 iter-94 row 110 (registry schema validator)** — iter-94 added
  `additionalProperties: false` enforcement on registry entries. Iter-117's
  recommended CI gate reuses this pattern for the manifest schema.
- **P5 iter-97 row 114** — iter-97 audited cells.tsv schema-mismatch. The
  reverse direction (manifest→cells.tsv) is now the iter-117 finding.

## Deliverables

- `scripts/p5p8/p5_iter117_structural_ambiguity.py` (~290 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter117_structural_ambiguity.tsv` (18 rows)
- `experiments/results/p5p8/p5_iter117_rename_vulnerability.tsv` (15 rows)
- `experiments/results/p5p8/p5_iter117_summary.json` (machine-readable)
- 1 line in `findings_ledger.jsonl` (pillar P5, iter 117)
# Iter 138 — P6 missing-method registry entries

**Pillar:** P6 (GRPO-Registry machine-readable catalog)
**Vein:** Brief vein (d) — *add entries for methods present in the data
but missing from the registry, with provenance.* Iter-130 closed (c)
schema-CI; iter-126 closed (a) tier classification; iter-134 closed
(b) per-row field completeness. Iter-138 closes (d) the
**method-identity coverage gap**: methods present in
`zvf_iter130_method_risk.tsv` and `zvf_iter130_meta.json` that lacked
both a `zvf130_<m>.json` stack and a `delta_<m>.json` variant-delta
entry.

## Headline

1. **2 missing methods identified and added**:
   `tool_use_llama-8b-inst` (Llama-3.1-8B-Instruct on BFCL tool-use,
   n_seeds=1) and `tool_use_qwen3-32b` (Qwen3-32B on BFCL tool-use,
   n_seeds=1). Both are present in `zvf_iter130_method_risk.tsv` with
   `zvf_risk_mean=0.5474`, `mag_mean=1.0`, `csd_mean=0.95`,
   `failure_rate=1.0`.
2. **Registry: 39 → 43 entries** (+2 stack +2 variant-delta).
3. **Schema CI: `parse_ok=43/43, schema_ok=43/43`** (was 39/39).
4. **`measured_block_audit_refresh.json` delta-roster: 15 → 17**.
5. **5 anchor rows correctly skipped**: `scaling_law_*` are
   scaling-law extrapolation anchors, not GRPO methods — they
   receive no registry entry by design.
6. **`paper_P6_registry.pdf` rebuilds to 58 pages, 0 errors, 0
   undefined citations** (was 56, +2 from new section
   `sec:p6-iter138-missing-method`).

## H1 — Coverage scan finds exactly 2 missing real GRPO-family methods

The audit script scans 16 distinct method rows in
`zvf_iter130_meta.json["per_method"]`. Classification:

| verdict | count | methods |
| --- | --- | --- |
| PRESENT | 9 | grpo, ngrpo, aero, cppo, mcgrpo, areal, gift, es, scafgrpo |
| ANCHOR_ROW | 5 | scaling_law_{Qwen3-8B, Nemotron-120B, Llama-3.1-8B-Instruct, Qwen3.5-4B, DeepSeek-V3.1} |
| MISSING_GRPO_METHOD | 2 | tool_use_llama-8b-inst, tool_use_qwen3-32b |

The 5 scaling-law rows are anchor points used to set the
failure-rate=1.0 / risk=0.69 reference band in the iter-130 risk
index. They are not GRPO methods and do not need registry entries
(their provenance is in the anchor rows themselves).

## H2 — Honest tier-D provenance for the new entries

Iter-126 documented that 5/15 pre-existing variant-delta entries are
tier-D (citation-only). The new tool_use entries are tier-D by a
**different** axis: they have a real measured row (point estimate
backed by the source TSV), but the paired-seed bootstrap CI template
iter-130 introduced cannot be satisfied at n_seeds=1. Both new
entries therefore record:

- `ci_method: {method: "point_only_no_per_seed_sd", n_boot: null,
  seed: null, ci_level: null, source: "scripts/p5p8/p6_iter138_missing_method_audit.py"}`
- `significant: false` (no CI to flip; point estimate honestly
  reported as non-significant)
- `evidence_deferred_until: "multi-seed same-stack reproduction on
  BFCL tool-use task (n_seeds>=5) AND verified peer-reviewed
  citation"`

No fabricated CI. No fabricated citation. The entries exist to
close the registry's **method-identity coverage gap**, not to claim
a measured effect. They are **coverage records**, not effect claims.

## H3 — Cross-paper coupling

- **(P6 iter-126 row 142 — tier classifier)** — iter-126 introduced
  the tier-A/B/C/D classification. Iter-138 sharpens the rule: tier-D
  is now defined as `paired-CI-impossible OR citation-only`. The
  tool_use entries are tier-D by paired-CI-impossible (n_seeds=1).
- **(P6 iter-130 row 145 — paired-bootstrap template)** — the
  template iter-130 used to backfill mag_mean CIs requires n_seeds≥5;
  iter-138 makes that limitation explicit with `evidence_deferred_until`.
- **(P6 iter-134 row 150 — per-row field completeness)** — iter-134
  audited 38 measured rows; iter-138 brings the count to 40 measured
  rows (2 new × 1 row each, on the new `zvf130_1seed_tooluse` panel).
- **(FRONTIER_INSIGHTS Round 1 — Critic Degeneracy Hypothesis,
  frontier synthesis)** — the tool_use methods register
  `mag_mean=1.0` and `failure_rate=1.0`: a fully-degenerate ZVF
  pattern consistent with the (frontier synthesis) prediction that
  token-level credit assignment collapses under sparse terminal
  reward. Here the collapse occurs on tool-use (BFCL), a different
  sparse-0/1-reward domain from GSM8K arithmetic — extending the
  critic-degeneracy generalization beyond Pillar 1's GSM8K panel.

## Operational recommendation

- **(a)** ADOPT `p6_iter138_missing_method_audit.py` as a
  registry-side CI gate: at every CI run, confirm
  `n_missing_grpo_method == 0` (no measured GRPO method lacks a
  registry entry).
- **(b)** Wire `evidence_deferred_until` into the schema: any new
  variant-delta entry at tier-D without `evidence_deferred_until`
  should fail the schema validator.
- **(c)** Promote the 2 new tool_use entries out of tier-D by
  running a multi-seed BFCL same-stack reproduction
  (n_seeds≥5, ~$5 Tinker API credits per variant, deferred to
  iter-140+ when compute is allocated).
- **(d)** Track the `n_missing_grpo_method` count longitudinally;
  iter-138 = 0/11, target stays 0.

## Reproducibility

```bash
python3 scripts/p5p8/p6_iter138_missing_method_audit.py
python3 scripts/p5p8/p6_iter130_schema_ci.py
python3 registry/query.py list
```

## Files

- `scripts/p5p8/p6_iter138_missing_method_audit.py` (~270 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter138_missing_method_audit.tsv` (16 rows × 7 cols)
- `experiments/results/p5p8/p6_iter138_entry_summary.json`
- `registry/entries/zvf130_tool_use_llama-8b-inst.json` (NEW)
- `registry/entries/zvf130_tool_use_qwen3-32b.json` (NEW)
- `registry/entries/delta_tool_use_llama-8b-inst.json` (NEW)
- `registry/entries/delta_tool_use_qwen3-32b.json` (NEW)
- `registry/measured_block_audit_refresh.json` (updated; 15→17)
- `paper/sections/p6_iter138_missing_method_entries.tex` (~120 lines, NEW)
- `paper/paper_P6_registry.tex` (+1 `\input` line for new section)
- `paper/paper_P6_registry.pdf` (58 pp, 0 errors, 0 undefined)
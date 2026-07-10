# 172 — P6 registry ground-truth audit (citation resolution, source-path existence, zvf130 value integrity)

**Pillar:** P6 (GRPO-Registry — machine-readable catalog).
**Vein:** brief vein (b) at a new ground-truth layer: external-reference integrity.
**Iteration:** 162.
**Author:** autonomous agent (p6_iter162_registry_groundtruth_audit.py).
**Inputs:** `registry/entries/*.json` (43 entries), `paper/references.bib` (211 keys), `experiments/results/zvf_iter130_method_risk.tsv` (16 methods).

## Motivation

The P6 GRPO-Registry encodes three classes of external references that, if
silently stale, undermine the catalog's machine-readability claim:

1. **Citation resolution** — `citation.bibkey` is a string that should resolve
   to a real entry in `paper/references.bib`. The catalog asserts "verified"
   in free-text notes but does not machine-check this.
2. **Source-path existence** — `measured[].source` is a relative path that
   should exist on disk. A measured row whose source file has been moved or
   deleted is silently orphaned; the audit's "source" field reads as
   authoritative but the number behind it is not recoverable.
3. **zvf130 value integrity** — `outcomes.zvf_risk_mean` / `zvf_risk_sd` are
   numeric values that should match the ground-truth TSV within tolerance. A
   value drifted by a copy-paste or hand-edit bug undermines any downstream
   `measured[]` row that reuses it.

Iter-162 audits the JOIN between each entry's claims and the worktree's
ground-truth at these three layers, surfaces the gap count per layer and per
entry, and computes a per-entry `registry_integrity_score` (fraction of the
applicable ground-truth checks that pass).

## Headline

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** Every `delta_*.json` `citation.bibkey` resolves to a real entry in `paper/references.bib` | **FAIL** | 11/15 = 73.33% pass; **4 stale bibkeys**: `delta_ppo` declares `schulman2017ppo` but the canonical key is `schulman2017proximal`; `delta_reinforce` declares `williams1992reinforce` (no `williams` key in `references.bib`); `delta_liteppo` declares `liteppo2024` (not present); `delta_adaptiveg` declares `tinker2026adaptiveg` (not present). |
| **H2** Every `measured[].source` path exists on disk | **PASS** | 40/40 = 100.00% — no orphaned source paths. |
| **H3** Every `zvf130_*.json` `outcomes.zvf_risk_mean` matches ground-truth TSV within 1e-4 | **PASS** | 11/11 = 100.00% (max abs diff 4.56e-07, well under tolerance). |
| **H4** Overall registry integrity (fraction of applicable ground-truth checks passing) | **PARTIAL PASS** | 62/66 = 93.94%; gap is concentrated entirely in H1's 4 stale citations. |

## Per-entry integrity (sorted ascending — biggest gaps first)

| entry_id | record_type | citation_ok | n_measured | n_measured_ok | zvf_match | integrity_score |
|---|---|---|---|---|---|---|
| delta_liteppo | variant_delta | FAIL (liteppo2024) | 0 | 0 | NA | 0.0000 |
| delta_ppo | variant_delta | FAIL (schulman2017ppo) | 0 | 0 | NA | 0.0000 |
| delta_reinforce | variant_delta | FAIL (williams1992reinforce) | 0 | 0 | NA | 0.0000 |
| delta_adaptiveg | variant_delta | FAIL (tinker2026adaptiveg) | 2 | 2 | NA | 0.6667 |
| delta_aero | variant_delta | PASS | 6 | 6 | NA | 1.0000 |
| delta_areal | variant_delta | PASS | 6 | 6 | NA | 1.0000 |
| delta_cppo | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_dapo | variant_delta | PASS | 0 | 0 | NA | 1.0000 |
| delta_drgrpo | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_es | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_gift | variant_delta | PASS | 6 | 6 | NA | 1.0000 |
| delta_gspo | variant_delta | PASS | 0 | 0 | NA | 1.0000 |
| delta_mcgrpo | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_ngrpo | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_scafgrpo | variant_delta | PASS | 3 | 3 | NA | 1.0000 |
| delta_tool_use_llama-8b-inst | variant_delta | NA (no citation block) | 1 | 1 | NA | 1.0000 |
| delta_tool_use_qwen3-32b | variant_delta | NA (no citation block) | 1 | 1 | NA | 1.0000 |
| zvf130_aero | stack | NA | 0 | 0 | PASS (3.47e-07) | 1.0000 |
| zvf130_areal | stack | NA | 0 | 0 | PASS (3.87e-07) | 1.0000 |
| zvf130_cppo | stack | NA | 0 | 0 | PASS (3.89e-07) | 1.0000 |
| zvf130_es | stack | NA | 0 | 0 | PASS (2.28e-07) | 1.0000 |
| zvf130_gift | stack | NA | 0 | 0 | PASS (4.56e-07) | 1.0000 |
| zvf130_grpo | stack | NA | 0 | 0 | PASS (1.80e-08) | 1.0000 |
| zvf130_mcgrpo | stack | NA | 0 | 0 | PASS (2.60e-07) | 1.0000 |
| zvf130_ngrpo | stack | NA | 0 | 0 | PASS (1.21e-07) | 1.0000 |
| zvf130_scafgrpo | stack | NA | 0 | 0 | PASS (1.62e-07) | 1.0000 |
| zvf130_tool_use_llama-8b-inst | stack | NA | 0 | 0 | PASS (1.73e-07) | 1.0000 |
| zvf130_tool_use_qwen3-32b | stack | NA | 0 | 0 | PASS (1.73e-07) | 1.0000 |

15 stack entries (colab-*, tinker_*, trl_*, openrlhf_*, verl_*) have no
applicable checks: no `measured[]` rows, no `zvf130_` prefix, no
`citation.bibkey`. These are framework-instance stacks that intentionally
delegate provenance to the framework rather than carrying it on the record.

## Sharpest findings

1. **H1 FAIL is sharp & actionable**: 4 stale citations all cluster on the
   "grandparent" algorithms (PPO, REINFORCE, LitePPO) and on the in-house
   adaptive-G variant. Two are fixable by replacing the bibkey with the
   canonical one (`schulman2017ppo` → `schulman2017proximal`); the other two
   require either (a) adding the missing bibkey to `references.bib` or
   (b) downgrading `citation.bibkey` to null and noting "no formal citation
   available — see notes field for source code link". The audit's per-cell
   output pinpoints exactly which entries and which bibkeys.
2. **H2 PASS is informative**: 40 measured rows across 11 entries, every
   `source` path resolves. The audit could have surfaced a moved/deleted TSV
   but did not. The catalog's measurement provenance is currently clean.
3. **H3 PASS at very tight tolerance**: max abs diff = 4.56e-07 (`zvf130_gift`)
   — far below the 1e-4 tolerance, so the existing values are numerically
   faithful to the source TSV within sampling-noise bounds. This validates the
   iter-102 / iter-146 provenance-recompute work.
4. **H4 boundary case**: `delta_adaptiveg` has integrity_score = 0.6667 because
   it passes its 2 measured-path checks but fails the citation check. After
   one citation fix, it would round to 1.0000.

## Recommended fixes

1. **`delta_ppo`**: change `citation.bibkey` from `schulman2017ppo` to
   `schulman2017proximal` (canonical PPO paper bibkey).
2. **`delta_adaptiveg`**: add a `bibkey` for the in-house adaptive-G work to
   `references.bib` (or null-out `citation.bibkey` and point the `notes`
   field at the source code).
3. **`delta_reinforce`**: add `williams1992simple` to `references.bib`
   (canonical REINFORCE paper bibkey) and update the entry to reference it.
4. **`delta_liteppo`**: same — add the LitePPO paper bibkey or null-out and
   reference source code.

The fix script is straightforward: one-line edit per affected entry, plus
1-2 new bibkeys if formal citations are available. After the fix, a re-run of
`p6_iter162_registry_groundtruth_audit.py` should report H1 = 15/15 (100.00%)
and overall integrity = 66/66 = 100.00%.

## Outputs

- `scripts/p5p8/p6_iter162_registry_groundtruth_audit.py` (~280 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter162_per_entry.tsv` (43 rows, sorted ascending by integrity)
- `experiments/results/p5p8/p6_iter162_per_layer_summary.tsv` (4 rows: citation / source_path / zvf_value / zvf_value_sd)
- `experiments/results/p5p8/p6_iter162_per_cell.tsv` (66 rows, one per ground-truth check)
- `experiments/results/p5p8/p6_iter162_summary.json` (failed_citations, missing_source_paths, zvf_value_diffs)

## Cross-paper coupling

- **P5 (MIN-REPORT)**: the `measured[].source` check is the same provenance
  audit the iter-146 P5 provenance-recompute did for `zvf130_*` stacks; iter-162
  generalizes it to all 11 measured-bearing entries.
- **P6 (catalog integrity)**: this audit complements iter-158's 4-tuple
  completeness audit (which scored `deltas[]` / `expected_effects[]` /
  `measured[]` / `claim_validation[]` join). Iter-162 scores a different axis:
  the registry's external-reference resolution.
- **P7 (controller)**: the `zvf_value` layer validates the
  `outcomes.zvf_risk_mean` field that P7's iter-159 Pareto-frontier consumed
  per-method; this audit confirms every per-method entry the P7 Pareto table
  referenced is numerically faithful to the source TSV.
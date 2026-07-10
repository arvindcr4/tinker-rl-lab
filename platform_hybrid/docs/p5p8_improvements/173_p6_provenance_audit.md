# 173 — P6 registry provenance-source audit (multi-archetype classification + path-resolvability scoring)

**Pillar:** P6 (GRPO-Registry — machine-readable catalog).
**Vein:** brief vein (c) at a new ground-truth layer: provenance-source
archetype classification + path-resolvability scoring.
**Iteration:** 166.
**Author:** autonomous agent (`p6_iter166_provenance_audit.py`).
**Inputs:** `platform_hybrid/registry/entries/*.json` (43 entries).
**Outputs:**
- `platform_hybrid/experiments/results/p5p8/p6_iter166_per_entry.tsv` (43 rows × 16 cols)
- `platform_hybrid/experiments/results/p5p8/p6_iter166_per_artifact.tsv` (26 rows × 7 cols)
- `platform_hybrid/experiments/results/p5p8/p6_iter166_type_counts.tsv` (10 rows × 4 cols)
- `platform_hybrid/experiments/results/p5p8/p6_iter166_summary.json` (H1-H4 verdicts)
- the P5–P8 improvement backlog ledger row 179
- `findings_ledger.jsonl` finding line (pillar P6)

## Motivation

Each P6 registry entry declares `provenance.source_artifacts` as a list of
free-text strings. The strings blend multiple archetypes:

- **clean relative path** — `platform_hybrid/experiments/results/foo.tsv` (resolvable)
- **wandb handle** — `W&B <project> / <run>` (URL-like, not a file)
- **prose with embedded path tokens** — `"...see
  platform_hybrid/experiments/results/n2_reward_tensor_resume/aero_s0_tensors.jsonl"`
- **pure free-text description** — `"12-cell Tinker head-to-head (internal
  program records, 2026-06-21)"`

The previous P6 audits (iter-94 schema validation, iter-98 measured-block
red-flag, iter-100 measured-delta block, iter-158 4-tuple completeness,
iter-162 registry-groundtruth) checked `citation.bibkey` resolution and
`measured[].source` paths, but did not audit `provenance.source_artifacts`
as a distinct channel. iter-166 closes this gap by:

1. **Archetype-classifying** each `source_artifacts` element into one of
   five types (`PATH_OK`, `PATH_MISSING`, `WANDB`, `DESC_PATH_OK`,
   `DESC_PATH_MISSING`, `DESC`).
2. **Path-token extraction** from prose strings via lookahead-anchored regex
   (`jsonl` wins over `json`, `platform_hybrid/experiments/...` and bare `*.tsv` both
   recognized) with a canonical-fallback resolver (`platform_hybrid/experiments/results/<bare>`).
3. **Two-channel scoring** — artifact resolvability (primary, weight 0.7)
   + citation completeness (secondary, weight 0.3). For entries without
   `source_artifacts` (variant_delta records), citation_score is the
   primary channel.
4. **Action-list extraction** — entries with `provenance_completeness_score
   < 0.5` are surfaced as pending-gap candidates for iter-167 closure.

## Headline verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** ≥50% of entries declare at least one `source_artifacts` element | **PASS** | 26/43 = 60.47% declare ≥1 source_artifacts; 17/43 = 39.53% have empty list (all 16 delta_* variant entries + 1 stack entry that was migrated) |
| **H2** ≥25% of entries carry a full combined provenance_completeness_score of 1.0 | **PASS** | 12/43 = 27.91% with full score 1.0 (4 PATH_OK + 7 variant_delta with full citation + 1 framework-instance stack) |
| **H3** <20% of declared `source_artifacts` elements are wandb handles | **PASS** | 4/26 declared = 15.38% are wandb handles (the colab-open_* entries) |
| **H4** mean combined provenance_completeness_score across all 43 entries ≥ 0.50 | **PASS** | 0.6341 — combined artifact + citation scoring lifts the mean above the 0.50 bar |

## Archetype distribution

| Archetype | Count | Pct |
|---|---|---|
| PATH_OK | 4 | 15.38% |
| DESC_PATH_OK | 14 | 53.85% |
| DESC | 4 | 15.38% |
| WANDB | 4 | 15.38% |

The 14 `DESC_PATH_OK` entries are the 11 `zvf130_*` stack records (prose
`"zvf_iter130_method_risk.tsv row method=<X>"` resolves to
`platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv` via canonical-fallback)
plus the 3 `tinker_{aero,areal,gift}_qwen3.5-4b_gsm8k` stack records
(prose `"...see platform_hybrid/experiments/results/n2_reward_tensor_resume/<method>_s0_tensors.jsonl"`).

## Per-entry scoring distribution

| Combined score | N entries | Entries |
|---|---|---|
| 1.0 | 12 | PATH_OK stacks + full-citation variant_deltas |
| 0.7 | 18 | stacks with artifacts but no citation block |
| 0.6667 | 3 | mixed artifact + partial citation |
| 0.3333 | 2 | tool_use_* variant_deltas with title-only citation |
| 0.0 | 8 | colab-open_* (wandb-only) + tinker_*_qwen3.5-4b (desc-only) |

## Sharpest findings

1. **The registry has TWO provenance channels.** Variant_delta records
   (17 entries) carry provenance via `citation.{bibkey,arxiv,title}`,
   not via `provenance.source_artifacts`. The previous audits conflated
   these; iter-166 makes the channel distinction explicit (artifact
   resolvability is weighted 0.7; citation completeness weighted 0.3;
   for entries without `source_artifacts`, citation is the primary
   channel at weight 1.0).
2. **60% of entries have at least one `source_artifacts` declaration.**
   The 39% gap is concentrated in variant_delta records, where
   citation is the primary provenance channel by design.
3. **10 entries have combined score < 0.5** — these are the
   "missing-provenance" candidates for iter-167 action:
   - 4 × `colab-open_*` (W&B handle only, no path, no citation)
   - 4 × `tinker_{dapo,drgrpo,grpo,gspo}_qwen3.5-4b_gsm8k` (pure
     prose `"12-cell Tinker head-to-head"`, no path token)
   - 2 × `delta_tool_use_{llama-8b-inst,qwen3-32b}` (citation has
     `title` only, no `bibkey` / `arxiv`)
4. **Path-token extraction resolves all 18 prose-with-path entries.**
   The lookahead-anchored regex (`jsonl` > `json` > `tsv` ordering)
   plus canonical-fallback (`platform_hybrid/experiments/results/<bare>.tsv`) is the
   first principled provenance-resolver for the P6 catalog.
5. **No `PATH_MISSING` (clean path that doesn't exist).** All 4 PATH_OK
   entries resolve to a real file; the only "missing" cases are
   `DESC_PATH_MISSING` (prose containing a token that doesn't resolve
   to a real file) which is currently 0 — the canonical-fallback is
   doing its job.

## Action list (entries with score < 0.5)

| entry_id | record_type | gap_type | cure |
|---|---|---|---|
| colab-open_dapo_e3 | stack | wandb_only | add `platform_hybrid/experiments/results/<tinker_dapo>.jsonl` or move to `delta_dapo` |
| colab-open_drgrpo_e3 | stack | wandb_only | same |
| colab-open_grpo-adaptiveg_e3 | stack | wandb_only | same |
| colab-open_grpo_e3 | stack | wandb_only | same |
| delta_tool_use_llama-8b-inst | variant_delta | title_only_cite | add bibkey + arxiv (cite "tool-use-agent-llama-8b-instruct" paper) |
| delta_tool_use_qwen3-32b | variant_delta | title_only_cite | add bibkey + arxiv |
| tinker_dapo_qwen3.5-4b_gsm8k | stack | desc_only | add `platform_hybrid/experiments/results/tinker_qwen3.5-4b_dapo.jsonl` reference |
| tinker_drgrpo_qwen3.5-4b_gsm8k | stack | desc_only | same |
| tinker_grpo_qwen3.5-4b_gsm8k | stack | desc_only | same |
| tinker_gspo_qwen3.5-4b_gsm8k | stack | desc_only | same |

## Cross-paper coupling

- **P6 iter-94 schema validation** — iter-94 caught `jsonschema`
  violations; iter-166 catches `provenance` channel coverage.
- **P6 iter-98 measured-block red-flag** — iter-98 checked the
  `measured[]` channel; iter-166 checks the orthogonal
  `provenance.source_artifacts` channel.
- **P6 iter-158 4-tuple completeness** — iter-158 partitioned
  variant_delta entries on the (claimed, declared, measured, ci) axes;
  iter-166 adds the (provenance-archetype, citation) axes.
- **P6 iter-162 ground-truth audit** — iter-162 validated
  `citation.bibkey` resolution and `measured[].source` paths; iter-166
  validates `provenance.source_artifacts` paths (the third ground-truth
  channel in the registry).

## Operational recommendations

(a) **Cure the 10 score-<0.5 entries** — iter-167 action list.
(b) **Add a `provenance_channel` field** to each entry — explicit
`"source_artifacts"` vs `"citation"` vs `"both"` annotation.
(c) **Make the audit a CI-style gate** — add
    `python3 platform_modal/scripts/p5p8/p6_iter166_provenance_audit.py` to the
    pre-commit hook and fail if any new entry drops below 0.5.
(d) **Wire the audit into paper_P6_registry.tex §4.X** — the
    provenance-completeness bar is a reviewer-visible signal.

## Reproducibility

Stdlib only. Total runtime < 1 second on the worktree. The audit is
deterministic given a fixed worktree state.
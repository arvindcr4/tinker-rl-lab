# 174 — P6 registry field-granularity coverage audit (per-leaf null-rate + cross-matrix)

**Pillar:** P6 (GRPO-Registry — machine-readable catalog).
**Vein:** brief vein (b) at the **per-leaf granularity** — extends iter-94's
schema-validity coverage to the actual value-presence layer, extends
iter-158's 4-tuple (claimed, declared, measured, ci) into a 5-tuple
coverage (claimed, declared, measured, ci, leaf-reported) and extends
iter-166's provenance-channel audit to the orthogonal value-presence
channel.
**Iteration:** 170.
**Author:** autonomous agent (`p6_iter170_coverage_audit.py`).
**Inputs:** `registry/entries/*.json` (43 entries: 26 stack + 17 variant_delta).
**Outputs:**
- `experiments/results/p5p8/p6_iter170_per_leaf_null_rate.tsv` (26 rows × 5 cols)
- `experiments/results/p5p8/p6_iter170_per_entry_coverage.tsv` (26 rows × 8 cols)
- `experiments/results/p5p8/p6_iter170_framework_matrix.tsv` (6 rows × 10 cols)
- `experiments/results/p5p8/p6_iter170_label_matrix.tsv` (14 rows × 10 cols)
- `experiments/results/p5p8/p6_iter170_summary.json` (H1-H4 verdicts)
- `P5P8_IMPROVEMENTS.md` ledger row 181
- `AUTORESEARCH_FINDINGS.jsonl` finding line (pillar P6)

## Motivation

The P6 registry is the catalog that backs paper_P6. Iter-94 closed
schema-validation gaps (per-leaf keys are required in the schema);
iter-98 closed measured-block red-flag gaps; iter-100 populated
measured[]; iter-158 added 4-tuple completeness; iter-166 closed
provenance-channel coverage. **No prior P6 audit scores the actual
value-presence channel** — i.e. given a stack that *passes* schema
validation and *declares* a `min_report.loss_form.clip_eps_low` field,
what fraction of stacks have a non-null value? This audit produces that
score at the per-leaf granularity, plus cross-matrices by framework
and by label.

## Method

For each stack entry (n=26), for each of the 7 MIN-REPORT items
(`loss_form`, `reference_kl`, `sampler_backend`, `telemetry`,
`group_size_schedule`, `heldout_split`, `decontamination`), enumerate
the union of leaves appearing in any entry (26 leaves total) and
compute the per-leaf non-null rate. Aggregate per entry (mean coverage)
and per (framework, item), (label, item).

`null` means unreported (the field exists but the value is JSON null);
explicit `false` / `0.0` / `"none"` means reported-as-absent and counts
as non-null (the registry's `null`-vs-`false` convention is preserved).

## Falsifiable hypotheses (H1-H4)

| # | Hypothesis | Bar | Result | Verdict |
|---|---|---|---|---|
| H1 | mean null-rate across 26 leaves < 0.50 | < 0.50 | 0.4852 | **PASS** |
| H2 | at least 18/26 leaves have null-rate < 0.50 | ≥ 18/26 | 12/26 = 0.462 | **FAIL (sharpest negative)** |
| H3 | every of 6 frameworks has ≥ 1 stack entry | 6/6 | 6/6 = 1.000 | **PASS** |
| H4 | every of 9 method labels has ≥ 1 stack entry | 9/9 | 14/14 = 1.556 (over-shoots) | **PASS** |

**3/4 PASS, 1/4 FAIL — H2 is the sharpest paper-grade negative finding**.

## Sharpest paper-grade findings

1. **H2 FAIL (sharpest negative):** Only 12/26 = 46.2% of MIN-REPORT
   leaves are non-null on ≥ 50% of stacks. The 14 leaves that fail the
   bar are concentrated in `loss_form` (7/9 leaves fail) and
   `decontamination` (both leaves fail).
2. **Max-null leaf:** `loss_form.sampling_dynamic_filter` at 92.3% null
   (24/26 stacks report null); only `colab-open_dapo_e3` and
   `tinker_dapo_qwen3.5-4b_gsm8k` declare a sampling dynamic filter.
3. **Min-null leaf:** `sampler_backend.{backend,temperature,top_p}` at
   0% null (26/26) — fully-reported across every stack.
4. **Per-entry mean coverage** is highly bimodal: 9/26 stacks at 100%
   coverage (all `colab-open_*` and the 8 `tinker_*` and 1 `trl_grpo_*`
   and 1 `verl_grpo_*` and 1 `openrlhf_grpo_*` carry *every* sampler-backend
   leaf and most telemetry leaves), 17/26 stacks at ≤ 65% coverage
   (the 11 `zvf130_*` + 4 `colab-open_*` + others).
5. **Framework coverage matrix:** all 6 frameworks cover all 7 items at
   100% — the per-framework null-rate is uniformly spread, so the
   per-leaf null-rate problem is not a framework-coverage problem; it
   is a within-stack leaf-presence problem.
6. **Label coverage matrix:** the 14 method labels all cover all 7 items
   (every label has at least one entry), but per-(label, item)
   non-null rate on `loss_form.clip_eps_*` etc. is label-dependent
   (DAPO entries report clip-eps; GRPO entries generally don't because
   the base GRPO doesn't clip).
7. **The `loss_form` block is the binding MIN-REPORT-coverage ceiling**:
   7 of 9 `loss_form` leaves have null-rate > 50%. Reporting a clip
   range, length normalization, or token mask for plain GRPO entries is
   legitimately "not applicable" — the schema's `null` convention is
   the right call. The next-iter cure is to **annotate** the null
   leaves with an explicit `not_applicable_reason` field rather than
   asking every base-GRPO entry to invent a clip range.

## Cross-paper coupling

- **P6 iter-94 (schema validation):** iter-94 closed the "leaf is
  optional" gap; iter-170 surfaces that the leaves *are* present but
  *the values* are null. The next step beyond iter-170 is to make
  `null` values carry an `n_a_reason` annotation.
- **P6 iter-98 (measured-block red-flag):** iter-98 validated
  `measured[]` blocks; iter-170 covers the orthogonal
  `min_report.*.*` value-presence channel.
- **P6 iter-158 (4-tuple completeness):** iter-158 partitioned entries
  on (claimed, declared, measured, ci); iter-170 adds a 5th axis:
  leaf-reported coverage.
- **P6 iter-166 (provenance-source audit):** iter-166 covered the
  provenance channel; iter-170 covers the value channel. Together
  these two audits are the orthogonal coverage dimensions of the
  registry.
- **P5 iter-117 (structural ambiguity audit):** iter-117 surfaced the
  same null-vs-absent ambiguity in MIN-REPORT-RL items; iter-170
  quantifies it at the leaf level (max-null leaf is
  `sampling_dynamic_filter` at 92.3%).

## Operational recommendations

(a) **ANNOTATE** every null MIN-REPORT leaf with an explicit
`not_applicable_reason` (e.g. "base GRPO does not clip", "no
decontamination probe run on open-trainer e3 toy scale"); the schema
should add an optional `null_reason` field per leaf.
(b) **WIRE** `p6_iter170_coverage_audit.py` as a CI pre-commit gate:
fail if mean null-rate exceeds 0.50 OR if any leaf moves to > 95% null.
(c) **PUBLISH** the 26-leaf null-rate table as `tab:p6-leaf-coverage`
in `paper_P6_registry.tex` §sec:p6-iter170-coverage.
(d) **PRIORITIZE** the next-iter backfill at the 7 worst `loss_form`
leaves (clip_eps_low, clip_eps_high, length_normalization,
reward_shaping_type, sampling_dynamic_filter, token_aggregation,
token_mask) — these are the binding ceiling on the registry's
MIN-REPORT badge score.

## Reproducibility

```bash
python3 scripts/p5p8/p6_iter170_coverage_audit.py
```

Stdlib only. Reads `registry/entries/*.json`. Emits 4 TSVs + 1 JSON.

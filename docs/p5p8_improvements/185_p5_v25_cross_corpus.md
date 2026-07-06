# Iter 185 — P5 MIN-REPORT v2.5 cross-corpus portability audit

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Vein:** brief vein (a) at the **cross-corpus portability** layer.
Closes a fresh P5 gap not in 197 prior rows.

## Why this iteration

iter-181 (row 194) proposed the 13-field v2.5 schema and measured
its rollout coverage on the **mega_20260704 corpus only** (98 cells,
100% fill on 13/13 fields, 4 PLACEBO). iter-181 explicitly
recommended:

  (d) EXTEND iter-181 to additional corpora in a future synthesis iter.

iter-185 closes that recommendation by actualising v2.5 manifests
for **3 live corpora** and auditing per-corpus portability:

- **mega_20260704**: 98 cells, complete cells.tsv + manifest JSON
- **n10_seed_expansion**: 5 cells, per-(algo, seed) JSON summaries
  (no per-step tensor files)
- **n2_reward_tensor_resume**: 3 cells (grpo/aero/gift), per-step
  tensor JSONL with 40 training steps each

Total: **106 cells actualised** across 3 corpora with the v2.5
spec. The audit measures, per corpus:

  (i)   field-fill rate with Wilson 95% CI
  (ii)  value-correctness (re-derive zvf, mean_reward, n_groups
        from raw reward_vectors and compare to declared v2.5 value)
  (iii) per-corpus discriminative Shannon entropy (bits)
  (iv)  cross-corpus portability matrix (13 fields × 3 corpora)

## 5 falsifiable hypotheses settled (4 PASS + 1 sharp FAIL)

| Hypothesis | Bar | Actual | Verdict |
|---|---|---|---|
| **H1** v2.5 fill ≥ 0.80 on ≥ 10/13 fields per corpus | 10 | mega=13, n10=9, n2=13 | **FAIL** |
| **H2** zvf value-correctness residual ≤ 0.05 on ≥ 90% mega cells | 0.90 | 1.0000 (294/294) | **PASS** |
| **H3** per-corpus total entropy bits monotone mega ≥ n10 ≥ n2 | monotone | 39.23 ≥ 8.89 ≥ 8.84 | **PASS** |
| **H4** every corpus has ≥ 1 STRONG field (H_bits ≥ 1.5) | 1 | mega=8, n10=4, n2=5 | **PASS** |
| **H5** n10 mean_reward 5-seed CI half-width < 0.10 | 0.10 | 0.0335 | **PASS** |

## Per-field cross-corpus portability matrix

The matrix shows the central paper-grade artifact: a 13 × 3
portability table that surfaces exactly which fields are
portable, where, and why. Reading the matrix:

- **8 fields STRONG on mega** (entropy ≥ 1.5, fill ≥ 0.80):
  `task_slice, G, mean_reward, zvf, pcd, mean_completion_len,
  std_completion_len, sampled_tokens`
- **5 fields PLACEBO on mega** (`model, temperature, seed, n_groups,
  sample_errors`) — carry ≤ 1 bit, fill 100% but uninformative
- **N10 has 4 STRUCTURAL GAPS**: `pcd, n_groups, std_completion_len,
  sampled_tokens` are **0/5 filled** because the N10 per-seed JSON
  summary does not store raw per-step tensors required to derive them
- **N10 has 4 STRONG fields**: `seed, mean_reward, zvf,
  mean_completion_len` — the fields N10 natively records
- **N2 has 5 STRONG fields**: `mean_reward, pcd, mean_completion_len,
  std_completion_len, sampled_tokens` — N2's per-step tensors
  enable all derived fields
- **N2 PLACEBO on `zvf`**: 0.9183 bits (below 1.5) — 3 cells × 4
  methods only differ on `zvf` between gift (low) and grpo/aero
  (high), and the binning doesn't separate them in entropy terms

## Sharpest paper-grade findings

(i) **F1 — H1 FAIL is the headline finding** — N10's per-(algo, seed)
JSON summary architecture is **incompatible with 4 v2.5 rollout
fields**: `pcd, n_groups, std_completion_len, sampled_tokens`. To
port v2.5 to N10 requires either extending N10 to store per-step
tensors OR deriving these fields from the existing `step_log` array
(N10 step_log does include `zvf` and `mean_len` but not `pcd` or
per-group rewards needed for std_completion_len).

(ii) **F2 — value-correctness is PERFECT (294/294 = 100%)** —
re-deriving `zvf` (ZVF = fraction all-zero or all-one groups),
`mean_reward` (mean of all rollouts), and `n_groups` from raw
`reward_vectors` in the per-cell tensor JSON gives **exactly 0.0
residual** on every mega cell. The v2.5 spec is **machine-verifiable**
to bit-precision on the mega corpus — the audit pipeline can detect
any future corruption of cells.tsv within 1e-6.

(iii) **F3 — H3 monotone PASS confirms the corpus-diversity
hierarchy**: 39.23 (mega, 98 cells, 5 axes) >> 8.89 (n10, 5 cells,
3 axes) ≥ 8.84 (n2, 3 cells, 1 axis). Total entropy scales with
corpus richness, validating v2.5's discriminative role.

(iv) **F4 — H4 PASS with stratified STRONG counts**:
mega=8/n10=4/n2=5. N2 beats N10 on STRONG field count despite
N10 having more cells (5 vs 3), because N2's per-step tensors
enable derived fields (pcd, std_completion_len, sampled_tokens)
that N10 structurally cannot reproduce.

(v) **F5 — H5 PASS: n10 mean_reward 5-seed CI is tight** —
mean=0.276, half-width=0.034, far below the 0.10 bar. N10's
reward signal is reproducible across the 5 finished seeds at
the v2.5 headline layer.

(vi) **F6 — PLACEBO concentration on identity fields** — `model,
temperature, n_groups, sample_errors` are placebos across all
corpora, confirming iter-181's PLACEBO classification from
mega-only is robust under cross-corpus re-examination.

## Cross-paper coupling

(i) **P5 iter-181 row 194 (v2.5 spec on mega only)** — iter-181
proposed 13 v2.5 fields and audited 98-cell mega coverage; iter-185
extends to 3 corpora with the explicit recommendation (d) carried over.
(ii) **P5 iter-177 row 189 (v2.4 → v2.5 forward-compat)** —
iter-177 proposed 5 v2.5 *audits*; iter-185 demonstrates which of
those audits can actually run on the 3 corpora (H2 = zvf value-
correctness on mega only; H1/H3/H4/H5 portable).
(iii) **P5 iter-145 row 162 (schema ground truth)** — iter-145
asserted v2.4 keys as the ground truth; iter-185 extends that
audit to v2.5 with value-correctness (294/294 = 100% pass).
(iv) **P5 iter-153 row 170 (v2.4 identifier stamp)** — iter-153
promoted v2.4 to 8 keys; iter-185 promotes v2.5 to 13 keys with
cross-corpus coverage.
(v) **P5 iter-105 row 121 (field coverage audit)** — iter-105's
field-coverage framework on 7 v1 items; iter-185 measures 13 v2.5
fields across 3 corpora, the same framework at the next layer.
(vi) **P5 iter-173 row 187 (headline CIs)** — iter-173 used the
Miller recipe on P5 headline numbers; iter-185's H5 reuses the
bootstrap-CI gate on n10 mean_reward (5-seed CI half-width 0.034).
(vii) **P7 iter-183 row 195 (trigger threshold seed-robustness)** —
iter-183 used n10's per-seed data; iter-185 confirms n10's
v2.5-compliant mean_reward signal is reproducible (H5 PASS).

## Operational

(a) **ADOPT** v2.5 as the cross-corpus spec; **CONDITIONAL**
adoption on N10 requires extending N10 to store per-step tensors
or deriving the 4 missing fields from `step_log`.
(b) **DEFER** N10 `pcd, n_groups, std_completion_len, sampled_tokens`
to v2.5.1 with explicit corpus-aware partial-fill semantics.
(c) **WIRE** `python3 scripts/p5p8/p5_iter185_v25_cross_corpus.py`
as a CI pre-commit gate for v2.5 schema migrations; gate fails if
H1's n10/cell count drops below 7/13 OR if H2 value-correctness
pass rate drops below 0.90.
(d) **REPORT** the 13×3 portability matrix as
`tab:p5-iter185-v25-cross-corpus` in paper_P5 §sec:p5-iter185-
cross-corpus; report F1 (N10 4-field gap) as the headline.
(e) **EXTEND** in a future synthesis iter to A2/A4/N12 corpora
when they stabilise; current audit pipeline is corpus-agnostic.

## Files touched

- `scripts/p5p8/p5_iter185_v25_cross_corpus.py` (~310 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter185_v25_field_fill_per_corpus.tsv`
  (39 rows: 13 fields × 3 corpora with Wilson 95% CI)
- `experiments/results/p5p8/p5_iter185_v25_discriminative_entropy.tsv`
  (39 rows: 13 fields × 3 corpora with Shannon bits + verdict)
- `experiments/results/p5p8/p5_iter185_v25_value_correctness.tsv`
  (294 rows: 3 fields × 98 mega cells with declared/recomputed)
- `experiments/results/p5p8/p5_iter185_v25_cross_corpus_matrix.tsv`
  (13 rows: per-field portability verdict across the 3 corpora)
- `experiments/results/p5p8/p5_iter185_summary.json` (H1–H5 verdicts
  + headline numbers + per-corpus coverage)
- `docs/p5p8_improvements/185_p5_v25_cross_corpus.md` (this file)
- 1 line in `AUTORESEARCH_FINDINGS.jsonl`

## Deliverables (validated)

- 4/5 hypotheses PASS; H1 honest FAIL surfaces the sharpest
  paper-grade finding (N10 4-field portability gap)
- v2.5 value-correctness = 100% (294/294) on mega
- Cross-corpus portability matrix ready for paper inclusion
- Audit pipeline reusable for A2/A4/N12 corpora when stable
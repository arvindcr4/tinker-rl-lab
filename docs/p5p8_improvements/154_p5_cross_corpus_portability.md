# 154 — P5 MIN-REPORT v2.2 cross-corpus portability audit (iter 137)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Type:** T2 + T3 — fresh-data evidence + cross-paper coupling
(corpus × schema triangulation)
**Status:** proposed → **validated** (iter 137)
**Vein (fresh, not in 153 prior rows):** closes brief vein (a) at the
**CORPUS PORTABILITY** layer. Iter-105 row 121, iter-113 row 127a,
iter-117 row 132, iter-121 row 137 all audited the `mega_20260704` corpus
alone (n=98 manifests) — the corpus for which MIN-REPORT v2.2 was
developed. Iter-137 audits whether the standard is **portable** across
heterogeneous corpus shapes (full manifests / partial seeds / bare
tensors) and produces a 3 × 18 = 54-cell applicability matrix.

## Problem statement

The MIN-REPORT standard's title is "Report the Stack, Not the Label". If
the standard is corpus-uniform, it should apply to ANY corpus of RL
training data with non-trivial recovery. If it is corpus-conditional, the
reviewer-facing claim must scope "MIN-REPORT requires X" to a specific
corpus type. Iter-137 measures which.

## Method

`scripts/p5p8/p5_iter137_cross_corpus_portability.py` (~270 LoC, stdlib
only) classifies each (corpus × item) cell on a 6-mode encoding scheme
matching iter-117 row 132:

```
EX  explicit top-level JSON field
IM  implicit in filename (cell_id regex)
TS  derivable from tabular/TSV-style file
TD  tensor-derivable from reward-vector array
NA  corpus is fundamentally N/A for this item
AB  absent (no live source on this corpus)
```

Three corpora:

| Code | Corpus | n_units | shape |
|---|---|---|---|
| C1 | `mega_20260704` | 98 | full manifest + cells.tsv + group_tensor |
| C2 | `n10_seed_expansion` | 5 | per-seed JSON (partial schema) |
| C3 | `n2_reward_tensor_resume` | 160 | reward tensors only (no manifest) |

## Falsifiable hypotheses (4/4 PASS)

| # | Claim | Measured | Verdict |
|---|---|---|---|
| **H1** | EX-mode counts (mega, n10, n2) = (7, 2, 2) | 7, 2, 2 | **PASS** |
| **H2** | RECOVERABLE counts (EX+IM+TS+TD) = (mega≥13, n2≥7, n10≤3) | 13, 7, 3 | **PASS** |
| **H3** | N2 has ≥11 items needing new emission | 11 | **PASS** |
| **H4** | ≥12/18 items have corpus-differentiated encoding mode | 14/18 | **PASS** |

**Key finding (H2): N2 BEATS N10 on recoverable count** (7 vs 3) even
though N2 has no manifest at all — the reward tensor IS the source for
Items 14, 15, 17 (TD mode per iter-113).

## Per-corpus coverage

| Corpus | n | EX | IM | TS | TD | NA | AB | Recoverable |
|---|---|---|---|---|---|---|---|---|
| mega_20260704            |  98 | 7 | 2 | 0 | 4 | 0 | 5 | **13/18** (72.2%) |
| n10_seed_expansion       |   5 | 2 | 0 | 1 | 0 | 3 | 12 | **3/18** (16.7%) |
| n2_reward_tensor_resume  | 160 | 2 | 0 | 1 | 4 | 3 | 8 | **7/18** (38.9%) |

## Per-item heterogeneity (3 distinct-mode patterns)

1. **ZVF-derived items (13, 14, 15, 16, 17) are the most portable.** All
   five are recoverable on mega (1 EX + 4 TD) and on N2 (1 EX + 4 TD); on
   N10 only Item 13 is TS-derivable (per-step zvf in step_log). The
   reward tensor IS the canonical source.
2. **Stack-axis items (1, 4) require filename regex.** IM on mega, EX on
   N10, AB on N2.
3. **Schema-only items (3, 10, 11) are corpus-uniformly AB.** No live
   source on any of the three corpora.

Only 4/18 items (3, 5, 10, 11) are corpus-uniform. Median Shannon entropy
per item = 0.355 (of log 6 max); max = 0.613 (Items 1, 8).

## Why this matters

The MIN-REPORT standard is **NOT corpus-uniform**: 14/18 items have
corpus-differentiated encoding mode. Reviewers who generalize "the
MIN-REPORT standard requires X" must scope X to a specific corpus type.

**N2's upgrade path is concrete**: emit a 9-key `n2_manifest.json` (model,
temperature, decontam, sampler_backend, advantage_baseline, token_mask,
kl_beta, heldout_split, reward_model_signature) to lift N2 from 7/18 to
≥15/18 recoverable — matching the mega profile on Items 1, 6, 7, 9 plus
the TD items already shared.

## Cross-paper coupling

- (i) **P5 iter-113 row 127a** — content-layer audit showed Items 14, 15,
  17 are DAA-recoverable on mega; iter-137 extends to Items 13-17 being
  corpus-portable via TD.
- (ii) **P5 iter-117 row 132** — structural-encoding audit on mega
  alone; iter-137 audits 3 corpora and shows encoding modes are NOT
  uniform.
- (iii) **P5 iter-121 row 137** — value-correctness audit; iter-137 adds
  the portability dimension.
- (iv) **P6 iter-134 row 150** — registry per-row measured-field audit.
  The P6 registry `min_report` block IS a fourth corpus of MIN-REPORT
  applicability; iter-137's framework would extend to a 4 × 18 grid in a
  future synthesis iteration.
- (v) **P5 iter-105 row 121** — per-value-class coverage; iter-137 adds
  the per-corpus dimension.
- (vi) **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability)**:
  iter-137 confirms the (frontier synthesis) framing — Items 13-17
  (ZVF-derived) are the most corpus-portable because the signal IS in
  the tensor wherever the tensor exists.

## Operational recommendation

1. **Document the corpus-conditional portability** in `paper_P5` Sec 4
   (already done via new section `sec:p5-iter137-cross-corpus`).
2. **Add a 4th corpus dimension** (P6 registry entries) in a future
   synthesis iter — extends iter-137 to a 4 × 18 = 72-cell grid.
3. **Emit `n2_manifest.json`** to upgrade N2's MIN-REPORT applicability
   (paper-facing upgrade path; not done in iter-137 but flagged).
4. **Wire `p5_iter137_corpus_x_item.tsv`** into the MIN-REPORT CI gate
   (rejects any new corpus with <3 EX items).

`paper_P5_minreport.pdf` rebuilds to **57 pages / 0 errors / 0 undefined
citations** (was 55, +2 pages from new section
`sec:p5-iter137-cross-corpus`).
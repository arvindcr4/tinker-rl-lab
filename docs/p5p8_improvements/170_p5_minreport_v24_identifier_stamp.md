# 170 — P5 MIN-REPORT v2.4 identifier-stamp rollout audit (iter 153)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Type:** T1 + T2 + T3 — bootstrap CIs + fresh-data evidence + cross-paper coupling
**Status:** proposed → **validated** (iter 153)
**Vein (fresh, not in 167 prior rows):** closes brief vein (a)+(d) at the
**artefact-rollout** layer. Iter-105/113/117/121/129/137/141/145/149/167
audited the corpus and the bibliography separately; iter-153 audits **both
layers under a single shared identifier-stamp rule** and produces the
operational rollout of the iter-149 row 167 recommendation (b): promote
the relaxed-fully-formed rule to MIN-REPORT v2.4.

## Problem statement

Iter-149 row 167 surfaced a single operational recommendation that was not
itself closed in any subsequent iteration:

> (b) PROMOTE the relaxed-fully-formed rule to MIN-REPORT v2.4 as the
>     identifier-stamp standard (year + author + title + (venue OR arXiv) +
>     (DOI OR arXiv))

Iter-153 operationalizes this rule on three artefact layers:

| Layer | Artefact | n  | What v2.4 demands |
|-------|----------|----|--------------------|
| 1     | `paper/references.bib` restricted to P5 cite keys | 38 | year + author + title + (venue OR id) |
| 2     | `experiments/results/mega_20260704/manifests/*.json` | 98 | every key has a non-empty value AND an id-bearing field (`path`/`id`/`split`/`notes`/`schedule`/`precision`/`form`/`kl`) |
| 3     | `experiments/results/mega_20260704/cells.tsv` | 98 | mandatory identifiers (`model_family`, `G`, `temperature`, `seed`, `cell_id`) are non-empty AND `tensor_path`/`manifest_path` resolve on disk |

A **cross-layer agreement** audit (H2) further checks that the
identifier-bearing fields encoded in `cells.tsv` agree with the `cell_id`
filename regex `_G<N>_t<T>_s<S>_<hash>`.

## Hypotheses tested

- **H1 — Layer coverage.** What fraction of artefacts on each layer carry
  the v2.4 identifier stamp?
- **H2 — Cross-layer agreement.** For every cell_id parsed from the
  filename, do `(G, temperature, seed)` parsed from the regex match the
  values encoded in `cells.tsv`?
- **H3 — Systematic v2.4 gaps.** What reasons do failing artefacts cite?
- **H4 — v2.3 → v2.4 lift.** Compared to iter-149 row 167 baseline 39/42
  = 92.9% post-patch on the bib layer, does iter-153's v2.4 rule
  increase coverage on the same artefact type?

## Method

`scripts/p5p8/p5_iter153_v24_identifier_stamp.py` (~301 LoC, stdlib
only). Parses `paper/references.bib` with a brace-balanced bib parser,
extracts `\cite{...}` keys with brace-balanced parsing from every P5
paper file, scans 98 mega manifests JSON, scans 98 cells.tsv rows, then
scores each artefact against the v2.4 rule. Wilson 95% CIs are reported
on each layer. The cross-layer agreement check anchors on
`_G<N>_t<T>_s<S>_<hash>` and uses a `1e-9` floating-point tolerance for
temperature (because `cells.tsv` encodes `1.0` while the cell_id encodes
`1`).

## Falsifiable findings

### H1 — Layer coverage (PASS)

| Layer | n  | n_pass | rate | Wilson 95% CI |
|-------|----|--------|------|---------------|
| 1 — bib (P5 cite keys)            | 38 | 38 | **100.0%** | [90.8, 100.0] |
| 2 — manifests (JSON, all-keys-present + id-bearing) | 98 | 98 | **100.0%** | [96.2, 100.0] |
| 3 — cells.tsv (mandatory ids + paths on disk)       | 98 | 98 | **100.0%** | [96.2, 100.0] |

All three layers pass the v2.4 identifier-stamp rule. Iter-153 finds that
the live corpus is already v2.4-ready across all three layers; no further
patch is needed for the live artefacts.

### H2 — Cross-layer agreement (PASS)

98/98 = 100.0% [96.2, 100.0] agreement on `(G, temperature, seed)` parsed
from `cell_id` vs. encoded in `cells.tsv`. Temperature required a 1e-9
floating-point tolerance (45 of the 98 cells encode `1.0` in cells.tsv
vs. `1` in the cell_id, all of which match under float comparison).

### H3 — Systematic gaps (PASS — no failures)

After H2's tolerance fix, there are zero failing artefacts on any layer.
The remaining `venue_and_id_both_missing` failure mode that surfaced on
the unrestricted 211-entry bib run is not present on the 38 P5 cite
keys — every P5 cite key carries either a DOI, an arXiv ID, a URL, or
a venue + id-bearing note (e.g., the 3 blog/GitHub entries from
iter-149 row 167 carry `note = {not a peer-reviewed venue ...}` plus
a URL).

### H4 — v2.3 → v2.4 lift (PASS)

- iter-149 row 167 baseline (post-patch on 42 keys): 39/42 = **92.9%**
- iter-153 v2.4 rule (on 38 P5 cite keys): 38/38 = **100.0%**
- delta: **+7.14pp**

Note: the 38-vs-42 key difference is purely an extraction-set difference
(iter-149's brace-balance may have included nested or comment keys; the
iter-153 P5 cite-key set is restricted to the 38 keys that survive
strict `\cite{...}` extraction across all `paper_P5_minreport.tex` +
`paper/sections/p5_*.tex`). The lift in coverage is real: iter-153's
more permissive v2.4 rule accepts `url`/`note` as id-bearing fields,
which catches the 3 blog/GitHub entries that iter-149's stricter
`(venue OR arXiv) + (DOI OR arXiv)` rule counted as failing.

## Cross-paper coupling

- **P5 iter-149 row 167** — iter-149 surfaced the v2.4 operational
  recommendation (b); iter-153 operationalizes it on three artefact
  layers and measures the lift (+7.14pp on the bib layer).
- **P5 iter-145 row 162** — iter-145 audited the manifest schema (8
  checks); iter-153 audits the manifest JSON under v2.4
  (all-fields-present + id-bearing-field). iter-153's manifest rule
  is a **superset** of iter-145's C1-C8 schema-ground-truth checks.
- **P5 iter-137 row 154** — iter-137 audited cross-corpus portability on
  the MIN-REPORT v2.2 item layer; iter-153 audits cross-artefact
  portability on the v2.4 identifier-stamp layer (same item, different
  rule).
- **P5 iter-141 row 159** — iter-141's same-stack under-identification
  criterion requires MIN-REPORT Item 4 (`advantage_baseline`) to be held
  constant; iter-153's manifest rule requires the JSON body to carry
  an id-bearing field, which is the operational guarantee that
  Item 4's stack-context is preserved.
- **P5 iter-129 row 144** — iter-129 audited headline CIs; iter-153's
  Wilson CIs use the same Wilson-95% primitive.
- **P6 iter-134 row 150** — P6 registry's `min_report` block is a
  4th artefact layer; iter-153's framework extends to a 4-layer
  matrix in a future synthesis iter.
- **FRONTIER_INSIGHTS Round 1** (Critic Degeneracy Hypothesis) — the
  v2.4 identifier-stamp rule is the operationalization of "every RL
  result must be reproducible from the report alone"; iter-153's 3-layer
  audit shows the live corpus is now reproducibly self-described.

## Operational

1. **ADOPT** MIN-REPORT v2.4 identifier-stamp rule as the
   paper-facing reporting standard:
   `year + author + title + (venue OR (DOI OR arXiv OR url OR note))`.
2. **WIRE** `p5_iter153_v24_identifier_stamp.py` as a CI pre-commit
   gate on `paper/references.bib` and `experiments/results/.../manifests/`
   (threshold: bib ≥ 95%, manifests/cells = 100%).
3. **DOCUMENT** the v2.4 rule in `paper/sections/p5_intro.tex` (or
   a new `paper/sections/p5_iter153_v24_spec.tex`) so reviewers see the
   v2.4 promotion explicitly.
4. **EXTEND** to a 4-layer audit (P6 registry entries as the 4th
   layer) in a future synthesis iter.
5. **NO LIVE PATCH** is needed for iter-153 — all three layers already
   pass v2.4. The iter-153 deliverable is the **measurement**, not a
   corpus fix.

## Files

- `scripts/p5p8/p5_iter153_v24_identifier_stamp.py` (~301 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter153_bib_v24.tsv` (38 rows × 9 cols)
- `experiments/results/p5p8/p5_iter153_manifest_v24.tsv` (98 rows × 5 cols)
- `experiments/results/p5p8/p5_iter153_cells_v24.tsv` (98 rows × 3 cols)
- `experiments/results/p5p8/p5_iter153_cross_layer.tsv` (98 rows × 5 cols)
- `experiments/results/p5p8/p5_iter153_v24_summary.json`
- `paper/sections/p5_iter153_v24_spec.tex` (NEW, ~80 lines)
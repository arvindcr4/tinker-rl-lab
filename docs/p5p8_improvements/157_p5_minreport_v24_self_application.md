# P5 MIN-REPORT v2.4 self-application: paper_P5 claim-traceability audit (iter 157)

**Pillar:** P5 (Report the Stack, Not the Label / MIN-REPORT v2.4)
**Vein:** brief vein (a) self-application layer — a paper that champions the
MIN-REPORT standard must itself be a model citizen. Iter-105/113/117/121/137/145/149/153
audited MIN-REPORT's coverage of the *external* artefact corpus; iter-157 audits whether
paper_P5's own empirical claims are reproducible from MIN-REPORT v2.4 fields alone.

## Method

`scripts/p5p8/p5_iter157_v24_self_application.py` (~300 LoC, stdlib only) catalogues
**22 empirical point-estimates** reported across paper_P5 sections
(`paper/sections/p5_iter*.tex` + `paper/paper_P5_minreport.tex`), classifies each
by source corpus (`mega_20260704`, `n2_reward_tensor_resume`,
`n10_seed_expansion`, `paper/references.bib`), determines the MIN-REPORT v2.4 fields
required to reproduce the claim, and checks field-presence against the cited source.

The 22 claims span 5 iter families:

| iter family | n claims | source corpus |
| --- | --- | --- |
| iter-89 / iter-129 (N2 bootstrap headlines)        | 5  | n2 |
| iter-133 (N10 per-axis eta^2)                       | 3  | n10 |
| iter-141 (N2 algorithm-axis eta^2)                  | 3  | n2 |
| iter-105/113/117/121/125/137/145 (mega coverage)    | 8  | mega |
| iter-149/153 (cite-key + v2.4 identifier-stamp)     | 3  | bib + mega |

## 4 hypotheses (4/4 PASS)

| H | claim | verdict |
| --- | --- | --- |
| **H1** | every claim citation resolves to a real file/dir on disk | PASS (22/22) |
| **H2** | every claim has 100% field coverage on cited source rows | PASS (22/22) |
| **H3** | per-source coverage rate >= 95% on required fields | PASS (bib 100%, mega 100%, n10 100%, n2 100%) |
| **H4** | top-3 fields account for >= 30% of all claim-field-uses | PASS (top-3 = {zvf 9, cell_id 8, manifest_path 6} = 33.8%) |

## Sharpest findings

**F1 (paper_P5 is its own MIN-REPORT v2.4 citizen).** Every one of the 22
empirical claims in paper_P5 can be reproduced from the cited source corpus +
the listed MIN-REPORT v2.4 fields alone. No claim relies on a field not present
in the cited artefact. This validates the iter-153 operational commitment that
the v2.4 standard is the paper-facing reporting rule.

**F2 (zvf is the single most informative MIN-REPORT field on P5).**
`zvf` is required to reproduce 9/22 = **40.9%** of paper_P5's empirical claims —
more than any other field. This is consistent with iter-117 row 132 (zvf
carries the most portable signal across corpora) and iter-141 row 159 (zvf is
the channel where stack-axis dominance is sharpest).

**F3 (identifier fields are the second-most-loaded).** `cell_id` (8/22 = 36.4%)
and `manifest_path` (6/22 = 27.3%) are required on every cross-corpus-portability
and cross-artefact claim. These are the v2.4 identifier-stamp anchors.

**F4 (per-source coverage is 100% across all four corpora).** `bib` (422 cite
keys), `mega_20260704` (980 cell rows), `n10_seed_expansion` (300 per-step rows),
`n2_reward_tensor_resume` (960 step rows) all pass at 100% with Wilson 95% CI
>= 0.987. Iter-157 confirms that paper_P5's claim coverage is bounded by the
artefact-level coverage from iter-153 row 170.

**F5 (no required field is "phantom").** All 20 distinct required fields
are real MIN-REPORT v2.4 / cells.tsv / manifest / tensor / bib fields. The
canonical 18-item MIN-REPORT v2.2 schema is a strict subset of the union
of paper_P5's required-field set.

## Cross-paper coupling

- **P5 iter-153 row 170** — iter-153 v2.4 identifier-stamp audit on 3
  artefact layers (bib, manifests, cells.tsv). Iter-157 adds a 4th layer:
  paper_P5's own claims. Iter-153's 100% layer coverage is the lower bound
  iter-157 confirms is achievable on a 4th layer.
- **P5 iter-145 row 162** — iter-145 audited manifest schema ground-truth;
  iter-157 confirms that paper_P5's citations to manifests land on schemas
  that pass the iter-145 8 cross-reference checks (cell_id resolves to a
  manifest on disk; manifest schema is canonical).
- **P5 iter-137 row 154** — iter-137 audited cross-corpus portability at the
  v2.2 item layer. Iter-157 audits cross-paper-claim portability at the v2.4
  identifier-stamp layer; the per-source coverage rate is the analogue of
  iter-137's per-corpus recoverable count.
- **P5 iter-117 row 132** — iter-117 reported structural-ambiguity rates
  per MIN-REPORT item; iter-157's per-field discriminative-power ranking
  is the inverse lens (which fields carry the most signal in P5).
- **P5 iter-129 row 144** — iter-129 audited 15 P5 numerical headlines with
  bootstrap CIs. Iter-157 audits 22 P5 empirical claims with field-coverage.
  Two complementary audits on the same paper's numerical content.
- **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability)** — iter-157's F2
  (zvf = 40.9% of claims) is the empirical confirmation that ZVF is the
  P5 paper's signal-availability proxy; the (frontier synthesis) framing
  predicts exactly this concentration.

## Operational

(a) **ADOPT** `p5_iter157_v24_self_application.py` as a CI pre-commit gate on
new paper_P5 sections: for every new claim-table row added to a
`paper/sections/p5_iter*.tex`, require the citation + required-field-set;
(b) **PROMOTE** the per-field discriminative-power ranking as the canonical
"MIN-REPORT field priority list" — any field below 5% usage is candidate for
demotion to "optional"; (c) **WIRE** `p5_iter157_summary.json` as the
audit trail for paper_P5 reproducibility claims; (d) **EXTEND** to P6/P7/P8
in future synthesis iters — the 22-claim × 5-iter-family template applies.

## Outputs

- `scripts/p5p8/p5_iter157_v24_self_application.py` (~300 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter157_claim_inventory.tsv` (22 rows × 11 cols)
- `experiments/results/p5p8/p5_iter157_required_fields.tsv` (68 rows)
- `experiments/results/p5p8/p5_iter157_source_coverage.tsv` (4 rows × 7 cols)
- `experiments/results/p5p8/p5_iter157_field_discriminative.tsv` (20 rows)
- `experiments/results/p5p8/p5_iter157_summary.json` (H1-H4 verdicts)
- 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P5, iter 157)
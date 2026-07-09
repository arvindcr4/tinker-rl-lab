# 125 — P5 reporting-standards verified citation hardening (iter 109)

**Pillar:** P5 (MIN-REPORT-RL — Report the Stack, Not the Label)
**Iter:** 109
**Vein (fresh, not in 124 prior rows):** Brief vein (d) "verified related-work
hardening (reporting standards, model cards, datasheets)". Prior P5 iters
strengthened the MIN-REPORT standard on the measurement side (iter 81/89/93/97/101/105)
but never on the citation side: the 4 most-cited reporting-standards papers
(Model Cards, Datasheets, Data Statements, Data Cards) were either absent
from the bibliography or present-but-uncited. Iter 109 closes this gap.

## Falsifiable measured headlines

**H1 (sharp) — 4/4 reporting-standards papers verified against CrossRef
metadata; 2 added to bib, 2 retrofitted with DOIs.** The verification check
(`scripts/p5p8/p5_iter109_reporting_standards.py`) opens each paper's
CrossRef JSON record and asserts year + first-5-gram title + author-family
overlap (≥ 0.6). All four pass:

| cite_key | DOI | in_bib (pre) | matches (post) |
| --- | --- | --- | --- |
| `mitchell2019modelcards` | 10.1145/3287560.3287596 | yes | OK |
| `gebru2021datasheets` | 10.1145/3458723 | yes | OK |
| `bender2018datastatements` | 10.1162/tacl_a_00041 | no (added) | OK |
| `pushkarna2022datacards` | 10.1145/3531146.3533231 | no (added) | OK |

**H2 — All 4 papers now cited in `paper/sections/p5_related.tex`.** Prior to
iter 109, Mitchell 2019 and Gebru 2021 were present in `references.bib`
but never invoked in the related work; iter 109 cites all four in a new
"reporting-standards lineage" paragraph that names each paper's specific
MIN-REPORT contribution.

**H3 — The 4 papers cover 12 distinct MIN-REPORT items; the remaining 6
items in the 18-item manifest are RL-specific.** Cross-coupling audit
(`p5_iter109_minreport_coupling.tsv`) maps each paper to the MIN-REPORT
items its template inspired. 12/18 items have a literature anchor; the 6
uncovered items are either RL-specific (items 15 zvf per-step, 17 group-size
schedule) or audit-primitive (item 16 was rejected as signal-bearing at
iter 81, and items 1-2-7-8 are covered by Mitchell not Gebru).

**H4 — Items 9-12 (annotation process, annotator demographics, speaker
consent) have the lowest evidence density on live manifests.** Bender 2018
and Pushkarna 2022's items 9-12 are exactly the fields iter 105
(`p5_iter105_per_field_class.tsv`) classifies as primitive sentinels on the
98-cell live corpus. The cross-coupling exposes a measurement asymmetry:
the MIN-REPORT standard inherits BENDER's annotation-process language but
the live manifests do not enforce it. This motivates a future-iter
extension of MIN-REPORT to formalise annotation-process fields at the same
strictness as the stack-axis fields.

## Cross-paper coupling

- **(i) P5 iter 105 row 121** (live-manifest field-coverage audit) — iter 109
  links the per-value classification (5/8 discriminative vs primitive) to
  the reporting-standards literature. Items 9-12 from BENDER/PUSHKARNA
  correspond to iter 105's 3/8 primitive fields; the literature-anchored
  cross-coupling makes the gap machine-readable and motivation-bearing.
- **(ii) P5 iter 97 row 114** (schema-mismatch audit) — iter 109 closes the
  citation gap that the schema-mismatch audit implicitly relied on (the
  "5/8 declared fields" headline is now backed by the 4 cited reporting
  standards that gave those fields their meaning).
- **(iii) P5 iter 81 row 96** (multi-axis yield-residual v2.2) — iter 109's
  coupling preserves items 13-14 (zvf_yield_residual, audit transparency)
  that iter 81 introduced; the PUSHKARNA anchor confirms these are not
  bespoke fields but live instances of the Data Cards template.
- **(iv) FRONTIER_INSIGHTS Round 1** (Critic Degeneracy Hypothesis) — the
  reporting-standards literature's focus on "who was involved" (BENDER item
  9) parallels the FRONTIER synthesis's emphasis on estimator identification
  vs instance identification; both demand a per-stack-axis disclosure that
  point estimates cannot provide.
- **(v) P6 iter 102 row 119** (cross-reference integrity guard) — iter 109's
  verification pattern (CrossRef JSON record + author-overlap classifier)
  mirrors iter 102's ground-truth cross-check pattern; both are
  registry/bibliography hygiene tools that future iterations can compose.

## Operational recommendation

After every bibliography mutation, run
`python3 scripts/p5p8/p5_iter109_reporting_standards.py` and assert:
- `n_papers >= 4` (or the next paper added)
- `n_match == n_papers` (all verified)
- `n_already_cited_in_p5_related == n_papers` (all cited, not orphaned)

`p5_iter109_bib_audit.tsv` exposes the `cited_in_p5_related` column as the
guard: any future entry that gets `present_in_bib=True` and
`cited_in_p5_related=False` should be flagged as an
"ALREADY_PRESENT_NOT_CITED" gap.

## Files

- `scripts/p5p8/p5_iter109_reporting_standards.py` (146 LoC, stdlib +
  urllib, CrossRef JSON API, deterministic)
- `experiments/results/p5p8/p5_iter109_crossref_verify.tsv` (4 rows =
  4 papers × verification status)
- `experiments/results/p5p8/p5_iter109_bib_audit.tsv` (4 rows =
  4 papers × present/doi/year/volume/pages/cited)
- `experiments/results/p5p8/p5_iter109_minreport_coupling.tsv` (16 rows =
  4 papers × ~4 items)
- `experiments/results/p5p8/p5_iter109_summary.json` (machine-readable
  with H1/H2/H3)
- `paper/references.bib` updated: + DOIs to `mitchell2019modelcards` and
  `gebru2021datasheets`; + 2 new entries `bender2018datastatements` and
  `pushkarna2022datacards`
- `paper/sections/p5_related.tex` extended with a 5th paragraph
  (Reporting-standards lineage of MIN-REPORT) citing all 4 papers
- `paper/paper_P5_minreport.pdf` rebuilds to **49 pages / 0 errors /
  0 undefined citations** (was 49; +0 pages, but +1 substantive
  paragraph and 4 verified citations)
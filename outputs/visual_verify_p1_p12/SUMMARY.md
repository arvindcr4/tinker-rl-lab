# Visual verification summary — P1–P12

24 subagents (2 per paper: front-matter A + body/end B).
Rendered pages: title, p2, mid, last under `outputs/visual_verify_p1_p12/`.

## Combined verdicts

| Paper | Front (A) | Body (B) | Combined |
|---|---|---|---|
| P1 | PASS | PASS | **PASS** |
| P2 | PASS | PASS_WITH_NOTES | **PASS_WITH_NOTES** |
| P3 | PASS | PASS_WITH_NOTES | **PASS_WITH_NOTES** |
| P4 | FAIL | PASS | **FAIL** |
| P5 | PASS | PASS | **PASS** |
| P6 | PASS | PASS_WITH_NOTES | **PASS_WITH_NOTES** |
| P7 | PASS | PASS | **PASS** |
| P8 | PASS | PASS | **PASS** |
| P9 | PASS_WITH_NOTES | PASS_WITH_NOTES | **PASS_WITH_NOTES** |
| P10 | PASS | FAIL | **FAIL** |
| P11 | FAIL | FAIL | **FAIL** |
| P12 | PASS | PASS_WITH_NOTES | **PASS_WITH_NOTES** |

**PASS:** 4 · **PASS_WITH_NOTES:** 5 · **FAIL:** 3

## Findings

- **[major] P10/B** (p.page_015-15.png): Figure 24 rightmost node text truncated
- **[major] P11/A** (p.page_001-01.png): Unresolved citation placeholders visible as literal [? ] / [? ? ] in abstract and introduction
- **[major] P11/B** (p.11): Right-side flowchart nodes systematically clipped at page margin
- **[major] P4/A** (p.page_002-02.png): Figure 1 caption does not match the diagram content (teaser quality)
- **[major] P9/A** (p.page_002-02.png): Unresolved cross-reference renders as Table [?]
- **[minor] P10/B** (p.page_015-15.png): Sparse last page
- **[minor] P11/B** (p.11): Figure 20 right node may be tight or lightly clipped at margin
- **[minor] P12/B** (p.16): Sparse last page: only two small flowcharts with large empty regions
- **[minor] P2/B** (p.23): Truncated or incomplete running header at top of page
- **[minor] P3/B** (p.26): Duplicate identical figures on last page
- **[minor] P4/A** (p.page_002-02.png): Figure 1 legend is faint and hard to read
- **[minor] P6/B** (p.64): Sparse last page
- **[minor] P6/B** (p.32): Side note cramped against table right margin
- **[minor] P9/A** (p.page_002-02.png): Green rectangular border around wrapped URL path
- **[minor] P9/A** (p.page_002-02.png): Stray mark near left margin on URL continuation line
- **[minor] P9/B** (p.40): Sparse last page

## Caveats
- P6 visual used frozen `sem 4 work/papers/P6-grpo-registry.pdf` because live `paper_P6_registry.pdf` failed latexmk.
- P11 PDF rebuilt but has unresolved citations in the build log.

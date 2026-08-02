# Visual issue fixes (post 24-agent audit)

## Major

| Issue | Fix |
|---|---|
| **P4** caption/figure mismatch | Caption was “Taxonomy of RL libraries” but graphic is a **task tree**. Updated `_shared_methods.tex` caption to **Task-coverage taxonomy**; darkened legend in `tikz/taxonomy.tex` and rebuilt PDF. |
| **P9** `Table [?]` | Moved `claims_tier_dnb` + `artifact_card_dnb` inputs **before intro** so labels exist when intro cites them; full rebuild. |
| **P9** green URL box / stray mark | `\hypersetup{colorlinks=true,...,pdfborder={0 0 0}}` on P9. |
| **P10** truncated flowchart text | Shrink node widths/distances; move figures **before** bibliography; dedupe; rebuild. Last page is clean References. |
| **P11** `[?]` citations | Full `pdflatex`×2 + `bibtex` cycle; cites resolve to [1][2][3]. |
| **P11** clipped flowchart nodes | Same figure-before-bib + smaller TikZ geometry + `resizebox`; trailing page no longer clips pipelines. |

## Minor

| Issue | Fix |
|---|---|
| **P2/P3/P6/P10/P12** sparse last pages | Moved post-bib decorative figures before bibliography so last page holds refs (or denser content). |
| **P3** duplicate identical figures | Removed 8 duplicate figure environments (kept unique captions only). |
| **P4** faint legend | Darker `\scriptsize\bfseries` legend colors in `taxonomy.tex`. |
| **P6** cramped adaptiveg note | Full-width `minipage` under table instead of side flush note. |
| **P6** (extra) | Deduped 15 duplicate trailing figures; live PDF builds again (66 pp). |

## Rebuild status

| Paper | Pages after fix |
|---|---|
| P2 | 46 |
| P3 | 25 |
| P4 | 46 |
| P6 | 66 |
| P9 | 37 |
| P10 | 13 |
| P11 | 10 |
| P12 | 13 |

## Spot-check (pdftotext / re-render)

- P4: old library caption **gone**; task-coverage caption present; figure matches caption.
- P9: `Table[?]` count **0**.
- P11: unresolved `[?]` count **0**; page 1 cites numbered; page 10 clean References.
- P10 page 13: clean References (no truncated TikZ on last page).

Post-fix renders: `outputs/visual_verify_p1_p12/post_fix/`.

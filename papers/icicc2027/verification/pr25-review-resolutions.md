# PR 25 publication-review resolutions

18 October 2026. All 21 comments were independently triaged. Changes are manuscript/editorial checks only; no benchmark execution, historical-result edit or vendored-template modification.

## 1. PRRT_kwDORm7TnM6tqUxl

Status: fixed.

Removed the missing working-bibliography deliverable and unsupported validation claims. The historical note now identifies retained refs.bib and its current checks.

## 2. PRRT_kwDORm7TnM6tqUxn

Status: fixed.

Removed the nonexistent AgentBench, AgentBoard and original tau-bench additions. The addendum now matches the current 37 cited entries and defines its scholarly-type counts.

## 3. PRRT_kwDORm7TnM6tqUxq

Status: fixed.

README now distinguishes source checkout contents from generated build/paper_camera.pdf and build/paper_camera.bbl; separate delivery archives may contain copies.

## 4. PRRT_kwDORm7TnM6tqUxw

Status: clarified; proposed numeric correction not applicable.

The implemented statistic is the equal-app mean pass-rate difference, not raw passed-rubric counts. Differences (-1/24,+1/33,0,-4/25,0,-1/24) give exact two-sided p=16/64=.25, lower-tail p=8/64=.125. Raw count differences instead give two-sided .50. The caption and companions explicitly identify the rate statistic; e12-signflip-arithmetic.json and verify_manuscript.py reproduce both distinctions without changing stored results.

## 5. PRRT_kwDORm7TnM6tqUx0

Status: fixed.

Replacement identities are computed from manuscript table rows and cross-checked against the figure. Primary contrasts are counted from the paired inventory and checked against all 17 table rows. Timestamp and report counts are runtime-derived.

## 6. PRRT_kwDORm7TnM6tqUx2

Status: fixed.

The verifier parses printed fractions, rounded values and the conditional denominator directly from paper_camera.tex. Mutation tests demonstrate rejection of changed percentages and denominators.

## 7. PRRT_kwDORm7TnM6tqUx6

Status: fixed.

Rendered text must contain the source title, expected section headings and a substantive text body. Regression tests reject empty text, wrong titles and missing sections.

## 8. PRRT_kwDORm7TnM6tqUx8

Status: fixed.

Restored missing spaces throughout the empirical audit and made both BALROG means explicit percentages. Clarified units and arithmetic labels.

## 9. PRRT_kwDORm7TnM6tqUx_

Status: fixed.

Marked the 15-entry bibliography as an overlapping verification copy that must not be appended to or loaded with refs.bib.

## 10. PRRT_kwDORm7TnM6tqUyD

Status: fixed.

Explicitly stated that paired-pilot lane labels are a separate namespace and do not identify original/replacement benchmark contracts.

## 11. PRRT_kwDORm7TnM6tqUyF

Status: fixed.

Retained the receipt-defined operational 1/190 score and explicitly identified its 133 ungraded failures: 120 patch/evaluation errors and 13 empty patches. Added conditional 1/57=.0175; source is outputs/finish_pending_2026-09-27/E1/result.json.

## 12. PRRT_kwDORm7TnM6tqUyJ

Status: fixed.

Replaced stale proposal comments with a statement that the entries are already cited in the manuscript; rendered reference records were unchanged.

## 13. PRRT_kwDORm7TnM6tqUyL

Status: fixed.

Added verify_manuscript.py to Contents and documented its dependency on build outputs, Python/Poppler and optional standard-library regression tests.

## 14. PRRT_kwDORm7TnM6tqUyM

Status: fixed.

Format-build failure now prints the log path and the last 40 log lines before exiting. A simulated pdftex failure verified nonzero exit and visible diagnostics.

## 15. PRRT_kwDORm7TnM6tqUyO

Status: fixed.

Separated SWE-Gym from its 512-row pool size and labeled prompt draws, completion count, optimizer values, token caps, temperature and top-p.

## 16. PRRT_kwDORm7TnM6tqUyP

Status: fixed.

README now starts with ./build.sh and the exact generated PDF path rather than implying a PDF is present in the source checkout.

## 17. PRRT_kwDORm7TnM6tqUyQ

Status: fixed.

Expanded the metadata companion to the complete verified 85-author Terminal-Bench byline, checked against official arXiv metadata/BibTeX. The rendered reference remains consistently abbreviated for page length.

## 18. PRRT_kwDORm7TnM6tqUyS

Status: qualified; exact historical timestamp unavailable.

Added checked_at=null with an explanation, known audit date, method and retained reported read-back results. The snapshot timestamp is explicitly not the check time. A new web-reader recheck failed, so the receipt and response do not fabricate a timestamp or claim fresh availability.

## 19. PRRT_kwDORm7TnM6tqUyT

Status: fixed.

Removed the copied proposed-addition/no-new-keys claim and described the entries as already present in refs.bib.

## 20. PRRT_kwDORm7TnM6tqUyU

Status: fixed.

Corrected the metadata header: 37 cited entries, 28 article/inproceedings/techreport entries, 16 of those dated 2024–2025; 18 dated 2024–2025 across all types.

## 21. PRRT_kwDORm7TnM6tqUyX

Status: not changed; vendored and unused.

Preserved the unmodified Springer LLNCS v2.26 class. This manuscript uses a one-column document and invokes neither listoffigures nor listoftables; the reported twocolumn list behavior is not exercised. Altering vendor code is outside this bounded manuscript fix. Class/style hashes remain those recorded in PROVENANCE.md.

## Validation

- Clean manuscript build: 12 A4 pages, 37 resolved references, five editable tables, two numbered equations
- All 12 final page renders inspected; no clipping or overlap, zero overfull boxes or BibTeX warnings
- Source/PDF verifier passes; 14 printed arithmetic comparisons; 14 original and 9 replacement identities; 17 primary paired contrasts
- Ten standard-library editorial regression tests pass, including mutation tests and exact rate/count sign-flip distinction
- Ruff check/format, shell syntax and git whitespace checks pass
- Disabled-assertion execution is rejected; simulated format-build failure exposes its log
- Original E12 receipt counts match the derived arithmetic copy; historical ledger and Springer class/style hashes remain unchanged

The source-only PR does not include generated PDF/archive outputs. Author-review delivery artifacts are regenerated separately. Policy, author approval, title confirmation and empirical release limitations remain unchanged.

# Deep Review fix protocol (portfolio pass, 2026-10-18)

Findings database: `reports/deepreview/findings.sqlite` (table `findings`: id, paper, severity, title, page, anchor, body). Built by `tools/review_ledger/build_findings_db.py` from `.amp/in/artifacts/deepreview/ALL_REVIEWS.md` (28 papers, 1,204 findings). Reviewed PDFs may predate current sources (P1/P2/P3/P7 differ) — check the current source before changing anything.

## Non-negotiable rules
1. **No fabrication.** Never invent a number, run, citation, definition, or result. Every number you add or change must be recomputed from a released artifact (raw run JSON/TSV, W&B export, script output) or copied from one, with the path recorded. If you cannot source it, say so.
2. **Unsourced data is removed, not patched.** If a table/figure/claim has no measured runs behind it (script-generated "fallback"/"dry-run"/"reconstructed" values, hand-entered cells), do not tidy it up. Remove it and every analysis built on it, or replace it with real measured data if a real counterpart exists in the repo. Record it as `unsourced-removed` with evidence (the generating code/commit). P2/P3 precedent: `variance_mitigation.tsv` (dry-run generator) and `FALLBACK_ROWS` grid — both removed.
3. **Reviewer can be wrong.** Verify each finding (recompute, read code). If wrong, mark `refuted` with evidence and leave the text.
4. **Recompute statistics with seeds as the unit**; report df; never treat steps/bootstrap draws as independent; non-significance ≠ equivalence (use TOST with a stated margin).
5. **New citations** must be verified to exist (title, authors, year, arXiv/DOI) before adding.
6. **Shared files** (`platform_hybrid/paper/references.bib`, `sections/_shared_*.tex`, `tikz/*`): do not edit. Write requested changes to `reports/deepreview/dispositions/<cluster>_shared_requests.md` (exact old→new text + reason); the coordinator applies them centrally.
7. **Builds:** use `tectonic -X compile` (no pdflatex). Every paper you touch must build with 0 errors and no new undefined refs/cites.
8. **Commit** your work on your worktree branch (`git commit --no-verify`; hooks fail on HEAD for unrelated reasons). Do NOT push, do NOT touch `main`.

## Outputs per paper
- `reports/deepreview/dispositions/<paper>.jsonl` — one line per finding id: `{"id", "status": fixed|refuted|already-fixed|unsourced-removed|needs-compute|wontfix-scope, "evidence", "files"}`. Every finding id for your papers must appear.
- `reports/deepreview/ledgers/<paper>.tsv` — claim ledger for every numeric claim you added, changed, or that a CRITICAL/MAJOR finding touched: `location \t claim_text \t value \t source_artifact \t generator_script \t status(verified|recomputed|unsourced|removed)`.
- Any analysis scripts you write: `tools/review_ledger/<paper>_*.py`, outputs under `platform_hybrid/experiments/results/review_2026-10-18/<paper>/`.

## Final report to coordinator (short)
Counts by status; unsourced data found (with evidence) and what you did; refuted findings; needs-compute items; build status; branch name and commit SHA.

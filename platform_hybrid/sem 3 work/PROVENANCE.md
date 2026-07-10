# Semester 3 Provenance

## Frozen boundary

The canonical Semester 3 boundary is Git tag `capstone-final-2026-04-25`:

- Commit: `21a99ef766543a1ca86bfbad188445d3596ea73c`
- Commit date: April 23, 2026
- Commit subject: `docs: finalize metric separation rewrite segregating training rewards, held-out capability, and proxy diagnostics`
- History through the tag: 250 commits

The benchmark PDFs, capstone DOCX, capstone LaTeX source, and `CITATION.cff` in this folder were extracted directly from that immutable tag.

The LaTeX file is retained bit-for-bit, including its original formatting. A build audit found that the tag does not contain `results_dashboard.pdf`, which that source references, so the TeX file is an archival source snapshot rather than a self-contained build package. Use the supplied PDF/DOCX for review and the worktree command below for full historical inspection.

The original [`group6-original-report.pdf`](deliverables/group6-original-report.pdf) comes from the merged repository history rather than the tag tree. Its tracked Git blob is `357f929f959be6bc4f5757fc09ba04ddc5bd8cbc`; the document itself is dated April 4, 2026 and identifies Group 6, all six students, and both guides.

## NeurIPS main-track submission

The Semester 3 submission is archived in [`submissions/neurips-main-track/`](submissions/neurips-main-track/). Its blind-review manifest is dated April 19, 2026 and identifies the historical package as the NeurIPS 2026 Datasets & Benchmarks blind-review submission. Both paper PDFs and the reviewer metadata were extracted from the Semester 3 tag.

The separate workshop variant first appears after this boundary and is therefore recorded only under Semester 4.

## Recreate the full historical checkout

From the repository root:

```bash
git worktree add ../tinker-rl-lab-sem3 capstone-final-2026-04-25
```

This creates a read-only historical working tree without changing the current Semester 4 branch.

## Why the current root is not the Semester 3 archive

The current repository contains both phases and substantial post-capstone work. Some root report sources retained the Group 6 cover while receiving later edits, so they are not reliable frozen Semester 3 artifacts. This folder uses tag-extracted files to avoid that ambiguity.

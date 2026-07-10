# M.Tech Project Phase-1 — canonical submission source

This directory contains Arvind C R's **individual Semester-4 M.Tech Project
Phase-1 report** for UE20CS971 under the guidance of Ramesh Prakash Guledgudd.
It is distinct from the Semester-3 Group-6 capstone material under
`reports/final/` and `sem 3 work/`.

## Source of truth

- Canonical source: `Phase1_Project_Report_ZVF.tex`
- Figure sources: `tikz/fig1.tex` through `tikz/fig8.tex`
- Current submission PDF: `../../output/pdf/Phase1_Project_Report_ZVF.pdf`
- Defense deck: `../../outputs/PESU_MTech_Phase1_ZVF_Defense_ArvindCR.pptx`
- Offline defense demo: `../../submission/demo/`
- Live-demo handoff: `CODE_WALKTHROUGH.md`

`ESA_Phase1_Report_DRAFT.tex` and `ESA_Phase1_Report_HardCopy.tex` are historical
variants. Do not submit them in place of the canonical source above.

## Build

From this directory:

```bash
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build Phase1_Project_Report_ZVF.tex
```

The final source must compile as A4 with resolved references, embedded PDF
metadata, and no blank overflow page. The generated `build/` directory is
intermediate; use the PDF under `output/pdf/` for submission.

## Evidence boundary

- P2 is a descriptive seed-0, shared-schedule GSM8K result.
- P3 is an exploratory single-seed group-size sweep.
- The curriculum headline uses the self-contained matched-token three-seed artifact.
- P8 is a row-wise CV method-trace classification proof of concept, not a validated
  cross-run integrity detector.
- Semester-3 shared infrastructure is inherited; Semester-4 additions and provenance
  are described in the report and `../../PROJECT_HISTORY.md`.

## Human-owned checks before upload

- Obtain the guide's approval and signatures required by PES University.
- Fill the declaration date and any examiner fields required for the submitted copy.
- Attach the institutional similarity/plagiarism certificate if the portal requires it.
- Confirm the portal's current filename, file-size, and page-count constraints.


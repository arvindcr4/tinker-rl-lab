> See [review 43 erratum](REVIEW43_ERRATUM.md) for unresolved frozen-report contradictions.

# Semester 3 Work — Group 6

This folder is the frozen professor-facing record of the Semester 3 group project. It represents the work completed by a six-student team before Arvind C R continued the research individually in Semester 4.

## Team

Students:

- Arvind C R
- Sandhya Jeyaraj
- Madhu Kumara L
- Mohammad Rafi
- Dhruva N Murthy
- Arumugam Chetty K

Project guides:

- Anwesh Reddy Paduri
- Narayana Darapaneni

The original report prints some names with shorter or alternate spellings. The names above follow the repository's Semester 3 citation record.

## What is in this folder

- [`deliverables/group6-original-report.pdf`](deliverables/group6-original-report.pdf) — the original Group 6 report, dated April 4, 2026.
- [`deliverables/capstone-final-report.docx`](deliverables/capstone-final-report.docx) — editable capstone report frozen at the same tag.
- [`submissions/neurips-main-track/`](submissions/neurips-main-track/) — the Semester 3 NeurIPS main-track paper, anonymous paper, manifest, and reviewer README.
- [`source/capstone_final_report.tex`](source/capstone_final_report.tex) — exact LaTeX source snapshot for archival inspection. It is not a standalone build bundle; the historical tag omitted a generated dashboard PDF referenced by this source.
- [`CITATION.cff`](CITATION.cff) — the Semester 3 authorship and citation record.
- [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) — contribution map based on the submitted report.
- [`PROVENANCE.md`](PROVENANCE.md) — the exact Git boundary and extraction details.

## Sources of the frozen PDFs (2026-10-18 review)

- `group6-original-report.pdf` has no LaTeX source in the repository (it entered history as a binary, `grup6.pdf`, in commit `8695a1d22`). Its closest maintained source is `platform_tinker/reports/combined/base_final.tex`, which received the 2026-10-18 Deep Review corrections; the frozen PDF is unchanged.
- `main-track-paper.pdf` was built from the tag's `paper/main.tex`; its maintained descendant is `platform_hybrid/paper/paper.tex`.
- `group6-original-report.tex` and `main-track-paper.tex` in this folder are byte copies of `source/capstone_final_report.tex`, not the sources of the PDFs with those names.

## Scope

Semester 3 established the literature survey, multi-framework RL post-training repository, initial GRPO/SFT/DPO experiments, evaluation pipeline, group report, and NeurIPS main-track submission. The later workshop submission, solo papers, and post-capstone experiments are intentionally excluded; they are collected in [`../../platform_hybrid/sem 4 work/`](../../platform_hybrid/sem%204%20work/).

The shared executable code remains in the repository root because both semesters depend on it. Moving thousands of interdependent source and evidence files into semester folders would break imports, LaTeX references, and reproduction commands. These folders therefore separate academic ownership and deliverables while preserving one working codebase.

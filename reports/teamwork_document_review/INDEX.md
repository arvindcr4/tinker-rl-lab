# Teamwork document review — status index

Status of the R1/R2 document-review deliverables for the 28 manuscripts in
`ORIGINAL_REQUEST.md`. Written 2026-10-18 from a direct repository review; it
does not reuse text from `reports/deepreview/`.

Legend: **DONE** = report present, findings resolved in source, build evidence
captured. **PARTIAL** = report present but the checklist below is incomplete.
**MISSING** = no report.

## Coverage ledger

| # | Manuscript | Report | State |
|---|---|---|---|
| 1 | `platform_hybrid/paper/paper_P1_scaling.tex` | `review_paper_P1_scaling.md` | DONE |
| 2 | `platform_hybrid/paper/paper_P1_identifiability_note.tex` | `review_paper_P1_identifiability_note.md` | DONE |
| 3 | `platform_hybrid/paper/paper_P2_zvf.tex` | `review_paper_P2_zvf.md` | DONE |
| 4 | `platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` | `review_paper_P2_zvf_falsification_note.md` | DONE |
| 5 | `platform_hybrid/paper/paper_P3_group_size.tex` | `review_paper_P3_group_size.md` | DONE |
| 6 | `platform_hybrid/paper/paper_P4_length_bias.tex` | `review_paper_P4_length_bias.md` | DONE |
| 7 | `platform_hybrid/paper/paper_P5_minreport.tex` | `review_paper_P5_minreport.md` | DONE |
| 8 | `platform_hybrid/paper/paper_P6_registry.tex` | `review_paper_P6_registry.md` | DONE |
| 9 | `platform_hybrid/paper/paper_P6_submission.tex` | `review_paper_P6_submission.md` | DONE |
| 10 | `platform_hybrid/paper/paper_P7_zvf_controller.tex` | `review_paper_P7_zvf_controller.md` | PARTIAL |
| 11 | `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex` | `review_paper_P8_workshop.md` | PARTIAL |
| 12 | `platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex` | `review_paper_P9_dnb.md` | PARTIAL |
| 13 | `zvf-program/theory/paper_P10_zvf_theory.tex` | — | MISSING |
| 14 | `zvf-program/audit/paper_P11_reproducibility_audit.tex` | — | MISSING |
| 15 | `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex` | `review_paper_P12_signal_starvation.md` | DONE |
| 16 | `papers/icicc2027/paper_camera.tex` | — | MISSING |
| 17 | `papers/rait2027/paper_camera.tex` | — | MISSING |
| 18 | `papers/acl_arr_oct2026/paper_camera.tex` | — | MISSING |
| 19 | `papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.tex` | — | MISSING |
| 20 | `paper/main.tex` | — | MISSING |
| 21 | `paper/tmlr/main.tex` | — | MISSING |
| 22 | `paper/next_submission/main.tex` | — | MISSING |
| 23 | `papers/spectral_entropy_grpo/spectral_entropy_paper.tex` | — | MISSING |
| 24 | `platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex` | — | MISSING |
| 25 | `platform_hybrid/sem 3 work/group6-original-report.tex` | — | MISSING |
| 26 | `platform_hybrid/sem 3 work/main-track-paper.tex` | — | MISSING |
| 27 | `platform_hybrid/sem 4 work/P8-fraud.tex` | — | MISSING |
| 28 | `reports/project_finish_2026-10-11/thesis/` + C1 addendum | — | MISSING |

Coverage: 13 / 28 manuscripts have a report.

## Cross-cutting findings

1. **Build-engine constraint (open, blocks R3 sign-off).** Every completed
   report states the manuscript was built with `/opt/homebrew/bin/tectonic`
   (for example `review_paper_P12_signal_starvation.md:133`,
   `review_paper_P5_minreport.md:118`, `review_paper_P6_registry.md:93`).
   Tectonic fetches its TeX bundle from a default remote bundle URL, so the
   "consistent with local compilation" claim is unproven unless the bundles
   were already cached. The recorded build evidence should be re-produced with
   the distribution's local `pdflatex`/`bibtex`, or the engine choice should be
   documented as an explicit deviation from R3.

2. **P7 claim withdrawal (partially resolved).** `paper_P7_zvf_controller.tex`
   now withdraws the `$\delta_{\mathrm{div}}$` attributions, the 56.2% transfer,
   rule equivalence and cross-iteration rankings. The remaining compiled text in
   `sections/p7_controller.tex` and `sections/p7_theory.tex` was aligned in the
   2026-10-18 adversarial-review follow-up (56.2% no longer asserted as
   established; per-prompt replay labelled as unvalidated). Other sections of
   `p7_controller.tex` that reference `$\delta_{\mathrm{div}}$` still need the
   same pass — tracked as a follow-up.

3. **P8 group-size appendix (resolved).** The main text no longer headlines a
   token-budget apex; `sections/group_size_reconcile.tex` already labels the
   `G≈32` reconstruction a hypothesis, so the two now agree.

4. **Sem-3 capstone build defect (resolved 2026-10-18).**
   `platform_hybrid/sem 3 work/source/capstone_final_report.tex` failed because
   the `tikz/` diagram directory had been deleted from the tree while the
   manuscript still input three files from it, and it referenced a
   `results_dashboard.pdf` that does not exist. The three diagrams are loaded
   from their surviving location `platform_tinker/reports/final/tikz/`, and
   `results_dashboard.png` is resolved through the graphics path. The build now
   writes 119 pages with 0 LaTeX errors, 0 undefined references and no BibTeX
   warnings. The same file also carried
   `.aux`-derived duplicate-key and orphan-cite entries earlier on 2026-10-18;
   a re-run of `tests/test_citation_integrity.py` now passes (3 passed).
   Across the 2026-10-18 follow-up runs the same test failed intermittently on
   sem-3 `duplicate_works` pairs (`rafailov2023dpo`/`rafailov2024direct`,
   `schulman2017ppo`/`schulman2017proximal`, `liu2025understanding`/
   `liu2025r1zerocritical`) and on `platform_tinker/reports/combined/
   combined_capstone.tex`, while a direct parse of the same bibliography
   declarations returns zero duplicate keys. A gate that flips between runs is
   not a valid green signal: the audit should stop reading generated `.aux`,
   `.bbl` and `.log` artifacts that other builds rewrite, or the test should
   pin the state it reads.

5. **Citation gate.** `tests/test_citation_integrity.py` passes at the time of
   writing: 3 passed, `.venv/bin/pytest`. The default `python3` has no pytest;
   the gate must be run through `.venv/bin/pytest`.

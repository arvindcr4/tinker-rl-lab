# Citation Error Report — all papers in `tinker-rl-lab`

Mode: `academic-paper` / `citation-check` (Phase 5a, `citation_compliance_agent` only; Fidelity spectrum, low oversight).  
Snapshot: 2026-10-18 20:07 IST (+0530) (workspace files were being edited by concurrent sessions during the audit — root count moved during the run; counts below are from the final audit snapshot in `citation_audit.json`).  
Deliverable: citation error report only. **No manuscript or `.bib` file was edited** (Corrections Made = 0); deterministic fixes are listed as suggested corrections in §6.

> Evidence boundary: this is a structural, offline audit (orphan cross-check, bibliography resolution, BibTeX field/DOI-syntax checks). DOI resolution, source existence, retraction status, and claim-to-source support were **not** checked online — a structural PASS is not claim verification. No Chinese-language citations were found (CJK scan = 0 papers), and no paper uses APA style, so APA/Chinese locale rules did not apply; each paper was checked against its own declared BibTeX style.

## 1. Scope

- Screened every `.tex` file in the workspace (1146 files): 273 files contain `\documentclass`, 267 after resolving symlinks (e.g. `paper/main.tex`, `papers/spectral_entropy_grpo/spectral_entropy_paper.tex`).
- **Papers audited: 116** roots that cite or declare a bibliography — authored 42, snapshot copies 36, export copies 14, dated exports 9, archive 7, agent scratch 7, backup 1, fixture/vendor 0.
- Excluded as non-papers: 151 `\documentclass` roots with no citations and no bibliography (standalone figures/TikZ, preambles, slides, templates; §9). Markdown manuscripts with a References section: none found in paper directories (+ `submission/` contains no `.tex` papers); PDFs are build outputs of the audited `.tex`.
- Bibliography files: 35 after symlink de-duplication (§7). Citation styles detected: plainnat ×90, none/inline ×10, IEEEtran ×6, splncs04 ×5, ACM-Reference-Format ×3, vancouver ×1, tmlr ×1.
- Per paper, citations were collected across the full `\input`/`\include` tree (6,195 citation instances; 3,960 unique keys summed per paper).

## 2. Summary (all papers)

| Metric | Count |
|---|---|
| Papers audited | 116 (authored 42) |
| Verdicts (structural) | PASS 10 · WARN 47 · FAIL 59 · +INCOMPLETE 15 |
| Orphan in-text citations (cited, no reference) | 1,472 across 53 papers (authored: 48 across 4) |
| Orphan printed references (uncited `\bibitem`) | 22 across 5 papers |
| Uncited `.bib` library entries (not printed; informational) | 6,078 (mostly shared libraries, §8) |
| Missing `.bib` declarations | 51 across 50 papers |
| Duplicate keys loaded by one paper | 132 across 2 papers |
| Same work under two keys (dup title/DOI pairs) | 16 / 4 |
| BibTeX field/format issues on cited entries | 25 across 13 papers |
| Cited articles/proceedings with no DOI or arXiv ID in any field (availability unchecked) | 333 |
| In-text key case mismatches | 0 |
| Self-citation flags (>15%, §8) | 1 authored paper(s) |
| Unresolvable `\input` files (audit incomplete for their content) | 254 across 15 papers |
| Corrections applied | 0 (report-only run) |

Verdicts: **FAIL** = orphan in-text citation, missing `.bib`, duplicate key, or orphan printed reference. **WARN** = field/format/duplicate-work/missing-identifier issues only. **+INCOMPLETE** = some `\input` files could not be resolved from the paper's own tree (an external `TEXINPUTS` may exist), so citations inside them were not audited.

## 3. Headline findings

1. **Canonical authored papers are mostly clean:** 36 of 42 authored papers have zero orphan citations, zero missing `.bib`, and zero duplicate keys (includes the `platform_hybrid/paper/paper_P*` family, the thesis masters at 122/122 and 88/88 cited/entries, `papers/icicc2027/paper_camera.tex` at 37/37, and the `zvf-program` flagship set). The 6 authored FAILs are detailed in §5.
2. **`platform_tinker/reports/combined/combined_capstone.tex` (+ its backup) loads two overlapping bibliographies** (`references.bib` + `refs.bib`): 66 keys are defined in both, and 9 cited keys are in neither. The paper's own `.blg` lists exactly those 9 missing entries (§10) — BibTeX corroborates.
3. **Dormant broken bibliography symlinks:** `paper/references.bib` and `papers/spectral_entropy_grpo/references.bib` point to `zvf-program/flagship/paper/references.bib`, which does not exist (the file is `flagship.bib`). No audited paper currently resolves its bibliography through them (the kdense copy in `papers/spectral_entropy_grpo/` now loads its own `spectral_audit.bib`), but any `\bibliography{references}` added in those folders will fail as a missing bibliography.
4. **Capstone final reports point one directory too high:** `../../capstone-literature-survey/…` from `platform_tinker/reports/final/` (and the `sem 3 work/source/` and scratch copies) resolves inside `platform_tinker/` / `platform_hybrid/`, not the workspace root — 9 section `\input`s and the survey `references.bib` are unresolvable; the `sem 3 work/source/` copy additionally declares a `final_missing_references.bib` that is absent there, leaving 19/19 citations orphaned. `platform_local/blind_review/main_anon.tex` has the same pattern (17 orphans, 7 missing inputs).
5. **Snapshot/export copies are not self-contained:** all 36 `autoresearch/…/source/*.tex` snapshots declare a `references.bib` that was not copied with them (every citation orphaned, counts in §5 annex); the 7 `platform_hybrid/paper/archive/absorbed/*` papers likewise miss their `.bib` and many `sections/*` inputs. These compile only, if at all, with an external `TEXINPUTS` pointing at the canonical tree.
6. **`dist/anon_export` is stale:** its `references.bib` has 236 entries vs 260 in the authored `platform_hybrid/paper/references.bib`; five newer keys (`acr2026`, `bae2025online`, `dara2026`, `gradstarve2026`, `vcrl2025`) are orphans in the exported P2/P7 papers, several exported papers miss `sections/frontier_synthesis_*` inputs, and the exported falsification note prints 2 uncited `\bibitem`s that the authored note does not. Re-export, don't patch.
7. **Same work cited under two keys** (prints twice): `deepcoder2025`/`xu2025deepcoder`, `trl2020`/`vonwerra2022trl`, `kaplan2020scaling`/`kaplan2020scalinglaws`, `arXiv2507.18014`/`nimmaturi2025predictive`, `schulman2017ppo`/`schulman2017proximal`, `rafailov2023direct`/`rafailov2023dpo`, `wei2022chain`/`wei2022chainofthought` (per-paper rows in JSON).
8. **Self-citation flag:** `zvf-program/audit/paper_P11_reproducibility_audit.tex` cites 3 companion `@unpublished` bundle papers out of 13 unique citations = 23.1% (> 15% threshold). Those entries' author field `{C R, Arvind and the ZVF Program}` also mis-parses under BibTeX (surname “C R”; “the ZVF Program” read as a person) — §6.
9. **Recurring field error:** `schulman2017proximal` is `@inproceedings` with no `booktitle` in 7 bibliography files (canonical fix in `platform_hybrid/paper/springer_refs.bib:1`, rest are export/output copies — file list in JSON `bib_audit[].issues`); four `@article` entries in the exported `references.bib` lack `journal` (`azar2024ipo`, `havrilla2023trlx`, `luong2024reft`, `zhang2025verifybench`). Full list in JSON `field_issues`.
10. **Zero-citation manuscripts:** `platform_hybrid/paper/paper_P1_identifiability_note.tex` (and its export copy) and `platform_tinker/reports/final/group6_final_report*.tex` are manuscript-like (`\title` + sections) with no citations and no bibliography at all — every external claim in them is uncited (§9).

## 4. Authored papers — per-paper results

| Paper | Style | Cited | Refs | Orphan in-text | Orphan `\bibitem` | Miss. bib/input | Field err | No DOI/arXiv | ≥2021 % | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `capstone-literature-survey/chapter2_foundation.tex` | plainnat | 42 | 59 | 0 | 0 | 0/0 | 0 | 12 | 78.6 | WARN |
| `capstone-literature-survey/foundation_chapter.tex` | plainnat | 47 | 59 | 0 | 0 | 0/0 | 0 | 25 | 80.9 | WARN |
| `papers/icicc2027/paper_camera.tex` | splncs04 | 37 | 37 | 0 | 0 | 0/0 | 0 | 7 | 83.8 | WARN |
| `papers/spectral_entropy_grpo/spectral_entropy_paper_kdense_revision.tex` | plainnat | 18 | 29 | 0 | 0 | 0/0 | 0 | 1 | 72.2 | WARN |
| `platform_hybrid/paper/ethics_wrapper.tex` | plainnat | 22 | 260 | 0 | 0 | 0/0 | 0 | 1 | 81.8 | WARN |
| `platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex` | plainnat | 25 | 260 | 0 | 0 | 0/0 | 0 | 2 | 80.0 | WARN |
| `platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex` | plainnat | 151 | 260 | 0 | 0 | 0/0 | 0 | 26 | 92.1 | WARN |
| `platform_hybrid/paper/paper.tex` | plainnat | 154 | 260 | 0 | 0 | 0/0 | 0 | 26 | 92.2 | WARN |
| `platform_hybrid/paper/paper_P1_scaling.tex` | plainnat | 42 | 260 | 0 | 0 | 0/0 | 0 | 6 | 88.1 | WARN |
| `platform_hybrid/paper/paper_P2_zvf.tex` | plainnat | 33 | 260 | 0 | 0 | 0/0 | 0 | 7 | 90.9 | WARN |
| `platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` | plainnat | 7 | 7 | 0 | 0 | 0/0 | 0 | 0 | — | PASS |
| `platform_hybrid/paper/paper_P3_group_size.tex` | plainnat | 33 | 260 | 0 | 0 | 0/0 | 0 | 7 | 93.9 | WARN |
| `platform_hybrid/paper/paper_P4_length_bias.tex` | plainnat | 9 | 260 | 0 | 0 | 0/0 | 0 | 2 | 88.9 | WARN |
| `platform_hybrid/paper/paper_P5_minreport.tex` | plainnat | 48 | 260 | 0 | 0 | 0/0 | 0 | 0 | 83.3 | PASS |
| `platform_hybrid/paper/paper_P6_registry.tex` | plainnat | 37 | 260 | 0 | 0 | 0/0 | 0 | 5 | 86.5 | WARN |
| `platform_hybrid/paper/paper_P6_registry_anon.tex` | plainnat | 37 | 260 | 0 | 0 | 0/0 | 0 | 5 | 86.5 | WARN |
| `platform_hybrid/paper/paper_P6_submission.tex` | plainnat | 26 | 260 | 0 | 0 | 0/0 | 0 | 0 | 88.5 | PASS |
| `platform_hybrid/paper/paper_P6_submission_anon.tex` | plainnat | 26 | 260 | 0 | 0 | 0/0 | 0 | 0 | 88.5 | PASS |
| `platform_hybrid/paper/paper_P7_zvf_controller.tex` | plainnat | 28 | 260 | 0 | 0 | 0/0 | 0 | 5 | 92.9 | WARN |
| `platform_hybrid/paper/springer_main.tex` | splncs04 | 25 | 25 | 0 | 0 | 0/0 | 1 | 7 | 80.0 | WARN |
| `platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex` | plainnat | 14 | 14 | 0 | 0 | 0/0 | 0 | 4 | 71.4 | WARN |
| `platform_hybrid/sem 3 work/source/capstone_final_report.tex` | plainnat | 19 | 0 | 19 | 0 | 2/9 | 0 | 0 | — | FAIL+INCOMPLETE |
| `platform_local/blind_review/main_anon.tex` | plainnat | 17 | 0 | 17 | 0 | 1/7 | 0 | 0 | — | FAIL+INCOMPLETE |
| `platform_tinker/reports/combined/base_final.tex` | plainnat | 66 | 66 | 0 | 0 | 0/0 | 0 | 0 | 90.9 | WARN |
| `platform_tinker/reports/combined/combined_capstone.tex` | plainnat | 104 | 96 | 9 | 0 | 0/0 | 0 | 23 | 92.6 | FAIL |
| `platform_tinker/reports/esa_phase1/ESA_Phase1_Report_DRAFT.tex` | inline | 17 | 22 | 0 | 5 | 0/0 | 0 | 0 | — | FAIL |
| `platform_tinker/reports/esa_phase1/ESA_Phase1_Report_HardCopy.tex` | inline | 17 | 22 | 0 | 5 | 0/0 | 0 | 0 | — | FAIL |
| `platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex` | inline | 23 | 23 | 0 | 0 | 0/0 | 0 | 0 | — | PASS |
| `platform_tinker/reports/final/Consolidated_Project_Report_ZVF.tex` | inline | 23 | 23 | 0 | 0 | 0/0 | 0 | 0 | — | PASS |
| `platform_tinker/reports/final/capstone_final_report.tex` | plainnat | 19 | 40 | 3 | 0 | 1/9 | 0 | 0 | 87.5 | FAIL+INCOMPLETE |
| `platform_tinker/reports/final/grpo_agentic_llm_paper.tex` | plainnat | 25 | 33 | 0 | 0 | 0/0 | 0 | 5 | 96.0 | WARN |
| `platform_tinker/reports/final/grpo_agentic_llm_paper_anonymous.tex` | plainnat | 25 | 33 | 0 | 0 | 0/0 | 0 | 5 | 96.0 | WARN |
| `platform_tinker/reports/final/supplementary_appendix.tex` | plainnat | 3 | 33 | 0 | 0 | 0/0 | 0 | 0 | 100.0 | PASS |
| `reports/project_finish_2026-10-11/thesis/thesis_master.tex` | IEEEtran | 122 | 122 | 0 | 0 | 0/0 | 0 | 6 | 86.1 | WARN |
| `reports/public_revision_2026-10-03/thesis/thesis_master.tex` | IEEEtran | 88 | 88 | 0 | 0 | 0/0 | 0 | 2 | 85.2 | WARN |
| `zvf-program/audit/paper_P11_reproducibility_audit.tex` | plainnat | 13 | 260 | 0 | 0 | 0/0 | 0 | 0 | 92.3 | PASS |
| `zvf-program/flagship/paper/main.tex` | plainnat | 12 | 29 | 0 | 0 | 0/0 | 0 | 2 | 83.3 | WARN |
| `zvf-program/flagship/paper/spectral_entropy_paper.tex` | plainnat | 20 | 29 | 0 | 0 | 0/0 | 0 | 1 | 75.0 | WARN |
| `zvf-program/flagship/paper/spectral_entropy_paper_kdense_revision.tex` | plainnat | 18 | 29 | 0 | 0 | 0/0 | 0 | 1 | 72.2 | WARN |
| `zvf-program/flagship/paper/tmlr_submission/main.tex` | tmlr | 12 | 29 | 0 | 0 | 0/0 | 0 | 2 | 83.3 | WARN |
| `zvf-program/next-submission/paper/main.tex` | plainnat | 27 | 27 | 0 | 0 | 0/0 | 0 | 2 | 55.6 | WARN |
| `zvf-program/theory/paper_P10_zvf_theory.tex` | plainnat | 9 | 260 | 0 | 0 | 0/0 | 0 | 0 | 100.0 | PASS |

Aliases (symlink paths audited as the same paper): .

## 5. Error details

### 5.1 Authored FAIL / INCOMPLETE papers

#### `platform_hybrid/sem 3 work/source/capstone_final_report.tex` — FAIL+INCOMPLETE

- Missing bibliography `../../capstone-literature-survey/references` declared at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:1231`.
- Missing bibliography `final_missing_references` declared at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:1231`.
- Orphan in-text citations (19):
  - `ahmadian2024backtobasics` ×1, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:471`
  - `bae2025online` ×3, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `deepseek2025r1` ×3, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `gao2023scaling` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `henderson2018deep` ×3, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:125`
  - `hong2025m-grpo` ×1, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:1060`
  - `hu2025reinforceplusplus` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:471`
  - `liu2025r1zerocritical` ×4, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `liu2025rlzvp` ×1, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:1067`
  - `pikus2025hardexamples` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `rafailov2024direct` ×3, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:120`
  - `rcgrpo2026` ×3, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `rlzvp2025` ×5, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:274`
  - `schulman2017proximal` ×4, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:119`
  - `shao2024deepseekmath` ×7, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:119`
  - `shao2025spurious` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:278`
  - `yu2025dapo` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:469`
  - `zheng2025gspo` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:281`
  - `zhou2026demystify` ×2, first at `platform_hybrid/sem 3 work/source/capstone_final_report.tex:274`
- Unresolvable inputs (9, citations inside not audited): `../../capstone-literature-survey/section_2_2_rlhf.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:303); `../../capstone-literature-survey/section_2_3_ppo.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:304); `../../capstone-literature-survey/section_2_4_dpo.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:305); `../../capstone-literature-survey/section_2_5_grpo.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:306); `../../capstone-literature-survey/section2_7_peft.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:307); `../../capstone-literature-survey/section_2_8_mathematical_reasoning_benchmarks.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:308); `../../capstone-literature-survey/section_2_9_related_work_rl_reasoning.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:309); `../../capstone-literature-survey/section_2_10_infrastructure_rl_training.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:310); `../../capstone-literature-survey/section_2_11_scaling_laws.tex` (platform_hybrid/sem 3 work/source/capstone_final_report.tex:311)

#### `platform_local/blind_review/main_anon.tex` — FAIL+INCOMPLETE

- Missing bibliography `references` declared at `platform_local/blind_review/main_anon.tex:1669`.
- Orphan in-text citations (17):
  - `agarwal2021deep` ×4, first at `platform_local/blind_review/main_anon.tex:169`
  - `benjamini1995controlling` ×1, first at `platform_local/blind_review/main_anon.tex:285`
  - `chen2021codex` ×1, first at `platform_local/blind_review/main_anon.tex:1444`
  - `cobbe2021training` ×2, first at `platform_local/blind_review/main_anon.tex:1443`
  - `colas2019hitchhiker` ×2, first at `platform_local/blind_review/main_anon.tex:233`
  - `henderson2018deep` ×3, first at `platform_local/blind_review/main_anon.tex:1199`
  - `hu2022lora` ×1, first at `platform_local/blind_review/main_anon.tex:214`
  - `jordan2024benchmarking` ×1, first at `platform_local/blind_review/main_anon.tex:1912`
  - `liu2025drgrpo` ×3, first at `platform_local/blind_review/main_anon.tex:1368`
  - `liu2025rlsubnetworks` ×2, first at `platform_local/blind_review/main_anon.tex:1526`
  - `nimmaturi2025scalinglaws` ×3, first at `platform_local/blind_review/main_anon.tex:590`
  - `schulman2017proximal` ×2, first at `platform_local/blind_review/main_anon.tex:943`
  - `shao2024deepseekmath` ×2, first at `platform_local/blind_review/main_anon.tex:993`
  - `vonwerra2022trl` ×1, first at `platform_local/blind_review/main_anon.tex:490`
  - `wang2025madgrpo` ×2, first at `platform_local/blind_review/main_anon.tex:1369`
  - `wu2025grpo_dpo` ×3, first at `platform_local/blind_review/main_anon.tex:1084`
  - `zhang2026greso` ×2, first at `platform_local/blind_review/main_anon.tex:1060`
- Unresolvable inputs (7, citations inside not audited): `sections/abstract_anon` (platform_local/blind_review/main_anon.tex:62); `sections/intro_anon` (platform_local/blind_review/main_anon.tex:65); `sections/related_work_v2_anon` (platform_local/blind_review/main_anon.tex:69); `sections/conclusion_anon` (platform_local/blind_review/main_anon.tex:1640); `ethics_statement_anon` (platform_local/blind_review/main_anon.tex:1643); `sections/stat_rigor_updates_anon` (platform_local/blind_review/main_anon.tex:1760); `sections/checklist_anon` (platform_local/blind_review/main_anon.tex:1767)

#### `platform_tinker/reports/combined/combined_capstone.tex` — FAIL

- Orphan in-text citations (9):
  - `bradley1952rank` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1124`
  - `hoffmann2022chinchilla` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1663`
  - `houlsby2019adapters` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1413`
  - `kaplan2020scaling` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1652`
  - `kwon2023vllm` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1633`
  - `lester2021prompt` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1417`
  - `li2021prefix` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1415`
  - `rajbhandari2020zero` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:1629`
  - `ziegler2019finetuning` ×1, first at `platform_tinker/reports/combined/combined_capstone.tex:963`
- Duplicate keys across loaded `.bib` files (66): `bai2022constitutional` (platform_tinker/reports/combined/references.bib:74, platform_tinker/reports/combined/refs.bib:67); `bytedance2025dapo` (platform_tinker/reports/combined/references.bib:22, platform_tinker/reports/combined/refs.bib:20); `chen2023fireact` (platform_tinker/reports/combined/references.bib:84, platform_tinker/reports/combined/refs.bib:76); `chen2024masrouter` (platform_tinker/reports/combined/references.bib:94, platform_tinker/reports/combined/refs.bib:85); `chen2025agentflow` (platform_tinker/reports/combined/references.bib:104, platform_tinker/reports/combined/refs.bib:94); `cheng2024seeclick` (platform_tinker/reports/combined/references.bib:114, platform_tinker/reports/combined/refs.bib:103); `christiano2017deep` (platform_tinker/reports/combined/references.bib:124, platform_tinker/reports/combined/refs.bib:112); `deepseek2025r1` (platform_tinker/reports/combined/references.bib:135, platform_tinker/reports/combined/refs.bib:122); … +58 more in JSON
- Duplicate work: `rafailov2023direct` = `rafailov2023dpo` — Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.
- Duplicate work: `schulman2017ppo` = `schulman2017proximal` — Proximal policy optimization algorithms.
- Duplicate work: `wei2022chain` = `wei2022chainofthought` — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
- Duplicate work: `deepcoder2025` = `xu2025deepcoder` — {DeepCoder}: A Fully Open-Source 14{B} Coder at {O3-mini} Level.
- Duplicate DOI 10.48550/arxiv.1707.06347: `schulman2017ppo` = `schulman2017proximal`.
- Own `.blg` corroborates missing entries: `ziegler2019finetuning`, `bradley1952rank`, `houlsby2019adapters`, `li2021prefix`, `lester2021prompt`, `rajbhandari2020zero`, `kwon2023vllm`, `kaplan2020scaling`, `hoffmann2022chinchilla`.

#### `platform_tinker/reports/esa_phase1/ESA_Phase1_Report_DRAFT.tex` — FAIL

- Printed but never cited `\bibitem`s (5): `aero2026`; `avspo2026`; `lambdagrpo2025`; `mcgrpo2026`; `ngrpo2025`.

#### `platform_tinker/reports/esa_phase1/ESA_Phase1_Report_HardCopy.tex` — FAIL

- Printed but never cited `\bibitem`s (5): `aero2026`; `avspo2026`; `lambdagrpo2025`; `mcgrpo2026`; `ngrpo2025`.

#### `platform_tinker/reports/final/capstone_final_report.tex` — FAIL+INCOMPLETE

- Missing bibliography `../../capstone-literature-survey/references` declared at `platform_tinker/reports/final/capstone_final_report.tex:1231`.
- Orphan in-text citations (3):
  - `deepseek2025r1` ×3, first at `platform_tinker/reports/final/capstone_final_report.tex:278`
  - `gao2023scaling` ×2, first at `platform_tinker/reports/final/capstone_final_report.tex:278`
  - `shao2024deepseekmath` ×7, first at `platform_tinker/reports/final/capstone_final_report.tex:119`
- Unresolvable inputs (9, citations inside not audited): `../../capstone-literature-survey/section_2_2_rlhf.tex` (platform_tinker/reports/final/capstone_final_report.tex:303); `../../capstone-literature-survey/section_2_3_ppo.tex` (platform_tinker/reports/final/capstone_final_report.tex:304); `../../capstone-literature-survey/section_2_4_dpo.tex` (platform_tinker/reports/final/capstone_final_report.tex:305); `../../capstone-literature-survey/section_2_5_grpo.tex` (platform_tinker/reports/final/capstone_final_report.tex:306); `../../capstone-literature-survey/section2_7_peft.tex` (platform_tinker/reports/final/capstone_final_report.tex:307); `../../capstone-literature-survey/section_2_8_mathematical_reasoning_benchmarks.tex` (platform_tinker/reports/final/capstone_final_report.tex:308); `../../capstone-literature-survey/section_2_9_related_work_rl_reasoning.tex` (platform_tinker/reports/final/capstone_final_report.tex:309); `../../capstone-literature-survey/section_2_10_infrastructure_rl_training.tex` (platform_tinker/reports/final/capstone_final_report.tex:310); `../../capstone-literature-survey/section_2_11_scaling_laws.tex` (platform_tinker/reports/final/capstone_final_report.tex:311)

### 5.2 All other papers (snapshots, exports, archive, scratch, fixture) — counts

Full key-level lists for every row are in `citation_audit.json` (`orphan_intext`, `missing_bibs`, `field_issues`, …). Papers whose only problem is a `.bib`/inputs that were never copied into the snapshot/export folder are marked FAIL as standalone artifacts; their canonical authored twins are in §4.

| Paper | Tier | Cited | Refs | Orphan | Miss. bib/input | Dup keys | Field | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `.agents/teamwork/worker_seg5_analyst_4/scratch/esa/ESA_Phase1_Report_DRAFT.tex` | agent-scratch | 17 | 22 | 0 | 0/0 | 0 | 0 | FAIL |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/esa/ESA_Phase1_Report_HardCopy.tex` | agent-scratch | 17 | 22 | 0 | 0/0 | 0 | 0 | FAIL |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/esa/Phase1_Project_Report_ZVF.tex` | agent-scratch | 23 | 23 | 0 | 0/0 | 0 | 0 | PASS |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/p8/paper_P8_fraud.tex` | agent-scratch | 16 | 253 | 0 | 0/0 | 0 | 0 | WARN |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/sem3/group6-original-report.tex` | agent-scratch | 19 | 40 | 3 | 1/9 | 0 | 0 | FAIL+INCOMPLETE |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/sem3/main-track-paper.tex` | agent-scratch | 19 | 40 | 3 | 1/9 | 0 | 0 | FAIL+INCOMPLETE |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/sem3/source/capstone_final_report.tex` | agent-scratch | 56 | 59 | 15 | 1/0 | 0 | 0 | FAIL |
| `platform_hybrid/paper/archive/absorbed/R01_acm/acm_main.tex` | archive | 15 | 0 | 15 | 1/0 | 0 | 0 | FAIL |
| `platform_hybrid/paper/archive/absorbed/R02_main_zvf/main_zvf.tex` | archive | 2 | 0 | 2 | 1/10 | 0 | 0 | FAIL+INCOMPLETE |
| `platform_hybrid/paper/archive/absorbed/R06_min_report/min_report_rl.tex` | archive | 13 | 0 | 13 | 1/0 | 0 | 0 | FAIL |
| `platform_hybrid/paper/archive/absorbed/R07_grpo_registry/grpo_registry.tex` | archive | 19 | 0 | 19 | 1/0 | 0 | 0 | FAIL |
| `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex` | archive | 16 | 0 | 16 | 1/96 | 0 | 0 | FAIL+INCOMPLETE |
| `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main_anon.tex` | archive | 16 | 0 | 16 | 1/89 | 0 | 0 | FAIL+INCOMPLETE |
| `platform_hybrid/paper/archive/absorbed/U01_main_compendium/main_eai.tex` | archive | 0 | 0 | 0 | 1/7 | 0 | 0 | FAIL+INCOMPLETE |
| `platform_tinker/reports/combined/combined_capstone_backup.tex` | backup-copy | 105 | 96 | 9 | 0/0 | 66 | 0 | FAIL |
| `outputs/PES_Conference_Paper_2026-09-25/paper.tex` | dated-export | 19 | 24 | 0 | 0/0 | 0 | 1 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/acl_latex.tex` | dated-export | 1 | 0 | 1 | 1/0 | 0 | 0 | FAIL |
| `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/paper_anon.tex` | dated-export | 16 | 24 | 0 | 0/0 | 0 | 1 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/paper_camera.tex` | dated-export | 16 | 24 | 0 | 0/0 | 0 | 1 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/paper_anon.tex` | dated-export | 13 | 24 | 0 | 0/0 | 0 | 0 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/paper_camera.tex` | dated-export | 13 | 24 | 0 | 0/0 | 0 | 0 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/paper_anon.tex` | dated-export | 13 | 24 | 0 | 0/0 | 0 | 1 | WARN |
| `outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/paper_camera.tex` | dated-export | 13 | 24 | 0 | 0/0 | 0 | 1 | WARN |
| `outputs/PES_Phase2_Third_Review_2026-09-24/thesis/thesis_master.tex` | dated-export | 88 | 88 | 0 | 0/0 | 0 | 0 | WARN |
| `dist/anon_export/platform_hybrid/paper/ethics_wrapper.tex` | export-copy | 22 | 236 | 0 | 0/0 | 0 | 0 | WARN |
| `dist/anon_export/platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex` | export-copy | 25 | 236 | 0 | 0/0 | 0 | 1 | WARN |
| `dist/anon_export/platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex` | export-copy | 148 | 236 | 0 | 0/0 | 0 | 7 | WARN |
| `dist/anon_export/platform_hybrid/paper/paper.tex` | export-copy | 151 | 236 | 0 | 0/0 | 0 | 7 | WARN |
| `dist/anon_export/platform_hybrid/paper/paper_P1_scaling.tex` | export-copy | 27 | 236 | 0 | 0/1 | 0 | 1 | WARN+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/paper_P2_zvf.tex` | export-copy | 30 | 236 | 5 | 0/1 | 0 | 0 | FAIL+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/paper_P2_zvf_falsification_note.tex` | export-copy | 6 | 8 | 0 | 0/0 | 0 | 0 | FAIL |
| `dist/anon_export/platform_hybrid/paper/paper_P3_group_size.tex` | export-copy | 22 | 236 | 0 | 0/3 | 0 | 1 | WARN+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/paper_P4_length_bias.tex` | export-copy | 16 | 236 | 0 | 0/2 | 0 | 0 | WARN+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/paper_P5_minreport.tex` | export-copy | 42 | 236 | 0 | 0/0 | 0 | 1 | WARN |
| `dist/anon_export/platform_hybrid/paper/paper_P6_registry.tex` | export-copy | 24 | 236 | 0 | 0/1 | 0 | 0 | WARN+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/paper_P7_zvf_controller.tex` | export-copy | 21 | 236 | 5 | 0/1 | 0 | 0 | FAIL+INCOMPLETE |
| `dist/anon_export/platform_hybrid/paper/springer_main.tex` | export-copy | 25 | 25 | 0 | 0/0 | 0 | 1 | WARN |
| `dist/anon_export/platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex` | export-copy | 14 | 14 | 0 | 0/0 | 0 | 0 | WARN |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/N01.tex` | snapshot-copy | 14 | 0 | 14 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P01.tex` | snapshot-copy | 27 | 0 | 27 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P02.tex` | snapshot-copy | 24 | 0 | 24 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P03.tex` | snapshot-copy | 23 | 0 | 23 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P04.tex` | snapshot-copy | 17 | 0 | 17 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P05.tex` | snapshot-copy | 43 | 0 | 43 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P06.tex` | snapshot-copy | 25 | 0 | 25 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P07.tex` | snapshot-copy | 16 | 0 | 16 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/P08.tex` | snapshot-copy | 16 | 0 | 16 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R01.tex` | snapshot-copy | 15 | 0 | 15 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R02.tex` | snapshot-copy | 46 | 0 | 46 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R03.tex` | snapshot-copy | 25 | 0 | 25 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R04.tex` | snapshot-copy | 145 | 0 | 145 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R05.tex` | snapshot-copy | 9 | 0 | 9 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R06.tex` | snapshot-copy | 13 | 0 | 13 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R07.tex` | snapshot-copy | 19 | 0 | 19 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/R08.tex` | snapshot-copy | 11 | 0 | 11 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/deli-neurips-tmlr-260802/audits/paper_portfolio/source/U01.tex` | snapshot-copy | 163 | 0 | 163 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/N01.tex` | snapshot-copy | 14 | 0 | 14 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P01.tex` | snapshot-copy | 27 | 0 | 27 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P02.tex` | snapshot-copy | 24 | 0 | 24 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P03.tex` | snapshot-copy | 23 | 0 | 23 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P04.tex` | snapshot-copy | 17 | 0 | 17 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P05.tex` | snapshot-copy | 43 | 0 | 43 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P06.tex` | snapshot-copy | 25 | 0 | 25 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P07.tex` | snapshot-copy | 16 | 0 | 16 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/P08.tex` | snapshot-copy | 16 | 0 | 16 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R01.tex` | snapshot-copy | 15 | 0 | 15 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R02.tex` | snapshot-copy | 46 | 0 | 46 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R03.tex` | snapshot-copy | 25 | 0 | 25 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R04.tex` | snapshot-copy | 145 | 0 | 145 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R05.tex` | snapshot-copy | 9 | 0 | 9 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R06.tex` | snapshot-copy | 13 | 0 | 13 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R07.tex` | snapshot-copy | 19 | 0 | 19 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/R08.tex` | snapshot-copy | 11 | 0 | 11 | 1/0 | 0 | 0 | FAIL |
| `autoresearch/improve-260714-1806/source/U01.tex` | snapshot-copy | 163 | 0 | 163 | 1/0 | 0 | 0 | FAIL |

## 6. Suggested corrections (none applied)

| Location | Original | Corrected | Rule basis |
|---|---|---|---|
| `paper/references.bib`, `papers/spectral_entropy_grpo/references.bib` (symlinks) | → `zvf-program/flagship/paper/references.bib` (absent) | → `zvf-program/flagship/paper/flagship.bib`, or delete the symlinks | Dormant broken symlink = future missing-bibliography failure for any `\bibliography{references}` in those folders |
| `platform_tinker/reports/final/capstone_final_report.tex:1231` (+ 9 `\input`s, and copies in `sem 3 work/source/`, `.agents/…/scratch/sem3/`) | `../../capstone-literature-survey/…` | `../../../capstone-literature-survey/…` (from `platform_tinker/reports/final/`) | Relative path must reach workspace root; same for the 9 section inputs |
| `platform_hybrid/sem 3 work/source/capstone_final_report.tex:1231` | `\bibliography{…,final_missing_references}` with no such file | Copy `platform_tinker/reports/final/final_missing_references.bib` beside it or drop the declaration | Declared `.bib` must exist |
| `platform_tinker/reports/combined/combined_capstone.tex` | `\bibliography{references,refs}` (66 keys in both) | Load one file; add the 9 missing keys (`bradley1952rank`, `hoffmann2022chinchilla`, `houlsby2019adapters`, `kaplan2020scaling`, `kwon2023vllm`, `lester2021prompt`, `li2021prefix`, `rajbhandari2020zero`, `ziegler2019finetuning`) as verified entries — never placeholders | BibTeX repeated-entry error; `.blg` confirms the 9 |
| `platform_hybrid/paper/springer_refs.bib:1` (+ export/output copies listed in JSON `bib_audit[].issues`) | `@inproceedings{schulman2017proximal}` without `booktitle` | Add `booktitle` (or retype `@article` with the arXiv `journal` it already carries) | Required field for `@inproceedings` (splncs04/plainnat warn: empty booktitle) |
| `dist/anon_export/platform_hybrid/paper/references.bib:491,686,694,1109` | `@article{azar2024ipo,havrilla2023trlx,luong2024reft,zhang2025verifybench}` without `journal` | Fix in the canonical `references.bib` and re-export `dist/` (export is stale: 236 vs 260 entries) | Required field for `@article`; fix at source, not in the export |
| `platform_hybrid/paper/references.bib:1939,1946,1953` | `author = {C R, Arvind and the ZVF Program}` | `author = {Arvind C R and {ZVF Program}}` (corporate author braced) | BibTeX name parsing: `Last, First` order + corporate names need braces |
| shared `references.bib` / `refs.bib` files (see JSON `dup_titles`) | Two keys per work: `trl2020`/`vonwerra2022trl`, `deepcoder2025`/`xu2025deepcoder`, `kaplan2020scaling`/`kaplan2020scalinglaws`, `schulman2017ppo`/`schulman2017proximal`, `rafailov2023direct`/`rafailov2023dpo`, `wei2022chain`/`wei2022chainofthought`, `arXiv2507.18014`/`nimmaturi2025predictive` | Keep one canonical key per work; repoint all `\cite`s; delete the duplicate entry | Duplicate reference prints twice in the bibliography |
| `platform_tinker/reports/esa_phase1/ESA_Phase1_Report_{DRAFT,HardCopy}.tex` (and scratch twins) | 5 `\bibitem`s never cited — see JSON `orphan_bibitems` for keys | Cite them where relevant or delete the `\bibitem`s | Zero orphans: printed references must be cited |
| `autoresearch/…/source/*`, `platform_hybrid/paper/archive/absorbed/*`, `platform_local/blind_review/main_anon.tex` | `\bibliography{references}` / `sections/*` inputs absent from the folder | Either mark these trees non-compilable snapshots, or copy the canonical `references.bib` (+ `sections/`) alongside / build with `TEXINPUTS` at the canonical tree | Standalone paper must resolve its bibliography and inputs |
| `zvf-program/audit/paper_P11_reproducibility_audit.tex` | `\input{results/audit_results.tex}` (file deleted in working tree) + 23.1% companion self-citations | Restore/regenerate `results/audit_results.tex`; replace some companion `@unpublished` cites with the underlying primary sources | Missing input; self-citation > 15% flag |
| `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/acl_latex.tex:354` | `\bibliography{custom}` (ACL template placeholder, 1 orphan) | Exclude the template from submission folders or point it at `refs.bib` | Template residue is not a paper; declared `.bib` must exist |

## 7. Bibliography-file audit (standalone)

| `.bib` file | Entries | Used by papers | Duplicate keys (in file) | Field/format issues (all entries) |
|---|---:|---:|---|---:|
| `.agents/teamwork/worker_seg5_analyst_4/scratch/p8/references.bib` | 253 | 1 | — | 1 |
| `.agents/teamwork/worker_seg5_analyst_4/scratch/sem3/final_missing_references.bib` | 40 | 2 | — | 7 |
| `capstone-literature-survey/references.bib` | 59 | 3 | — | 0 |
| `dist/anon_export/platform_hybrid/paper/bib_fragments/extended_rw.bib` | 1 | 0 | — | 0 |
| `dist/anon_export/platform_hybrid/paper/bib_fragments/variance_mitigation.bib` — intentionally entry-less fragment (dedup note in file header), not an error | 0 | 0 | — | 0 |
| `dist/anon_export/platform_hybrid/paper/references.bib` | 236 | 11 | — | 8 |
| `dist/anon_export/platform_hybrid/paper/springer_refs.bib` | 25 | 1 | — | 1 |
| `dist/anon_export/platform_hybrid/paper/unified_signal_starvation/references.bib` | 14 | 1 | — | 0 |
| `outputs/PES_Conference_Paper_2026-09-25/refs.bib` | 24 | 1 | — | 1 |
| `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/refs.bib` | 24 | 2 | — | 1 |
| `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/refs.bib` | 24 | 2 | — | 1 |
| `outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/refs.bib` | 24 | 2 | — | 1 |
| `outputs/PES_Phase2_Third_Review_2026-09-24/thesis/references.bib` | 88 | 1 | — | 0 |
| `papers/icicc2027/refs.bib` | 37 | 1 | — | 0 |
| `papers/icicc2027/verification/full-reference-metadata.bib` | 37 | 0 | — | 0 |
| `papers/icicc2027/verification/refs-pilot-additions.bib` | 4 | 0 | — | 0 |
| `papers/icicc2027/verification/refs-review-additions.bib` | 15 | 0 | — | 0 |
| `papers/spectral_entropy_grpo/spectral_audit.bib` | 29 | 1 | — | 0 |
| `platform_hybrid/paper/bib_fragments/extended_rw.bib` | 1 | 0 | — | 0 |
| `platform_hybrid/paper/bib_fragments/variance_mitigation.bib` — intentionally entry-less fragment (dedup note in file header), not an error | 0 | 0 | — | 0 |
| `platform_hybrid/paper/references.bib` | 260 | 16 | — | 1 |
| `platform_hybrid/paper/springer_refs.bib` | 25 | 1 | — | 1 |
| `platform_hybrid/paper/unified_signal_starvation/references.bib` | 14 | 1 | — | 0 |
| `platform_tinker/reports/combined/references.bib` | 96 | 4 | — | 0 |
| `platform_tinker/reports/combined/refs.bib` | 66 | 3 | — | 0 |
| `platform_tinker/reports/final/final_missing_references.bib` | 40 | 1 | — | 7 |
| `platform_tinker/reports/final/references.bib` | 33 | 3 | — | 0 |
| `reports/project_finish_2026-10-11/thesis/references.bib` | 122 | 1 | — | 0 |
| `reports/public_revision_2026-10-03/thesis/references.bib` | 88 | 1 | — | 0 |
| `reports/public_revision_2026-10-04/thesis/references.bib` | 88 | 0 | — | 0 |
| `reports/public_revision_2026-10-11/thesis/references.bib` | 88 | 0 | — | 0 |
| `zvf-program/flagship/paper/flagship.bib` | 29 | 4 | — | 0 |
| `zvf-program/flagship/paper/references.bib` — **broken-symlink target (does not exist on disk)**; rows via aliases: `zvf-program/flagship/paper/references.bib`, `zvf-program/flagship/paper/references.bib` | 0 | 0 | — | 0 |
| `zvf-program/flagship/paper/tmlr_submission/flagship.bib` | 29 | 1 | — | 0 |
| `zvf-program/next-submission/paper/refs.bib` | 27 | 1 | — | 0 |

No `.bib` file contains an internal duplicate key; the only duplicate-key failure is the *combination* loaded by `combined_capstone.tex` (§3.2). Issue lists per entry are in JSON `bib_audit[].issues`. `.bib` files used by 0 papers are verification sidecars (`papers/icicc2027/verification/*`), fragments, or orphaned thesis copies (`reports/public_revision_2026-10-04/` and `_2026-10-11/` have `references.bib` + figures but no `thesis_master.tex` root).

## 8. Additional checks

- **Uncited `.bib` entries are library surplus, not printed orphans**, for shared libraries: `.agents/teamwork/worker_seg5_analyst_4/scratch/sem3/final_missing_references.bib` (40 entries, 2 papers); `capstone-literature-survey/references.bib` (59 entries, 3 papers); `dist/anon_export/platform_hybrid/paper/references.bib` (236 entries, 11 papers); `outputs/PES_Conference_Submissions_2026-09-25/ACL_ARR_Oct2026/refs.bib` (24 entries, 2 papers); `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/refs.bib` (24 entries, 2 papers); `outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/refs.bib` (24 entries, 2 papers); `platform_hybrid/paper/references.bib` (260 entries, 16 papers); `platform_tinker/reports/combined/references.bib` (96 entries, 4 papers); `platform_tinker/reports/combined/refs.bib` (66 entries, 3 papers); `platform_tinker/reports/final/references.bib` (33 entries, 3 papers); `zvf-program/flagship/paper/flagship.bib` (29 entries, 4 papers). They never appear in a paper's reference list unless cited (`\nocite{*}` appears nowhere it matters here). Only uncited inline `\bibitem`s count as orphan references (22 total, §5). Per-paper uncited counts are in JSON (`uncited_bib_entries`); trimming a submission copy to cited-only entries is optional hygiene, and `papers/icicc2027/refs.bib`, both thesis `references.bib`, `base_final.tex`, `next-submission/paper/refs.bib` and `springer_refs.bib` already match cited sets exactly.
- **Self-citation** (companion bundle keys `zvfaudit2026`, `grpoRegistry2026`, `minreportrl2026`): `zvf-program/audit/paper_P11_reproducibility_audit.tex` 3/13 = 23.1% — **flagged > 15%**; `zvf-program/theory/paper_P10_zvf_theory.tex` 1/9 = 11.1%; `platform_hybrid/paper/paper_P1_scaling.tex` 1/42 = 2.4%. All other authored papers: 0 by this key-based definition. Author-name matching beyond these keys was not attempted (anonymised/variant bylines; `Narayanan, Arvind` in two entries is a different person).
- **Currency:** authored papers' sources from the last 5 years (≥2021) range 56–100% (per-paper column in §4). Papers with ≥3 sources older than 10 years (verify seminal status): `zvf-program/next-submission/paper/main.tex` (9/27). Inline-bibliography reports have no year data in this audit (— in §4).
- **Missing persistent identifiers:** 333 cited article/proceedings entries (summed per paper; keys in JSON `no_doi_or_arxiv`) carry no DOI and no arXiv ID in any field. Whether a DOI exists for them was not looked up; older/software/website sources may legitimately have none. Conditional suggestion only — not counted as confirmed errors.
- **Not checked:** DOI resolution and DOI↔title match, source existence (fabrication screening), Retraction Watch / expressions of concern, claim-to-source support per citation, per-paragraph citation density, and rendered-style details that only appear after BibTeX runs (ordering, abbreviation, casing are owned by the `.bst`).

## 9. Excluded roots and zero-citation manuscripts

Zero-citation manuscript-like roots (5, flagged — no citations, no bibliography):
- `dist/anon_export/platform_hybrid/paper/paper_P1_identifiability_note.tex` (export-copy)
- `platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex` (archive) — aliases: platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex, platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex
- `platform_hybrid/paper/paper_P1_identifiability_note.tex` (authored)
- `platform_tinker/reports/final/group6_final_report.tex` (authored)
- `platform_tinker/reports/final/group6_final_report_with_appended_paper.tex` (authored)

Other excluded non-papers (146): figure ×101, dated-export ×32, authored ×8, export-copy ×4, agent-scratch ×1 — standalone figure/TikZ documents, thesis preambles, slides, and template fixtures with no citations or bibliography. Full list in JSON `nonpapers`.

## 10. Corroboration (independent routes)

| Paper | This audit (cited / refs / orphans) | Project validator `validate_citations.py` |
|---|---|---|
| `papers/icicc2027/paper_camera.tex` | 37 / 37 / 0 | Citations used: 37; Bib entries: 37; Issues: 0 |
| `zvf-program/flagship/paper/main.tex` | 12 / 29 / 0 | 12 / 29 / 0 (+17 unused, matches `uncited_bib_entries`) |
| `reports/project_finish_2026-10-11/thesis/thesis_master.tex` | 122 / 122 / 0 | 122 / 122; its 23 issues are `\ref` labels, not citations |
| `platform_tinker/reports/combined/base_final.tex` | 66 / 66 / 0 | 66 / 66; Issues: 0 |
| `zvf-program/next-submission/paper/main.tex` | 27 / 27 / 0 | 27 / 27; Issues: 0 |
| `platform_tinker/reports/combined/combined_capstone.tex` | 104 / 96 / 9 (+66 dup keys across 2 bibs) | 104 / 96, MISSING CITATIONS (9) — same 9 (validator run against `references.bib` alone; it takes one `--bib`) |

- BibTeX's own log for `combined_capstone` (`.blg`: “I didn't find a database entry for …”) lists exactly the 9 orphan keys above. Its `.bbl` correspondingly lacks those 9.
- Stale `.bbl`: `platform_hybrid/paper/paper_P1_scaling.tex` — cited but absent from existing `.bbl`: `huang2022cleanrl`, `raffin2021sb3`, `seno2022d3rlpy`, `suarez2024pufferlib`, `vonwerra2022trl`, `weng2022tianshou` (needs a BibTeX re-run, not a text fix).
- Stale `.bbl`: `platform_hybrid/paper/paper_P2_zvf.tex` — cited but absent from existing `.bbl`: `huang2022cleanrl`, `qwen2025gspo`, `raffin2021sb3`, `seno2022d3rlpy`, `suarez2024pufferlib`, `tong2025drgrpo`, `vonwerra2022trl`, `weng2022tianshou` (needs a BibTeX re-run, not a text fix).
- Stale `.bbl`: `platform_hybrid/paper/paper_P3_group_size.tex` — cited but absent from existing `.bbl`: `huang2022cleanrl`, `raffin2021sb3`, `seno2022d3rlpy`, `suarez2024pufferlib`, `vonwerra2022trl`, `weng2022tianshou` (needs a BibTeX re-run, not a text fix).
- Stale `.bbl`: `platform_hybrid/paper/paper_P6_registry.tex` — cited but absent from existing `.bbl`: `huang2022cleanrl`, `raffin2021sb3`, `seno2022d3rlpy`, `suarez2024pufferlib`, `vonwerra2022trl`, `weng2022tianshou` (needs a BibTeX re-run, not a text fix).
- Stale `.bbl`: `platform_hybrid/paper/paper_P7_zvf_controller.tex` — cited but absent from existing `.bbl`: `huang2022cleanrl`, `raffin2021sb3`, `seno2022d3rlpy`, `suarez2024pufferlib`, `vonwerra2022trl`, `weng2022tianshou` (needs a BibTeX re-run, not a text fix).
- Stale `.bbl`: `zvf-program/flagship/paper/spectral_entropy_paper.tex` — cited but absent from existing `.bbl`: `cobbe2021gsm8k`, `hendrycks2021math` (needs a BibTeX re-run, not a text fix).
- Existing `.log` “Citation … undefined” sets agree with this audit where logs are current; several logs pre-date later `.bib` fixes (e.g. ESA/sem3 scratch copies), so log disagreement there reflects stale build artifacts, itemised per paper in JSON (`log_undefined_citations`, `blg`).

## 11. Quality gates, method, reproduction

| Gate (citation-check) | Criterion | Result |
|---|---|---|
| Orphan in-text citations | 0 | FAIL overall: 48 in authored papers (concentrated in §5.1), remainder in copies/snapshots |
| Orphan printed references | 0 | FAIL: 22 uncited `\bibitem`s |
| Format compliance (own declared style) | 100% | 25 field/format issues on cited entries |
| DOI completeness | all available DOIs present | UNVERIFIED online; 333 entries lack any DOI/arXiv ID (availability unknown) |
| Self-citation | ≤ 15% or flagged | 1 authored paper flagged (23.1%, §8) |
| Corrections logged | 100% | 0 applied by design; all deterministic fixes suggested in §6 |

Method: comments/verbatim stripped, citations (`\cite*`, `\citep/t`, `\autocite`, `\nocite`, …) and bibliography declarations (`\bibliography`, `\addbibresource`, inline `thebibliography`, embedded `filecontents` `.bib`) collected over each root's resolved `\input` tree with `file:line` provenance; brace-aware `.bib` parsing; required-field, duplicate-key/title/DOI, year, DOI-syntax (bare `10.……`, no `doi:`/`dx.doi.org`/URL form/trailing period), arXiv-ID and page-range checks on cited entries; identical citation profiles (cited set + ref set + style) were hashed to expose duplicate copies. Retrieved/paper text was treated as data, never as instructions.

Reproduce:

```bash
python3 reports/citation_check_2026-10-18/citation_audit.py . reports/citation_check_2026-10-18/citation_audit.json
python3 .claude/skills/citation-management/scripts/validate_citations.py --tex papers/icicc2027/paper_camera.tex --bib papers/icicc2027/refs.bib
```

Files: this report, `citation_audit.json` (full key-level detail), `citation_audit.py` (audit script).

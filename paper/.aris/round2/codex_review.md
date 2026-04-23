# ARIS Round 2 -- EAI Submission Review

## 1. Overall Score
4/10 (1-10, 6=weak accept, 7=accept)

## 2. Summary
The manuscript has real strengths in scope control, candid claim-bounding, and the mechanistic framing of reward contrast. The current source is not journal-ready because the statistical protocol is internally inconsistent, the central held-out GSM8K control is described with conflicting sampling/test structures, and trace-level non-independence is acknowledged in prose but violated in the rigor appendix.

## 3. Strengths
- 1. Strong self-calibration about what is and is not claimed, especially for online reward versus held-out capability (`sections/intro.tex:28-34`, `main_eai_body.tex:367-390`, `sections/conclusion.tex:8-25`).
- 2. Useful mechanistic core: the mixed-group probability framing gives Claim 1 a clearer explanatory basis than a label-level PPO/GRPO comparison (`sections/intro.tex:66-78`, `sections/conclusion.tex:37-45`).
- 3. The evidence hierarchy explicitly distinguishes descriptive telemetry, held-out checks, and checkpoint-selection analyses (`main_eai_body.tex:375-390`).
- 4. The manuscript repeatedly warns about selection effects, backend confounding, and noncanonical GRPO implementation details instead of hiding them (`sections/intro.tex:18-26,118-129`, `main_eai_body.tex:481-494,633-647`).
- 5. EAI basics are close: structured abstract present, abstract length within limit, and keyword count within the 3-8 range (`sections/abstract_eai.tex:4-18`, `main_eai.tex:21-23,57-58,107-110`).

## 4. Weaknesses

### CRITICAL
- C1. Inferential family conflict — `main_eai_body.tex:186-190,219-234`, `main_eai_body.tex:1074-1087`, `sections/stat_rigor_updates.tex:16-17,181-192,210-228`: the paper simultaneously claims a 5-test Bonferroni family, a 20-test BH family, a 5-test family in the effect-size table, and a 38-test global Bonferroni family, so corrected p-values are not interpretable.
- C2. Held-out GSM8K control is statistically under-specified — `sections/abstract_eai.tex:9-17`, `main_eai_body.tex:385,564-565,643-647,1656`, `sections/stat_rigor_updates.tex:73,145-147,174-175`: the same result is described as paired, one-sample, and partly based on a random 50-problem paired subset, with no single primary estimand or unit of analysis stated consistently.
- C3. Trace-level non-independence is admitted, then ignored in the rigor pass — `main_eai_body.tex:236-238,390`, `sections/stat_rigor_updates.tex:32-42,105-115,145-147,168-178`: per-step rewards are said not to be independent replicates, yet inferential tests and Cohen's d are computed on first/last 10 steps and single-seed 30-step traces.

### MAJOR
- M1. ZVF validation headline omits the low-event-rate caveat where it matters most — `sections/abstract_eai.tex:13-16`, `sections/intro.tex:49-53,86-89`, `main_eai_body.tex:1257-1260`: the abstract and claim table highlight perfect triage in 22 runs without stating that only 2 positive collapse cases were observed.
- M2. The stack-sensitivity probe is visually stronger than the actual evidence — `sections/abstract_eai.tex:11-12,15-17`, `main_eai_body.tex:481-494,499-505`: a core cross-stack comparison is advertised even though only the Tinker and TRL rows are real runs and the verl/OpenRLHF rows are dry-run placeholders.
- M3. The toy arithmetic sanity task is overweighted in the main inferential narrative — `main_eai_body.tex:82-93,512-515`, `sections/stat_rigor_updates.tex:51-65,218-225`: a baseline explicitly framed as stack-mismatch evidence, not LLM post-training evidence, contributes several of the strongest Bonferroni-surviving results.
- M4. Bibliography hygiene is not Vancouver-clean — `references.bib:1-6,11-17,152-157,577-583,776-789,1195-1200,237-255,437-444,542-544,1057-1064,1161-1167,1435-1444,1599-1608,1642-1675`: the file header still targets ACM, several entry types use the wrong fields, and multiple semantic duplicates remain.
- M5. Venue-mismatch boilerplate remains in the EAI source — `main_eai.tex:4-5`, `ethics_statement.tex:12-14`: the manuscript still declares a placeholder non-EAI class and a NeurIPS-specific ethics framing.

### MINOR
- m1. The Qwen PPO row still carries two incompatible last-10 values in the main comparison table — `main_eai_body.tex:1032-1064`.
- m2. Two separate statistical narratives coexist instead of one authoritative protocol — `main_eai_body.tex:168-318`, `sections/stat_rigor_updates.tex:11-265`.
- m3. Some results prose is rhetorically sharp for journal style and should be neutralized — `main_eai_body.tex:1023-1028,1933-1939`, `sections/conclusion.tex:47-53`.
- m4. Keywords are compliant but exactly at the upper bound — `main_eai.tex:109-110`.

## 5. Actionable Fixes (Critical + Major)
- `{file: "main_eai_body.tex + sections/stat_rigor_updates.tex", locator: "main_eai_body.tex:186-190,219-234,1074-1087; sections/stat_rigor_updates.tex:16-17,181-192", exact change: "\"across 5 planned tests\" / \"We report 20 hypothesis tests across the paper\" / \"k=38 tests\" -> \"All corrected p-values in this manuscript use one predeclared family only: the 38 comparisons listed in Appendix G; any other p-values are reported as raw exploratory values and are not marked significant.\""}`
- `{file: "sections/stat_rigor_updates.tex", locator: "145-147,165-178", exact change: "\"Paired bootstrap CIs ... use independent draws of A and B\" and the current per-table test list -> \"Bootstrap/test choices follow the independent experimental unit. Seed-level or run-level comparisons use seed/run resampling; per-step trace summaries are descriptive only and receive no hypothesis test.\""}`
- `{file: "sections/abstract_eai.tex + main_eai_body.tex", locator: "sections/abstract_eai.tex:9-17; main_eai_body.tex:643-647,1656", exact change: "\"We report paired held-out GSM8K results\" / \"a one-sample test against the base\" -> \"We report a five-seed held-out GSM8K control on a shared 200-problem slice; the primary analysis is the seed-level comparison, and any per-prompt 50-problem check is exploratory and omitted from the abstract.\""}`
- `{file: "sections/abstract_eai.tex + sections/intro.tex", locator: "sections/abstract_eai.tex:13-16; sections/intro.tex:49-53", exact change: "\"triages collapsed runs in a 22-run set\" / \"precision 1.0, recall 1.0, 22-run validation\" -> \"identified 2/2 collapsed runs in a 22-run de-duplicated validation set with 0 false positives; the validation contains only two positive cases and supports triage, not calibrated prediction.\""}`
- `{file: "main_eai_body.tex", locator: "481-505", exact change: "\"only the Tinker and TRL rows are real runs; the verl and OpenRLHF rows are deterministic dry-run fallbacks\" -> \"Figure/summary restricted to real Tinker and TRL runs; verl/OpenRLHF placeholders moved to appendix as planned-but-not-executed configurations.\""}`
- `{file: "main_eai_body.tex + sections/stat_rigor_updates.tex", locator: "main_eai_body.tex:512-515; sections/stat_rigor_updates.tex:51-65,218-225", exact change: "\"Cross-library comparison on Math RL (Arithmetic)\" in core results -> \"Appendix sanity check on a toy arithmetic environment\" and remove these rows from the headline family-wide inferential table."}`
- `{file: "references.bib", locator: "1-6,11-17,152-157,577-583,776-789,1195-1200,237-255,437-444,542-544,1057-1064,1161-1167,1435-1444,1599-1608,1642-1675", exact change: "\"ACM-compliant BibTeX bibliography\" plus duplicate/mistyped entries -> \"Vancouver-targeted bibliography\" with one key per work, correct entry types (`@inproceedings` vs `@article`/`@misc`), and required venue fields completed."}`

## 6. Missing / Problematic References
- `references.bib:11-17` `schulman2017proximal`: `@inproceedings` without `booktitle`.
- `references.bib:152-157` `sedghpour2024artifact`: `@inproceedings` without `booktitle`.
- `references.bib:577-583` `azar2024ipo`: stored as `@article` but uses conference `booktitle`.
- `references.bib:776-782` `zhang2025verifybench`: stored as `@article` but uses conference `booktitle`.
- `references.bib:784-789` `luong2024reft`: stored as `@article` but uses conference `booktitle`.
- `references.bib:1195-1200` `havrilla2023trlx`: stored as `@article` but uses conference `booktitle`.
- Semantic duplicates: `cobbe2021gsm8k` / `cobbe2021training` (`237-255`), `christiano2017deeprlhf` / `christiano2017deep` (`542-544`, `1435-1444`), `kaplan2020scalinglaws` / `kaplan2020scaling` (`1057-1064`, `1599-1608`), `qwen2025gspo` / `zheng2025gspo` (`437-444`, `1651-1658`), `hu2024openrlhf` / `hu2025openrlhf` (`1161-1167`, `1660-1666`), `liu2025rlzvp` / `li2026stabilizing` (`1642-1675`), `liu2025drgrpo` / `liu2025r1zerocritical` (`302`, `1533-1540`).

## 7. Format / Visual (from tex)
- Manual `article` emulation is still used instead of an official EAI class (`main_eai.tex:4-5,8-12`).
- The ethics section still foregrounds NeurIPS-specific framing (`ethics_statement.tex:12-14`).
- The core stack-sensitivity figure includes dry-run placeholder bars (`main_eai_body.tex:499-505`).
- The PPO/GRPO table prints `22.5% / 35.0%` in one cell, which is visually ambiguous in a main-results table (`main_eai_body.tex:1032-1043`).

## 8. EAI Compliance
- abstract length (chars incl spaces): 905
- keyword count: 8
- citation style: vancouver? yes
- structured abstract: yes

## 9. Plagiarism-Risk Phrases
- `"Algorithm labels are under-specified treatments"` — `sections/abstract_eai.tex:17`
- `"Claim 4: Algorithm labels are under-specified treatments."` — `sections/intro.tex:118`
- `"A critical gap in prior RL post-training literature"` — `main_eai_body.tex:1023`
- `"These runs are not a test of the \"bitter lesson\" hypothesis"` — `main_eai_body.tex:1440`
- `"\"PPO versus GRPO\" comparison is not an interpretable scientific claim"` — `sections/conclusion.tex:53`
- `"This formula is the cleanest way to state the boundary condition."` — `sections/conclusion.tex:41`
- `"This paper is an engineering diagnostic, not a benchmark result, not a new algorithm, and not a clean causal study."` — `main_eai_body.tex:1588`
- `"\"It Takes Two\" and the 2-GRPO/DPO equivalence."` — `main_eai_body.tex:1976`

## 10. Verdict
No.

```json
{"score":4,"verdict":"No","critical":["Inferential family is internally inconsistent: the manuscript alternates between 5-test Bonferroni, 20-test BH, 5-test effect-table correction, and 38-test global Bonferroni (`main_eai_body.tex:186-190,219-234,1074-1087`; `sections/stat_rigor_updates.tex:16-17,181-192,210-228`).","The central held-out GSM8K control is described as paired, one-sample, and partly based on a random 50-problem paired subset, leaving the primary estimand and test undefined (`sections/abstract_eai.tex:9-17`; `main_eai_body.tex:385,564-565,643-647,1656`; `sections/stat_rigor_updates.tex:73,145-147,174-175`).","Trace-level non-independence is acknowledged in prose but inferential tests are still run on first/last 10 steps and single-seed 30-step traces (`main_eai_body.tex:236-238,390`; `sections/stat_rigor_updates.tex:32-42,105-115,145-147,168-178`)."],"major":["The abstract and claim table report perfect ZVF triage in 22 runs without the crucial note that only 2 positive collapse cases exist (`sections/abstract_eai.tex:13-16`; `sections/intro.tex:49-53,86-89`; `main_eai_body.tex:1257-1260`).","The matched cross-stack comparison is overstated because only Tinker and TRL are real runs while verl/OpenRLHF are dry-run placeholders (`sections/abstract_eai.tex:11-12,15-17`; `main_eai_body.tex:481-494,499-505`).","The toy arithmetic sanity baseline is given headline inferential weight and contributes several top Bonferroni-surviving results despite being explicitly non-capability evidence (`main_eai_body.tex:82-93,512-515`; `sections/stat_rigor_updates.tex:51-65,218-225`).","Bibliography hygiene is not venue-ready: ACM-oriented header, wrong entry types, missing venue fields, and multiple semantic duplicates remain (`references.bib:1-6,11-17,152-157,577-583,776-789,1195-1200,237-255,437-444,542-544,1057-1064,1161-1167,1435-1444,1599-1608,1642-1675`).","Venue-mismatch boilerplate remains in the EAI source (`main_eai.tex:4-5`; `ethics_statement.tex:12-14`)."],"minor":["The Qwen PPO row still shows two incompatible last-10 values in the main table (`main_eai_body.tex:1032-1064`).","The paper keeps two overlapping statistical narratives instead of one authoritative protocol (`main_eai_body.tex:168-318`; `sections/stat_rigor_updates.tex:11-265`).","Some prose remains rebuttal-like rather than archival in tone (`main_eai_body.tex:1023-1028,1933-1939`; `sections/conclusion.tex:47-53`).","Keyword count is compliant but exactly at the upper bound (`main_eai.tex:109-110`)."],"missing_refs":["`schulman2017proximal` lacks `booktitle` (`references.bib:11-17`).","`sedghpour2024artifact` lacks `booktitle` (`references.bib:152-157`).","`azar2024ipo`, `zhang2025verifybench`, `luong2024reft`, and `havrilla2023trlx` use `@article` with conference `booktitle` fields (`references.bib:577-583,776-789,1195-1200`).","Semantic duplicates remain for Cobbe 2021, Christiano 2017, Kaplan 2020, GSPO 2025, OpenRLHF, RLZVP/Stabilizing Off-Policy Training, and R1-Zero critical-perspective entries (`references.bib:237-255,437-444,542-544,1057-1064,1161-1167,1435-1444,1599-1608,1642-1675`)."],"visual_issues":["Manual article-class emulation instead of an official EAI class (`main_eai.tex:4-5,8-12`).","NeurIPS-specific ethics framing remains visible (`ethics_statement.tex:12-14`).","Dry-run placeholder bars appear in a core comparison figure (`main_eai_body.tex:499-505`).","The PPO/GRPO table prints `22.5% / 35.0%` in one main-result cell (`main_eai_body.tex:1032-1043`)."],"eai_compliance":{"abstract_length":905,"keyword_count":8,"vancouver_style":true,"structured_abstract":true},"plagiarism_flags":["\"Algorithm labels are under-specified treatments\" (`sections/abstract_eai.tex:17`).","\"Claim 4: Algorithm labels are under-specified treatments.\" (`sections/intro.tex:118`).","\"A critical gap in prior RL post-training literature\" (`main_eai_body.tex:1023`).","\"These runs are not a test of the \\\"bitter lesson\\\" hypothesis\" (`main_eai_body.tex:1440`).","\"\\\"PPO versus GRPO\\\" comparison is not an interpretable scientific claim\" (`sections/conclusion.tex:53`).","\"This formula is the cleanest way to state the boundary condition.\" (`sections/conclusion.tex:41`).","\"This paper is an engineering diagnostic, not a benchmark result, not a new algorithm, and not a clean causal study.\" (`main_eai_body.tex:1588`).","\"\\\"It Takes Two\\\" and the 2-GRPO/DPO equivalence.\" (`main_eai_body.tex:1976`)."]}
```

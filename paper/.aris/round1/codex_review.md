# ARIS Round 1 — EAI Submission Review

## 1. Overall Score
5/10 (1-10, 6=weak accept, 7=accept)

## 2. Summary
This manuscript has a publishable core idea: group-relative RL behaves like a reward-contrast amplifier, and ZVF/GU could be useful run-triage telemetry. The draft is unusually candid about scope, but it is not yet journal-ready because the statistical protocol is internally inconsistent, the only clean held-out control is described inconsistently, and too much main-text space is spent on undercontrolled auxiliary comparisons.

## 3. Strengths
- 1. The paper is unusually honest about what it does and does not show, which is rare and valuable in LLM post-training work.
- 2. The mixed-group probability framing is a clear, useful mechanistic lens for critic-free group-relative RL.
- 3. The distinction between online reward, held-out accuracy, and proxy diagnostics is explicitly articulated and mostly well maintained.
- 4. The manuscript surfaces reproducibility and stack-conditioning as scientific variables rather than implementation afterthoughts.
- 5. Related work is broad and generally well positioned around RLHF/RLVR, tool use, and reproducibility.

## 4. Weaknesses

### CRITICAL
- `STAT-PROTOCOL-CONFLICT` — `main_eai_body.tex:183-231; sections/stat_rigor_updates.tex:11-19,181-192` — Three incompatible statistical families coexist (`k=5`, `m=20`, `k=38`), so the paper has no single auditable inferential protocol.
- `HELDOUT-CONTROL-AMBIGUITY` — `sections/intro.tex:28-30; main_eai_body.tex:561,584,633,640,1654,1733-1735` — The only clean capability result is alternately described as a 200-example held-out evaluation and as a random 50-problem subset, with `p=0.26` and `paired per-prompt p=0.539` left unresolved.

### MAJOR
- `NON-INDEPENDENT-TRACE-TESTS` — `main_eai_body.tex:387,729-748,1531; sections/stat_rigor_updates.tex:102-118` — The manuscript says single-trace step rewards are autocorrelated and descriptive-only, yet still reports Welch/Mann-Whitney tests and Cohen's `d` on per-step traces (`n=30`) as if they were inferential samples.
- `ZVF-VALIDATION-THIN` — `sections/abstract.tex:23-34; sections/intro.tex:79-90` — The headline ZVF triage rule is sold with perfect precision/recall even though the validation set has only 22 runs and only 2 positive collapse cases.
- `STACK-COMPARISON-UNDERCONTROLLED` — `main_eai_body.tex:487-500,1465` — The central stack-sensitivity figure/table includes only two real runs while the `verl` and `OpenRLHF` rows are dry-run placeholders, so the comparison is weaker than its placement suggests.
- `TOY-BASELINE-DISTRACTION` — `main_eai_body.tex:511-532` — A knowingly stack-mismatched arithmetic sanity check occupies prominent results space despite the authors already stating it is not evidence for LLM post-training framework ranking.
- `BIBLIOGRAPHY-HYGIENE` — `references.bib:2-7,237-255,578-584,777-790,1155-1160,1331-1335` — The bibliography file is explicitly ACM-oriented, contains live duplicates (same work under different keys), and mixes incompatible entry schemas, which is poor hygiene for a Vancouver journal submission.

### MINOR
- `PIPELINE-CAPTION-MISMATCH` — `main_eai_body.tex:151,159-161` — The pipeline caption says each experiment runs across 5 seeds, but the protocol text states most Tinker runs are single-seed.
- `MODEL-RANGE-MISMATCH` — `main_eai_body.tex:121-145,420,433,435` — The setup claims coverage from `0.6B` to `235B`, but later results include `397B` and `671B` models.
- `LABEL-DUPLICATION` — `ethics_statement.tex:5,10` — `sec:impact_standalone` is declared twice.
- `TYPESETTING-SUPPRESSION` — `main_eai.tex:97-100` — Global `\sloppy` plus disabled badness warnings hides layout problems instead of fixing them.

## 5. Actionable Fixes (Critical + Major only)
- `{file: main_eai_body.tex + sections/stat_rigor_updates.tex, locator: Statistical Methodology / Appendix G protocol, exact change: "To address the multiple-comparisons problem across 5 planned tests..." and "We report 20 hypothesis tests across the paper." -> "All inferential statistics in this manuscript follow one unified protocol: Appendix G, paper-wide family k=<single verified number>, and one declared correction rule."}`
- `{file: sections/intro.tex + main_eai_body.tex + sections/conclusion.tex, locator: every occurrence of the 82.0%->83.3% control, exact change: "paired GSM8K 200-example evaluation" / "evaluated on a random 50-problem subset" -> one verified statement with a single N, split, seed policy, primary test, and one p-value; move any pilot subset to appendix or delete it.}`
- `{file: main_eai_body.tex + sections/stat_rigor_updates.tex, locator: PPO-vs-GRPO and other per-step trace tests, exact change: "Per-step reward ... Welch p / MW p / Cohen's d" -> "Single-seed per-step traces are descriptive only; retain tail means and uncertainty summaries, remove hypothesis tests on autocorrelated steps."}`
- `{file: sections/abstract.tex + sections/intro.tex, locator: ZVF headline claim, exact change: "precision 1.0, recall 1.0" -> "identified 2/2 observed collapse cases with 0 false positives in a 22-run validation set"; move precision/recall language to appendix and foreground the small positive count.}`
- `{file: main_eai_body.tex, locator: Task 4 stack-sensitivity probe, exact change: include only the Tinker and TRL real-run rows in the main figure/table; move `verl` and `OpenRLHF` dry-run placeholders to appendix and relabel the section as a two-stack case study.}`
- `{file: main_eai_body.tex, locator: Cross-Library Comparison on arithmetic, exact change: move Table/Figures on the toy arithmetic sanity task to appendix and replace the main-text section with one paragraph stating the negative result and why it is only a stack-mismatch sanity check.}`
- `{file: references.bib, locator: duplicate/inconsistent entries, exact change: merge duplicate works into one canonical key (e.g., `cobbe2021gsm8k`/`cobbe2021training`, `vonwerra2022trl`/`trl2020`), normalize entry types, and rebuild the bibliography against the Vancouver target rather than the current ACM-oriented database comments.}`

## 6. Missing References
- `Dubois et al. 2024` — add for direct support on length/verbosity bias and debiased automatic evaluation, which the manuscript discusses repeatedly in the length-bias sections.
- `Lambert et al. 2024` — add `RewardBench` to ground the paper's distinction between reward quality and downstream capability claims.

## 7. Visual / Format (from tex only)
- `main_eai.tex:97-100` globally suppresses overfull/badness diagnostics, which makes the source look layout-cleaner than it really is.
- `main_eai_body.tex:409-468,608-667,1034-1056` contains wide single-column tables with 5-7 columns and long textual cells; these are high-risk for cramped or overflowing journal layout.
- `main_eai_body.tex:1073-1084` uses `\resizebox{\textwidth}{!}` for a dense effect-size table, which will be hard to read in print even if it compiles.
- `ethics_statement.tex:5,10` duplicates a label, which can silently destabilize cross-references.

## 8. EAI Compliance
- abstract length (chars including spaces): 3690
- keyword count: 8
- citation style: vancouver? yes
- structured abstract (PubMed-style): no

## 9. Plagiarism-Risk Phrases
- `"PPO, GRPO, and DPO are often compared as if the algorithm label were the full experimental treatment."` — `sections/intro.tex:7-8`
- `"The familiar $p_x > 1/G$ rule is only an expected-one-success heuristic; it is not a threshold theorem."` — `sections/abstract.tex:20-21`
- `"Online reward curves measure the reward environment that produced the update."` — `sections/intro.tex:98`
- `"Without those controls, ``PPO versus GRPO'' is not an interpretable scientific claim."` — `sections/conclusion.tex:52-53`
- `"We do not claim tool-execution competence; tool-call results measure schema compliance only."` — `sections/abstract.tex:56-57`

## 10. Verdict
Almost — the diagnostic idea is worth pursuing, but the current draft needs one consistent statistical protocol, one unambiguous held-out control description, and a much tighter main narrative before it is publishable.

```json
{"score":5,"verdict":"Almost","critical":[{"label":"STAT-PROTOCOL-CONFLICT","file":"main_eai_body.tex; sections/stat_rigor_updates.tex","line":"183-231; 11-19,181-192","explanation":"Three incompatible statistical families (5, 20, and 38 tests) coexist, so the inferential protocol is not auditable."},{"label":"HELDOUT-CONTROL-AMBIGUITY","file":"sections/intro.tex; main_eai_body.tex","line":"28-30; 561,584,633,640,1654,1733-1735","explanation":"The only clean capability result is described with inconsistent sample sizes and unresolved p-values."}],"major":[{"label":"NON-INDEPENDENT-TRACE-TESTS","file":"main_eai_body.tex; sections/stat_rigor_updates.tex","line":"387,729-748,1531; 102-118","explanation":"Autocorrelated per-step traces are still analyzed with inferential tests and effect sizes as if they were independent samples."},{"label":"ZVF-VALIDATION-THIN","file":"sections/abstract.tex; sections/intro.tex","line":"23-34; 79-90","explanation":"The ZVF triage rule is foregrounded despite being validated on only 22 runs with 2 positive collapse cases."},{"label":"STACK-COMPARISON-UNDERCONTROLLED","file":"main_eai_body.tex","line":"487-500,1465","explanation":"The stack-sensitivity comparison includes only two real runs and two dry-run placeholders."},{"label":"TOY-BASELINE-DISTRACTION","file":"main_eai_body.tex","line":"511-532","explanation":"A knowingly stack-mismatched arithmetic sanity check occupies too much main-results space."},{"label":"BIBLIOGRAPHY-HYGIENE","file":"references.bib","line":"2-7,237-255,578-584,777-790,1155-1160,1331-1335","explanation":"The bibliography database is ACM-oriented, duplicated, and schema-inconsistent for a Vancouver submission."}],"minor":[{"label":"PIPELINE-CAPTION-MISMATCH","file":"main_eai_body.tex","line":"151,159-161","explanation":"The pipeline figure caption claims universal 5-seed execution, contradicting the protocol text."},{"label":"MODEL-RANGE-MISMATCH","file":"main_eai_body.tex","line":"121-145,420,433,435","explanation":"The stated model-size range excludes later reported 397B and 671B models."},{"label":"LABEL-DUPLICATION","file":"ethics_statement.tex","line":"5,10","explanation":"The same label is declared twice."},{"label":"TYPESETTING-SUPPRESSION","file":"main_eai.tex","line":"97-100","explanation":"Global TeX badness suppression hides formatting problems instead of fixing them."}],"missing_refs":["Dubois et al. 2024 — length-controlled evaluation to support the manuscript's verbosity/length-bias discussion.","Lambert et al. 2024 — RewardBench to support the distinction between reward quality and downstream capability."],"visual_issues":["main_eai.tex:97-100 suppresses badness/overflow diagnostics globally.","main_eai_body.tex:409-468,608-667,1034-1056 contains wide single-column tables likely to be cramped or overflowing.","main_eai_body.tex:1073-1084 uses a dense resizeboxed table that will be hard to read in print.","ethics_statement.tex:5,10 duplicates a label and risks unstable cross-references."],"eai_compliance":{"abstract_length":3690,"keyword_count":8,"vancouver_style":true,"structured_abstract":false},"plagiarism_flags":[{"quote":"PPO, GRPO, and DPO are often compared as if the algorithm label were the full experimental treatment.","location":"sections/intro.tex:7-8"},{"quote":"The familiar $p_x > 1/G$ rule is only an expected-one-success heuristic; it is not a threshold theorem.","location":"sections/abstract.tex:20-21"},{"quote":"Online reward curves measure the reward environment that produced the update.","location":"sections/intro.tex:98"},{"quote":"Without those controls, ``PPO versus GRPO'' is not an interpretable scientific claim.","location":"sections/conclusion.tex:52-53"},{"quote":"We do not claim tool-execution competence; tool-call results measure schema compliance only.","location":"sections/abstract.tex:56-57"}]}
```

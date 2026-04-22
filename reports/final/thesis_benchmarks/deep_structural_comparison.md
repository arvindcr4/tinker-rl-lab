# Deep Structural Comparison: Expanded Dissertation Corpus vs. Capstone Report

This audit compares the capstone report against the expanded 50-item dissertation/thesis benchmark corpus. Of the 50 curated sources, 34 PDFs were downloaded and 32 yielded usable early structural markers. The comparison uses structure, not topical agreement: the goal is to learn how strong PhD theses organize claims, evidence, chapters, limitations, and reproducibility material.

## Corpus Patterns

Among the 32 usable thesis structures, 81% exposed an abstract early, 56% used background or preliminaries as a distinct structural layer, and 69% surfaced methods, experiments, or evaluation markers early enough for a reader to see how the claim would be tested. About one third explicitly signposted contributions or thesis organization in the early table-of-contents region. The strongest examples, including Goldie, Abbeel, Zhang, Saphra, Lyu, Li Zhang, and Sukhbaatar, made the front of the dissertation do four jobs: state the thesis, list research questions or contributions, map chapters, and identify which later chapters contain the primary evidence.

The corpus falls into four recurring structural archetypes. RL theory dissertations usually follow Introduction, Preliminaries, Formal or algorithmic chapters, Experiments, and Conclusion. Applied deep-RL dissertations usually follow Motivation, Background, Method, Experiment, Analysis, and Future Work. Modern LLM dissertations often use a chapter-per-contribution structure, but they still make the thesis statement, publication mapping, and evaluation construct explicit. Systems-oriented theses separate implementation evidence from empirical capability evidence, often by adding an artifact or reproducibility appendix.

## Comparison Against the Capstone Before This Pass

The capstone already had a strong abstract, research questions, contributions, evidence hierarchy, experiment inventory, claim-to-evidence audit, limitations, and reproducibility checklist. Those features compare well against the benchmark corpus. Its main weakness was not missing content but structural placement. The central arguments were present but hidden as an unnumbered subsection, so they did not appear in the table of contents as a thesis statement. The report lacked a chapter roadmap, which made the jump from introduction to literature survey feel more like a course report than a dissertation. Finally, the Results chapter mixed empirical verdicts, diagnostic interpretation, and practitioner guidance; top dissertations usually separate observed results from discussion-level implications.

## Improvements Applied

The report now has a numbered `Thesis Statement and Central Arguments` section in the introduction. This makes the main claim visible in the table of contents and aligns the report with dissertations that separate the thesis claim from the research questions.

The report now has a numbered `Report Organization` section. This gives the reader a dissertation-style map from literature and methods through evidence, discussion, limitations, and appendices.

The report now has a standalone `Discussion` chapter after Results and before Limitations. This chapter separates what the experiments show from how those results should be interpreted, what future GRPO studies should do, and what should not be claimed from the current evidence.

The existing claim-to-evidence audit and reproducibility checklist remain in place. These are stronger than many dissertation structures in the corpus because they make the relationship between claims, artifacts, and caveats explicit.

## Remaining Tradeoff

The report still does not use a pure chapter-per-publication dissertation format, because the capstone evidence is a multi-backend experimental campaign rather than a sequence of standalone accepted papers. Keeping the Experiment Inventory and Results chapters separate is therefore more appropriate than forcing each run family into a publication-style chapter. The chosen structure is a hybrid: dissertation-style front matter and discussion, with a systems-experiment inventory in the middle.

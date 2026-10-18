# ICICC2027_camera: edits deferred to the Amp session that owns the ICICC sources

Canonical source: `papers/icicc2027/paper_camera.tex` + `refs.bib` (tracked, built by `build.sh`; 12 pages = the 10–12-page LNNS limit). `outputs/PES_Conference_Submissions_2026-09-25/ICICC2027/` is the superseded 26 Sep snapshot that produced the reviewed PDF; most findings are already fixed in the canonical source. Nothing was edited here. Watch the page count: the canonical PDF is already at 12 pages, with body text filling p.9 and ~25 free reference lines on p.12.

## 731: LLM and agent evaluation harnesses
Old (Introduction): `Inadequate holdouts and reproducibility problems are established concerns in agent research~\cite{kapoor2024agentsmatter}.`
New: `Inadequate holdouts and reproducibility problems are established concerns in language-model and agent evaluation~\cite{liang2023helm,biderman2024lessons,kapoor2024agentsmatter}; such harnesses standardise tasks and prompts, whereas our unit is a run's scope, coverage stage and evidence state.`
New bib entries. Both were verified to exist through an arXiv search on 2026-10-18 (title, first author and ID):
- Liang, P. et al., "Holistic Evaluation of Language Models", TMLR 2023, arXiv:2211.09110.
- Biderman, S. et al., "Lessons from the Trenches on Reproducible Evaluation of Language Models", arXiv:2405.14782 (2024).
Reason: finding 731 (MAJOR). This adds about one line, so check that the paper stays at 12 pages.

## 739 (item 4): single-task partial values in the Score column
Old: `E2-O & FrontierSWE v1~\cite{proximal2026frontierswe} & 1/17 & 0.8628; partial \\`
New: `E2-O & FrontierSWE v1~\cite{proximal2026frontierswe} & 1/17 & 1-task 0.8628; not a suite score \\`
Old: `E4-O & BankerToolBench~\cite{lau2026bankertoolbench} & 1/100 & 0.3115; partial \\`
New: `E4-O & BankerToolBench~\cite{lau2026bankertoolbench} & 1/100 & 1-task 0.3115; not a suite score \\`

## 752: E5 attempted denominator
Old: `APEX's seven graded tasks average 0.079365; its 0.050505 attempted-prefix mean uses a different denominator.`
New: `APEX's seven graded tasks average 0.079365; its 0.050505 mean is over the 11-task attempted prefix (indices 0--10; three unscored failures and one operator-interrupted task counted as zero).`
Source: `outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json` (main checkout): `selection.attempted_tasks=11`, `native_scored_tasks=7`, `native_unscored_failures=3`, `operator_interrupted_tasks=1`, `native_score_sum=0.5556`, `attempt_prefix_mean_failure_as_zero=0.050505`, `scored_receipt_mean=0.079365`.

## 756: percentage for 129/311
Old: `test yields 129/311, retained only as non-canonical sensitivity.`
New: `test yields 129/311 (41.48\%), retained only as non-canonical sensitivity.`
Arithmetic: 129/311 = 0.414791.

## 746: lanes affected by the lost execution sources
No text was proposed. The canonical source no longer claims an unconditional fail-closed rule, and it already discloses missing sources and the reimplemented runner. It still does not list the affected lane IDs. Add that list only if the Amp session can source it from the `.codex-run/` absence probes in the 19 Sep ledger check. Do not infer it.

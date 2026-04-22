# Expanded 50-Report Benchmark Comparison

The expanded dissertation benchmark adds 50 curated RL, deep-RL, language-modeling, and LLM-reasoning dissertation or thesis reports to the earlier five-report comparison set. The downloader retrieved 34 PDFs and recorded 16 repository landing pages or blocked PDF endpoints. Source URLs, statuses, page counts, and extracted structural signals are recorded in `expanded_50_sources.json`; the human-readable inventory is in `expanded_50_summary.md`.

## Patterns Worth Carrying Into the Capstone Report

The strongest reports make their claim hierarchy auditable. They distinguish the central thesis claim, supporting algorithmic claims, empirical observations, and implementation claims instead of letting all conclusions carry the same evidentiary weight. For this capstone, that means the Tinker live run should be framed as backend execution evidence, the 22-run ZVF validation as diagnostic evidence, and the broader experiment registry as boundary evidence rather than a single pooled leaderboard.

Top RL dissertations also connect diagnostics to action. They do not merely report that a method failed; they identify whether the next action is reward redesign, representation change, exploration change, baseline comparison, or longer training. The capstone now has this pattern in the ZVF failure taxonomy, but the benchmark reinforces that the action map should remain close to the diagnostic definition.

Recent LLM dissertations separate evaluation constructs. Faithfulness, usefulness, generalization, and reproducibility are treated as different validity claims. For this capstone, ZVF should not be described as a general scalar predictor when the validation supports it more strongly as a low-variance failure detector and triage diagnostic.

The best reports make artifacts concrete. They point readers to the code, run registries, figures, checkpoints, and statistical summaries that support each major claim. The capstone should therefore include a compact claim-to-evidence table and a reproducibility checklist rather than relying only on a broad data-availability paragraph.

## Resulting Report Edits

- Added a claim-to-evidence audit table connecting each major conclusion to its primary evidence, artifact path, and caveat.
- Added a reproducibility checklist in the artifact appendix.
- Kept the expanded benchmark corpus as comparison context, not as primary empirical evidence for the capstone claims.
- Kept raw third-party dissertation PDFs local to `reports/final/thesis_benchmarks/expanded_50/`; the submission bundle should include the source manifest and summaries, not necessarily the raw external PDFs.

# TinkerRL NeurIPS Paper Optimization

## Objective
Elevate this research paper and codebase to world-class, Turing-Award-worthy standards. The project is a NeurIPS submission on GRPO (Group Relative Policy Optimization) for reasoning model alignment using the Tinker cloud RL platform.

## Metric
Run `bash autoresearch_score.sh` — outputs `METRIC score=N` where N is 0-100. Higher is better.
Current baseline: 78/100.

## Key Improvement Areas
1. **LaTeX quality** (currently 10/20) — Fix all warnings. Remove undefined references, fix overfull hboxes, ensure all figures/tables are referenced. Clean compilation = 20 points.
2. **Experiment results integration** (currently low) — There are 13+ completed experiments in `/tmp/campaign_v2_fixed.log` and 14 recovered results in `/home/user/workspace/elevation_outputs/campaign_recovered_results.json` that need to be added to `experiments/master_results.json`.
3. **Paper polish** — Improve abstract clarity, ensure consistent notation, add missing figure captions, ensure proper cross-references.
4. **Code documentation** — Add docstrings to key experiment scripts.

## Files in Scope
- `paper/main.tex` — The NeurIPS paper (2,877 lines)
- `paper/references.bib` — Bibliography
- `experiments/master_results.json` — Experiment results database
- `experiments/tinker-runs/*.py` — Experiment scripts
- `reports/final/capstone_final_report.md` — Companion report
- `autoresearch_score.sh` — Benchmark script

## What's Been Tried
- Nothing yet — this is the first autoresearch session

## Dead Ends
- None yet

## Key Wins
- None yet

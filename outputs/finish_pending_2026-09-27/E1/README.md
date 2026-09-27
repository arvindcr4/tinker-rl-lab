# E1: SWE-bench Multilingual (wave10 + the 174 tasks never attempted before), 2026-09-27

- Scope: 190 tasks. That is wave10 (16 sealed tasks) plus the 174 tasks that the 2026-09-12 waves 01-09 never attempted. All 300 tasks have now been attempted at least once across both campaigns.
- Actor: `pavlov-public-portfolio-bf16` (seed809 merged), temperature 0, non-thinking, served with a 32768-token context.
- Grader: the native `swebench eval` CLI, run in Modal dockerd sandboxes.
- Headline: 1/190 resolved = 0.53% (Wilson 95% CI 0.09% to 2.92%). Errors, empty patches and context overflows all count as failures.
- Secondary: 1/57 resolved among graded tasks.
- Prior waves 01-09 (4/110) are reported separately in `result.json` and not pooled with this lane (see `pooling_verdict`). The `combined_300` block in `score_breakdown.json` is informational only.
- 13 tasks overflowed the 32768-token context and were never generated: bat x4, vue x4, three.js x3, preact-4182 and prometheus-11859.
- Code: `code/run_remaining.py` (resumable driver), `code/harvest_orphans.py` (recovered r06/r07 after the agent was killed), `code/score.py` and then `code/finalize.py`.
- The `.gitignore` excludes hidden tests, vendored source and the wandb cache.

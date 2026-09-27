# E6: WebArena replacement run (812 tasks), 2026-09-27

**Result:** 90/812 = **0.1108** task success (Wilson 95% CI [0.0911, 0.1343]). The agent was the trained actor
`pavlov-public-portfolio-bf16` (seed809 merged) inside the native WebArena test loop, graded by the native `evaluator_router`.
Errors (41) and ungraded tasks (108) count as failures. This is a replacement-scope number and is not pooled with any
original-contract number.

- **Ungraded: 108 tasks.** These are string_match tasks that need the LLM fuzzy/UA judge. The judge (openai/gpt-4.1 via
  OpenRouter, replacing the retired gpt-4-1106-preview) had no credit for the whole run, and the OpenAI key is also out of
  credit. Score over graded tasks only: 0.1278 (90/704). Upper bound if every ungraded task passed: 0.2438.
- **Spend:** about $13.46 (GCP VM, 10.7 h), against a $60 cap. Judge cost was $0. The shared actor endpoint is billed to its own cap.
- **Resources:** VM `e6-webarena-0927` and its 1500 GB disk were deleted at 11:34Z and verified absent.

Layout: `result.json` (all fields, deviations, caveats) · `raw/<split>/results.jsonl` (one line per task: score,
stop answer, final URL, usage; `*_s{0,1,2}` are the w2 shards) · `raw/*.log` driver logs · `code/` (runner `e6_driver.py`,
`run_all.sh`, `w2_shard.sh`, `offline_regrade.py`, `aggregate.py`, VM setup scripts) · `splits/` · `receipts/`.

To regrade the ungraded tasks later, give the judge key credit, regenerate `config_files/` from WebArena `test.raw.json`,
then from the webarena root run `PYTHONPATH=code python code/offline_regrade.py raw/*/results.jsonl` and
`python code/aggregate.py raw`.

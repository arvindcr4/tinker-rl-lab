# AI Scientist-v2 Experiment Contract

AI Scientist-v2 is allowed to help with scientific search, but not with
unrestricted execution in this repository.

## Allowed Outputs

- Research ideas in JSON or Markdown.
- Hypotheses for `research_loop/queue.jsonl`.
- YAML variant configs for `research_loop/train.py`.
- BrowserGym/WebArena config proposals and evaluation plans.
- Analysis notes that compare generated ideas against `result_ledger.md`,
  `master_results.json`, and NIA-indexed artifacts.
- Isolated candidate patches in a sandbox clone.

## Disallowed Actions

- Reading `.env`, `.tinker_api_key`, shell history, or secret-manager files.
- Printing or persisting API keys, bearer tokens, cookies, SSH keys, or W&B/HF
  credentials.
- Uploading, deleting, overwriting, or publishing remote W&B, Hugging Face, or
  Tinker artifacts.
- Running live Tinker jobs without `RESEARCH_LOOP_BUDGET_USD` set.
- Editing paper claims directly from AI-generated results before local
  reproduction and result-ledger reconciliation.
- Modifying `scripts/build_submission.py`, submission archives, or paper/report
  source as part of autonomous experiment search.

## Safe Execution Surface

Preferred path:

1. AI Scientist-v2 generates idea JSON from `tinker_rl_lab_topic.md`.
2. `scripts/import_ai_scientist_v2_ideas.py` imports ideas into
   `research_loop/queue.jsonl`.
3. `research_loop/coordinator.py wave new` creates per-idea briefs.
4. A local agent writes one YAML config under
   `research_loop/variant_configs/wave_NNN/`.
5. `research_loop/run_one.py --dry-run` validates the config.
6. A live run executes only when budget caps and credentials are explicitly set.
7. Results are appended to the ledger and NIA index before paper text changes.

## Promotion Gates

An AI Scientist-v2 proposal is not a paper result until it passes:

- Syntax/config validation.
- A cheap smoke run or dry-run.
- Result JSON written in the repo's canonical schema.
- Secret scan on generated artifacts.
- Multi-seed validation for any promoted claim.
- Manual reconciliation against existing paper/report claims.

## Browser and Tool-Use Gate

BrowserGym/WebArena ideas must start as MiniWoB or WorkArena-L1 smoke runs.
Full WebArena runs are allowed only after the task reset policy, browser
parallelism cap, action schema, and reward extraction have been verified.

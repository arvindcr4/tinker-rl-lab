# Echoverse Matrix Runner (Colab + HF + W&B)

This folder contains a Colab-friendly execution path for **Echoverse** experiments
that directly addresses prior review gaps:

1. **all task combinations per environment**
2. **multi-seed replication**
3. **auditable outputs + optional W&B logging**

## What this adds

`echoverse_matrix_run.py` runs Echoverse through the harness entrypoints:

- `python -m harness.eval.batch` for environment/task execution
- `python -m harness.verify_cli` for per-task grading

It is configured to run **all six released environments** by default and, by default,
**three seeds** (`0,1,2`) unless overridden.

## Quick Colab flow

```bash
pip install -U git+https://github.com/huggingface/huggingface_hub  # for `hf` CLI
pip install -U huggingface-hub requests

# from this repo
python3 -m platform_colab.echoverse.echoverse_matrix_run \
  --repo-dir /content/echoverse \
  --output-root /content/echoverse_runs \
  --envs echostay,echoforge,datepickers,datepickers_ood,nested_filter,nested_filter_ood \
  --seeds 0,1,2 \
  --seed-filter 0,2 \
  --agent fara15 \
  --agent-base-url http://localhost:5002/v1/ \
  --agent-model Fara1.5-9B \
  --max-rounds 100 \
  --use-wandb
```

If your setup uses an OpenAI-compatible endpoint for the *judge*, keep W&B optional but
ensure judge credentials are available (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, etc.).

## Required secrets

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for pulling the DBs from `microsoft/Echoverse`
- optional: W&B credentials for experiment tracking (`WANDB_API_KEY`)

## Outputs

All results are written under `--output-root`:

- `echoverse_matrix_manifest.json` (aggregated run-level summary)
- `echoverse_matrix_task_results.csv` (per-task pass/fail rows)
- `SyntheticEnv-eval/...` directories containing the raw harness trajectories and
  `final_db_state.db` for each task

## Parameters to cite in protocol text

Use this exact design language in your paper/addendum to answer reviewer scope comments:

- `task_grid = {env × task} for each seed`
- `seeds = [0,1,2]` (configurable)
- `judge`: OpenAI-compatible LLM, same protocol for all envs and tasks
- `evaluation metric`: pass rate over all tasks + per-task confidence from harness verifier
- `robustness`: replicate with multiple seeds and keep the complete manifest in repo artifacts

## Notes

- `--seed-filter` optionally narrows the executed runs to a subset of `--seeds` (for example `--seeds 0,1,2 --seed-filter 0,2`).
- Use exactly one of `--agent-base-url` or `--agent-base-urls`, not both.

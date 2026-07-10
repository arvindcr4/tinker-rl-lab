# Tinker and W&B experiment registry

Generated: `2026-07-10T19:09:18.107503+00:00`

## Coverage

- Tinker training runs: **949**
- Tinker checkpoints: **1061**
- W&B projects: **17**
- W&B runs: **940**
- Exact ID matches: **41**
- Unique UUID matches: **0**
- Model/time candidates: **21**
- Unmatched Tinker runs: **887**

`candidate` is intentionally not treated as a confirmed join. Historical
Tinker runs generally lack dataset, algorithm, learning-rate, and metric
metadata, so exact correlation is possible only when a W&B run records the
Tinker training ID (or its unique UUID).

## W&B projects

| project | runs |
|---|---:|
| huggingface | 3 |
| inference | 0 |
| pesmode | 0 |
| quickstart_playground | 0 |
| rlvr-openings | 61 |
| skyrl-tinker | 3 |
| tinker-agentic-smoke | 10 |
| tinker-new-research | 29 |
| tinker-rl-lab-world-class | 174 |
| tinker-rl-scaling | 88 |
| tinker-rl-webarena | 40 |
| tinker-rl-zvf-counterfactual | 9 |
| tinker-structural-ceiling | 72 |
| webarena-eval | 26 |
| zvf-audit | 368 |
| zvf-audit-v2 | 41 |
| zvf-colab-experiments | 16 |

## Files

- `tinker_runs.jsonl`: all Tinker run metadata.
- `tinker_checkpoints.jsonl`: every checkpoint returned by Tinker.
- `wandb_runs.jsonl`: sanitized W&B metadata, config, and summary.
- `tinker_wandb_correlation.csv`: one row per Tinker run.
- `manifest.json`: counts and distributions.


## v2 (2026-07-11): Hugging Face three-way rectification

- `hf_runs.jsonl`: harvest of 33 `arvindcr4/*tinker*` HF model repos (19 unique Tinker run IDs, all present in the registry).
- `tinker_wandb_hf_correlation.csv`: v1 + `wandb_model_labels`, `wandb_model_consistent`, `hf_repos`, `hf_model_label`, `hf_model_consistent`, `resolution`.
- `RECTIFICATION.md`: resolution distribution and the conflict/arbitration quarantine list.
- Generator: `../../tinker-runs/rectify_correlation_with_hf.py` (needs `HF_TOKEN`; read-only).

Headline: 33/41 exact links confirmed (5 additionally HF-corroborated); of the 8 v1 model conflicts, **5 are HF-arbitrated W&B mislinks** — including the Nemotron↔gpt-oss ID swap (true Nemotron run: `657a920a-…`, verified via `training_run.json` on HF) — and 3 remain unresolved. 5 runs are preserved only on HF.

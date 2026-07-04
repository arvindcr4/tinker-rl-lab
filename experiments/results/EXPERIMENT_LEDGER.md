# Experiment Ledger

Master inventory of historical training/evaluation runs across Tinker and Weights & Biases.
Future campaigns should check this ledger to **omit already-run experiments** and **reuse old checkpoints**.

## Files produced

- `experiments/results/experiment_ledger.tsv` — machine-readable master ledger.
- `experiments/results/tinker_run_inventory.tsv` — Tinker run + checkpoint details.
- `experiments/results/wandb_inventory/*.tsv` — per-project W&B inventories.

## Validation notes

- Tinker inventory built by paginating `RestClient.list_training_runs` and `RestClient.list_checkpoints`.
- W&B inventory files already existed for 13 projects; row counts were validated against the live W&B API and all matched exactly.
- Missing W&B project `tinker-new-research` was discovered (entity has 14 projects, not 13) and its TSV was built from live runs.
- No existing W&B TSVs were rebuilt.

## Run counts per source

| source | project | runs |
|---|---|---|
| tinker | tinker | 844 |
| wandb | huggingface | 3 |
| wandb | pesmode | 0 |
| wandb | quickstart_playground | 0 |
| wandb | skyrl-tinker | 3 |
| wandb | tinker-agentic-smoke | 10 |
| wandb | tinker-new-research | 9 |
| wandb | tinker-rl-lab-world-class | 174 |
| wandb | tinker-rl-scaling | 88 |
| wandb | tinker-rl-webarena | 40 |
| wandb | tinker-rl-zvf-counterfactual | 9 |
| wandb | tinker-structural-ceiling | 72 |
| wandb | webarena-eval | 26 |
| wandb | zvf-audit | 368 |
| wandb | zvf-colab-experiments | 16 |
| **wandb total** | | **818** |
| **grand total** | | **1662** |

## Config-space coverage (W&B only)

- Unique models: **36**
- Unique tasks/datasets: **10**
- Unique algorithms: **7**
- Unique group sizes (G): **7**
- Distinct model × task × algo × G combinations: **121**

### Models tried

- `NousResearch/Meta-Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-8B-Base`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-4B`
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.1-Base`
- `google/gemma-2-9b-it`
- `gpt2`
- `meta-llama/Llama-3.1-70B`
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `unsloth/Llama-3.2-3B`

### Tasks/datasets tried

- `gpt_oss_low_reasoning`
- `gsm8k`
- `gsm8k_heldout`
- `held_out_eval`
- `humaneval`
- `kl_tracking`
- `llama3`
- `nemotron3_disable_thinking`
- `qwen3_instruct`
- `tool_use`

### Algorithms tried

- `GRPO`
- `PPO`
- `TRL-GRPO`
- `grpo`
- `per-group regression; continuous reward; population-standardized advantage`
- `ppo_reinforce`
- `reinforce`

### Group sizes (G) tried

- `16`
- `2`
- `32`
- `4`
- `6`
- `64`
- `8`

## Tinker caveats

The Tinker SDK (`tinker==0.22.7`) exposes **run IDs, base models, LoRA info, checkpoint times, and `tinker://` paths**, but it does **not** expose training configuration metadata (algorithm, dataset, group size, steps, learning rate, or reward metric). All historical runs in this account have `user_metadata=null`. Therefore Tinker rows in the ledger use `UNKNOWN` for task/algo/G/steps and an empty `headline_metric`. Reuse Tinker checkpoints for sampling evals by passing their `tinker://` path to `ServiceClient.create_sampling_client(model_path=...)`.

## Oldest and newest runs

- Oldest: **2026-01-08T04:55:18.876133+00:00** — `tinker` / `tinker` / `b6f141d9-e02f-5225-9cac-e646d2bafa09:train:0`
- Newest: **2026-07-04T09:38:14.031888+00:00** — `tinker` / `tinker` / `81aa65f5-f17f-562c-ae2d-b00e8931675d:train:1`

## How to use for campaign planning

1. Load `experiment_ledger.tsv`.
2. Filter on `source`, `model`, `task`, `algo`, `G`, and `steps`.
3. If a matching row exists, **omit** that experiment.
4. For Tinker rows, cross-reference `tinker_run_inventory.tsv` for reusable `tinker://` checkpoint paths.

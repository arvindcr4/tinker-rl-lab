# atropos/ — INDEX

**Purpose:** Integration layer connecting NousResearch Atropos (RL environments) with the Thinking Machines Tinker API. Runs GRPO training on Atropos envs from a local machine, plus eval harnesses, configs, and experiment notebooks.

**Key files:**
- `README.md` — quickstart: `run-api` + `launch_training.py` + an env `serve`; how to use any Atropos env with Tinker.
- `launch_training.py` — entry point that starts the Tinker trainer against the Atropos Trajectory API for a given config.
- `serve.py` — standalone Tinker inference server (OpenAI-compatible, no training/atropos).
- `train_grpo_unsloth.py` — drop-in Unsloth/TRL GRPO trainer reading the same `configs/gsm8k_*.yaml` (open-source baseline).
- `train_grpo_humaneval.py` — TRL GRPO for HumanEval (code-exec reward) and tool-use tasks via `--task`.
- `eval_reasoning_suite.py` — deterministic GSM8K/GSM1k/GSM-Symbolic eval for post-training generalization claims.
- `eval_arenahard.py`, `eval_toxicchat.py` — ArenaHard (GPT-4-judge format) and ToxicChat toxicity eval harnesses.
- `run_stats.py` — runs statistical tests (see `tinker_atropos/stats_utils.py`) on completed experiments.
- `generate_slides.py` — builds PPTX deck of results; `slides_8th_call.md`, `transcript_8th_call.md` are talk material.
- `CLAIM_SUPPORT_EXPERIMENTS.md` — experiment matrix defending reasoning claims beyond GSM8K saturation.
- `pyproject.toml`, `requirements_unsloth.txt`, `.pre-commit-config.yaml` — packaging/deps/lint.
- `run_*.sh` — orchestration scripts (start api+env+trainer). `run_experiment.sh`/`run_experiment_generic.sh` = single run; `run_all_experiments.sh`/`run_pending_experiments.sh` = batches; `run_logp_steering.sh`, `run_claim_support_*.sh`, `run_toxicchat_eval.sh` = specific suites.

**Subfolders:**
- `tinker_atropos/` — core package: trainer, config, types, environments, tests, utils (see its INDEX.md)
- `configs/` — YAML training configs per model/env + hyperparameter sweep (see its INDEX.md)
- `notebooks/` — Jupyter experiment writeups + notebook generator (see its INDEX.md)

**Find it fast:**
- to start a training run → `launch_training.py` + `configs/default.yaml`
- to run an open-source (non-Tinker) baseline → `train_grpo_unsloth.py`
- to add/modify an RL task → `tinker_atropos/environments/`
- to evaluate a checkpoint → `eval_reasoning_suite.py` / `run_claim_support_evals.sh`

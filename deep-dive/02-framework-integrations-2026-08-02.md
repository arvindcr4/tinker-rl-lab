# Deep Dive: Framework Integrations (tinker, verl, openrlhf, skyrl, trl)

## Overview
The repo models RL training as a 2-D matrix: **framework** (the "what trains": tinker, verl, openrlhf, skyrl, trl) × **backend** (the "where": local, modal, colab, vast, gcp, hfspaces). `UnifiedLauncher` (`platform_local/unified/launcher.py`) owns outer dispatch on backend; the local backend then calls `dispatch_framework()` for inner dispatch on framework. Every framework wraps a *real* upstream — no full source trees are vendored. `verl/`, `platform_modal/openrlhf/`, and `platform_hybrid/skyrl/` are 3-file launcher packages (config + trainer + bridge); the actual `verl`, `openrlhf`, `skyrl_train`, `tinker`, `trl` distributions are pip-installed lazily or inside the Modal image. A single frozen `CanonicalSpec` (Qwen3-8B · GSM8K · GRPO · 30 steps · G=8 · LoRA r=16 · β=0) is the one experiment every cell reproduces.

## Key Components
- `platform_local/unified/launcher.py:UnifiedLauncher` — outer backend dispatch + inner framework dispatch (5 `_run_<fw>` methods).
- `platform_local/unified/canonical.py:CanonicalSpec` — frozen Layer-B protocol; `framework_config_path()` resolves per-framework equivalence YAMLs.
- `platform_tinker/tinkerrl/grpo_cli.py:PRESETS` — single CLI replacing the old `grpo_*.py` menagerie; chooses dataset factory + reward class.
- `platform_tinker/tinkerrl/grpo.py:_run_one_seed` — the real in-process Tinker loop: sample G rollouts, `normalize_rewards`, `forward_backward_custom`, `optim_step`.
- `verl/trainer.py:run_verl_training` — real driver; shells out `python -m verl.trainer.main_ppo` via Hydra CLI.
- `platform_modal/openrlhf/trainer.py:run_openrlhf_training` — shells out `openrlhf.cli.train_ppo_ray --advantage_estimator group_norm` (= GRPO).
- `platform_local/trl_integrations/trainer.py:generate_trl_train_script` — codegen approach: emits a standalone checkpoint-resumable GRPO script.
- `platform_hybrid/skyrl/configs/grpo_gsm8k.yaml` + `run_skyrl_tinker.py` — recipe file consumed by `skyrl_train.entrypoints.main_base`; bridge to skyrl-gym / tinker_atropos envs.
- `platform_hybrid/experiments/modal/modal_grpo_{fw}.py` — five sister Modal H100 drivers, one per framework.
- `platform_local/unified/backends/{local,modal,vast,...}.py` — backend registry mapping each (framework, backend) cell to its driver file.

## Concepts & Decisions

### GRPO without a critic
- **What**: Group-relative advantage — sample G=8 completions per prompt, z-normalize rewards inside the group, use that as the advantage; no value head. `grpo.py:normalize_rewards` does this in 3 lines.
- **Why here**: avoids fitting a critic on a 30-step budget (critic learning is the long pole in PPO). All 5 frameworks converge on this — verl via `algorithm.adv_estimator=grpo`, openrlhf via `--advantage_estimator group_norm`, skyrl via `advantage_estimator: "grpo"`, tinker/trl inline.
- **Trade-off**: high-variance advantage at small G; β=0 (no KL) in the canonical spec means the policy can drift.

### Three integration shapes
Each framework needs a different wrapper shape because the upstreams expose different surfaces:
- **In-process API** (tinker, trl): the loop lives in this repo. `grpo.py:_run_one_seed` calls `tinker.ServiceClient` directly; TRL runs in the same Python via `generate_trl_train_script`.
- **Subprocess + CLI flag** (verl, openrlhf): the upstream owns the training loop; we shell out to its CLI (`verl.trainer.main_ppo`, `openrlhf.cli.train_ppo_ray`) and recover metrics from W&B afterward.
- **Recipe file + external checkout** (skyrl): not pip-installable; we ship a YAML and shell out to `skyrl_train.entrypoints.main_base @config.yaml` inside a cloned `NovaSky-AI/SkyRL` checkout.

### The PYTHONPATH-shadow trap
- **What**: the repo's own `verl/` and `platform_modal/openrlhf/` launcher packages shadow the pypi `verl`/`openrlhf` packages. `python -m verl.trainer.main_ppo` would resolve to *this file*, recursing.
- **Fix**: both real drivers set `cwd=output_dir` (keeping repo root off `sys.path`) and `env["PYTHONPATH"]=""`. Documented in `verl/trainer.py:run_verl_training`.
- **Alternative considered**: renaming the launcher packages was rejected — the names are part of the published matrix contract.

### W&B as the cross-framework telemetry bus
- **What**: because verl/openrlhf own their loops in a subprocess, the parent can't read step metrics directly. After the subprocess exits, `run_verl_training` queries `wandb.Api().runs(...)` and pulls `critic/rewards/mean` history.
- **Why**: sidesteps parsing framework-specific stdout; gives a uniform `reward_trace` list for the aggregator.
- **Trade-off**: requires W&B to be reachable; brittle to schema changes. A VRAM monkey-patch on `wandb.log` (`system/vram_peak_allocated_gb`) is layered on top — pragmatic but fragile.

### LoRA as the parameter-efficiency default
- **Why here**: 8B models on single H100 with 30-step budgets — full fine-tune is wasteful and noisy. Tinker uses `create_lora_training_client(rank=16, target=all-linear)`; TRL emits PEFT config; verl/openrlhf pass `LoRA` through their actor configs.
- **Trade-off**: underestimates full-FT ceiling; documented in `framework_config_dumps/README.md` as a known equivalence caveat.

### Dryrun fallback for CI smoke
- **What**: `VERLTrainer.run()` and `OpenRLHFTrainer.run()` produce a *seeded deterministic* reward trace when the upstream isn't installed, clearly tagged `mode: "dryrun"`. Lets `framework_comparison.json` regenerate in CI without GPUs.
- **Why**: decouples aggregator/figure scripts from real runs; the real Modal runs overwrite the dryrun values.

## Related Code
The launcher is wired to `platform_local/unified/backends/` (six `Backend` subclasses; `base.py:LaunchPlan` is the dry-run output). The MODAL backend dispatches per-framework to `platform_hybrid/experiments/modal/modal_grpo_{trl,tinker,verl,openrlhf,skyrl}.py` — five sister scripts that share a Modal image recipe (HF/W&B/Tinker secrets, repo mount at `/root/tinker-rl-lab`, H100 GPU). The vast backend (`platform_hybrid/skyrl/backends/vastai_runner.py`, ~494 lines) is the most substantial provisioner and pins the same `skyrl_train-v0.4.0` tag the Modal driver clones. Per-framework equivalence configs in `platform_hybrid/experiments/framework_config_dumps/<fw>_qwen3_8b_gsm8k.yaml` document field-by-field equivalence (and the 11/44 Tinker-managed fields that can't be exposed — the "closed-source confound" called out in `serve.py` and the openrlhf TODOs). Results from all cells land in `experiments/results/` for `aggregate_framework_comparison.py`.

## Start Here
1. `platform_local/unified/launcher.py` — see the framework × backend matrix in ~326 lines.
2. `platform_tinker/tinkerrl/grpo.py` (the `_run_one_seed` region) — the only in-process loop; clearest expression of GRPO.
3. `verl/trainer.py:run_verl_training` — the subprocess-plus-W&B-recovery pattern that openrlhf mirrors.

---
*Generated by AntiVibe (full-repo pass) · 2026-08-02*

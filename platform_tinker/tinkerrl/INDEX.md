# tinkerrl/ — INDEX

**Purpose:** Consolidated, deepened GRPO training loop against the Tinker API — the single source of truth that replaced the copy-pasted top-level `grpo_exp_*.py` / `grpo_100_*.py` / `grpo_gsm8k_base.py` / `grpo_tooluse_tinker.py` scripts.

**Key files:**
- `grpo.py` — core module: `GRPOConfig`, `TrainingExample`, `InMemoryDataset`, reward adapters (`MathReward`, `ToolCallReward`), `normalize_rewards`, `make_grpo_loss_fn`, and `run_grpo` (owns seed loop, sampling, advantages, optim step, checkpoints).
- `grpo_cli.py` — CLI entrypoint (`python -m tinkerrl.grpo_cli --preset <name>`); presets encode the old per-experiment configs; flags override fields.
- `__init__.py` — re-exports the public `grpo` interface.

**Find it fast:**
- to run a GRPO experiment → `python -m tinkerrl.grpo_cli --preset <tooluse_synth|gsm8k>`
- to change loss/advantage logic → `grpo.py`

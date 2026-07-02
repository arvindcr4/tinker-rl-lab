# verl/ — INDEX

**Purpose:** Integration of **verl** (Volcano Engine RL, HybridFlow, Ray + vLLM) with tinker-rl-lab. Provides a real GRPO runner for cross-framework comparison, with a seeded `dryrun` fallback when verl isn't installed (keeps CI/`framework_comparison.json` reproducible).

**Key files:**
- `trainer.py` — `VERLTrainer`: GRPO via verl's PPO trainer + group-norm advantage on single-GPU local backend; loop mirrors TRL/Tinker for apples-to-apples reward traces. Exposes `run_verl_training`, `run`.
- `config.py` — `VERLConfig` (+ `VERLModelConfig`, `VERLOptimizerConfig`, `VERLAlgorithmConfig`), pydantic, YAML-loadable.
- `__init__.py` — exports `VERLTrainer`, `VERLConfig`, `run_verl_training`, `run`.

**Find it fast:**
- to run/compare verl GRPO → `trainer.py`
- real H100 results are produced by → `experiments/modal/modal_grpo_verl.py`

# tests/ — INDEX

**Purpose:** Lightweight unit/smoke tests for the core RL code and repo utilities (fast, CPU-only).

**Key files:**
- `test_grpo_module.py` — unit tests for the consolidated `tinkerrl.grpo` module (config, datasets, reward adapters, run result; uses mocks).
- `test_grpo_loss.py` — numerical tests for GRPO reward normalization / loss math (self-contained `normalize_rewards`).
- `test_experiments.py` — smoke test that files under `experiments/implementations/` parse (AST) and are well-formed.
- `test_seed.py` — verifies `utils.seed.set_global_seed` determinism.
- `test_stats.py` — smoke tests that `utils.stats` imports and functions.

**Find it fast:**
- to validate core GRPO logic → `test_grpo_module.py`, `test_grpo_loss.py`
- run all → `pytest tests/`

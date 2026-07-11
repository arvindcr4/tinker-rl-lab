# tests/ — INDEX

**Purpose:** Lightweight unit/smoke tests for the core RL code and repo utilities (fast, CPU-only).

**Key files:**
- `test_grpo_module.py` — unit tests for the consolidated `tinkerrl.grpo` module (config, datasets, reward adapters, run result; uses mocks).
- `test_grpo_loss.py` — numerical tests for GRPO reward normalization / loss math (self-contained `normalize_rewards`).
- `test_experiments.py` — smoke test that files under `platform_hybrid/experiments/implementations/` parse (AST) and are well-formed.
- `test_seed.py` — verifies `utils.seed.set_global_seed` determinism.
- `test_stats.py` — smoke tests that `utils.stats` imports and functions.
- `test_peft_support.py` — PEFT validation, CLI behavior, and resumable generated TRL scripts.
- `test_experiments_next_quality.py` — experiment quality metrics and restart-safety contracts.
- `test_provenance.py` — moved-source resolution and missing-source enforcement for MIN-REPORT records.
- `test_figure_module.py` — results adapters, fallback provenance, and figure-module routing.
- `test_audit_runner.py` — structured audit results and in-process suite collection.
- `test_grpo_cli.py` — legacy preset compatibility and checkpoint-manifest safety.

**Find it fast:**
- to validate core GRPO logic → `test_grpo_module.py`, `test_grpo_loss.py`
- run the CI-equivalent repository gate → `make check`
- run tests only → `pytest tests/`

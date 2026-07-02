# unified/ — INDEX

**Purpose:** Single launcher that dispatches RL training across all integrated frameworks (skyrl, tinker/atropos, verl, openrlhf, trl) with a common CLI, algorithm set, and `TrainingResult` schema.

**Key files:**
- `__init__.py` — **the launcher itself** (no separate `launcher.py`): defines `UnifiedLauncher` (per-framework `_run_*` dispatch, `FRAMEWORKS`/`ALGORITHMS` tables, `TrainingResult`) and `main()`. Run as `python -m unified.launcher ...` (module resolves here).

**Find it fast:**
- to launch any framework → `python -m unified.launcher --framework <skyrl|tinker|verl|openrlhf|trl> --model <m> --algorithm <grpo|ppo|...>`
- to add a new framework → extend `FRAMEWORKS` + add a `_run_<name>` method in `__init__.py`

# atropos/tinker_atropos/ — INDEX

**Purpose:** Core package. Bridges Atropos RL environments to the Tinker API: a GRPO trainer that also serves inference, config/type schemas, and the environment implementations.

**Key files:**
- `trainer.py` — `TinkerAtroposTrainer`: RL training + inference via Tinker; connects to Atropos Trajectory API. Also hosts a FastAPI OpenAI-compatible server (`/v1/completions`, `/chat/completions`, `/generate`, `/logprobs`). `main()` is the run loop.
- `config.py` — pydantic configs: `EnvConfig`, `OpenAIServerConfig`, `TinkerConfig`, `TinkerAtroposConfig` (loads the YAML in `configs/`); `generate_run_suffix()` for unique wandb names.
- `types.py` — request/response pydantic models for the completions/chat endpoints (`CompletionRequest`, `ChatMessage`, etc.).
- `stats_utils.py` — GRPO statistics toolkit: bootstrap CI, two-proportion z-test, Spearman trend, Mann-Whitney, Cohen's d, Chow structural-break/phase-transition tests, `run_full_analysis`/`print_report`.
- `__init__.py` — package docstring listing outstanding adversarial-review limitations (ZVF, short-run snapshots, closed-source confound, generalization, single-seed).

**Subfolders:**
- `environments/` — Atropos GRPO env implementations: GSM8K, MATH, tool-use, HumanEval, logp-steering, MoE-routing, curriculum, agentic (see its INDEX.md)
- `tests/` — pytest suite for serve/logprobs/distillation/logp-steering (see its INDEX.md)
- `utils/` — checkpoint weight download helper (see its INDEX.md)

**Find it fast:**
- to change training loop / GRPO update → `trainer.py`
- to understand a config field → `config.py`
- to run significance tests on rewards → `stats_utils.py`

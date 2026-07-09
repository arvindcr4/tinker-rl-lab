# skyrl/ — INDEX

**Purpose:** SkyRL (NovaSky-AI) integration with the Tinker API. Runs SkyRL tx — a *local* Tinker API server — on your own/remote GPUs, plus a bridge to the hosted Tinker API for SkyRL-style GRPO.

**Key files:**
- `run_skyrl_tinker.py` — SkyRL↔hosted-Tinker bridge; runs SkyRL-style GRPO against hosted Tinker API (config-driven, CLI overrides for model/env/steps).
- `README.md` — setup, architecture diagram, `skyrl.tinker.api` local-server commands, env vars.

**Subfolders:**
- `backends/` — remote compute launchers (vast.ai). See its INDEX.md.
- `configs/` — YAML training configs (GRPO on GSM8K/Math, hosted-Tinker). See its INDEX.md.
- `notebooks/` — `skyrl_colab_training.ipynb`, run SkyRL tx in Colab (T4/A100).

**Find it fast:**
- to run GRPO on hosted Tinker → `run_skyrl_tinker.py`
- to start the local Tinker API server → see README `skyrl.tinker.api` section
- to provision GPUs remotely → `backends/vastai_runner.py`

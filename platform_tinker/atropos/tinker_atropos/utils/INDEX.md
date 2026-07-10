# atropos/tinker_atropos/utils/ — INDEX

**Purpose:** Small helper utilities for the package.

**Key files:**
- `download_weights.py` — downloads trained LoRA/sampler checkpoints from Tinker to local `.tar` archives. Edit the `RUN_IDS` list, then run; fetches `tinker://<run_id>/sampler_weights/final` per run.
- `__init__.py` — package marker; docstring tracks outstanding adversarial-review TODOs (ZVF metric, short-run snapshots, etc.).

**Find it fast:**
- to pull a finished run's weights locally → `download_weights.py` (set `RUN_IDS`)

# experiments/tool_use_zvf_sweep/ — INDEX

**Purpose:** A Tinker sweep probing Zero-Variance-Fraction (ZVF) behavior on tool-use (format-gated) tasks, contrasting two reward designs (v1 sparse / v2 partial-credit) — motivated by the finding that ZVF saturates at 1.0 for format-gated tasks (see TODOs about ERF alternative).

**Key files:**
- `gen_configs.py` — generates the sweep: (qwen3_4b, qwen3_8b) × (v1, v2 reward) × seeds → `configs/*.yaml`.
- `run_sweep.py` — sequential Tinker runner with a USD cost cap (`--cap-usd`), one run at a time.
- `cost_estimate.py` — pre-flight per-run cost estimate.
- `results_to_tex.py` — splices ZVF / pass@1 numbers from a results CSV into the paper appendix table.
- `manifest.tsv` — index of the 12 real configs (config, reward, seed, tokenizer).

**Subfolders:**
- `configs/` — generated sweep YAMLs (smoke + 4b/8b × v1/v2 × seeds 42–44). See its INDEX.md.

**Find it fast:**
- regenerate configs → `gen_configs.py`
- launch under a budget → `run_sweep.py --cap-usd N configs/<file>.yaml`
- config↔reward↔seed map → `manifest.tsv`

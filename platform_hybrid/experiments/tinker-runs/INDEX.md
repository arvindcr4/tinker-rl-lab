# experiments/tinker-runs/ — INDEX

**Purpose:** GRPO experiment campaigns run against the **Tinker cloud API** (the "bitter lesson" world-class campaign: every model × GSM8K/tool-use, multi-seed, group-size + ablation sweeps). Scripts here launch runs; `logs/` + `results/` hold their output.

**Key files:**
- `campaign_v2.py` — canonical Tinker SDK GRPO campaign (`forward_backward_custom()` + `optim_step()`); the `run_experiment()`/`EXPERIMENT` schema reused elsewhere.
- `massive_campaign.py` — max-parallel launch: all untested models, base-vs-instruct pairs, G-sweep, multi-seed, cross-task.
- `wave6_ablations.py` — Qwen3-8B GSM8K 1-D sweeps (temperature / LoRA rank / batch), 200 steps; `wave6_resume.py` reruns only missing runs.

**Subfolders:** (see each INDEX.md)
- `scripts/` — individual/parameterized run scripts (grpo_gsm8k_base, exp A–D, 100-step variants, parallel runner, 70B seeds).
- `results/` — per-run result JSONs (scale/frontier/moe/cross_tool/arch + campaign dumps).
- `logs/` — plaintext `.log` training console output.

**Find it fast:**
- reusable Tinker GRPO loop → `campaign_v2.py`
- hyperparameter ablations → `wave6_ablations.py`
- a specific run's numbers → `results/<name>.json`

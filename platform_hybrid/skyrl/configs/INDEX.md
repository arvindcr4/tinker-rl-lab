# skyrl/configs/ — INDEX

**Purpose:** YAML run configs consumed by `skyrl/run_skyrl_tinker.py` (model, training, env, GRPO knobs).

**Key files:**
- `grpo_gsm8k.yaml` — GRPO training config targeting the GSM8K math environment.
- `grpo_math.yaml` — GRPO config for the general math-RL environment.
- `tinker_hosted.yaml` — config pointing SkyRL-style GRPO at the *hosted* Tinker API (default for `run_skyrl_tinker.py`).

**Find it fast:**
- to change model/steps/group-size → edit the relevant `*.yaml` (or override via CLI flags)

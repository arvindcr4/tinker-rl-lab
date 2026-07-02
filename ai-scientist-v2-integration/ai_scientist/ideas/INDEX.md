# ai-scientist-v2-integration/ai_scientist/ideas/ — INDEX

**Purpose:** Experiment code templates + idea specs the AI Scientist v2 executes via `exec()` (no `__main__` guard; graceful fallback when API keys missing).

**Key files:**
- `tinker_grpo_rl.py` — GRPO on GSM8K via the Tinker SDK (managed infra, no local GPU); tunables: MODEL, LORA_RANK, GROUP_SIZE, STEPS, LR, NUM_SEEDS, reward_fn.
- `trl_local_grpo.py` — GRPO run locally via HuggingFace TRL (no Tinker key needed); for fast iteration / exhausted credits.
- `trl_local_grpo.json` — idea spec pairing with the local TRL template.
- `tool_use_reward_design.json` — idea: 4-tier dense reward for tool-use RL to fix ZVF saturation (vs binary reward).

**Find it fast:**
- to run GRPO on managed infra → `tinker_grpo_rl.py`
- to run GRPO locally → `trl_local_grpo.py`
- to explore dense tool-use rewards → `tool_use_reward_design.json`

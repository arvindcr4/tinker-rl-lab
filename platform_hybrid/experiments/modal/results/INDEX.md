# experiments/modal/results/ — INDEX

**Purpose:** Output JSON from the Modal H100 runs (consumed by the top-level aggregator and cited in the paper's PPO/held-out rows).

**Key files:**
- `modal_parallel_results.json` — single dict keyed by experiment id: `ppo_qwen3-8b`, `ppo_llama-8b-inst`, `humaneval_qwen3-8b`, `kl_qwen3-8b`, `heldout_qwen3-32b`, `heldout_qwen3.5-27b`. Each value holds the reward trace / eval metrics. This is the canonical source for the corrected PPO numbers (e.g. `ppo_gsm8k_Qwen3-8B_s42`).

**Find it fast:**
- canonical Modal PPO trace → `modal_parallel_results.json` → key `ppo_qwen3-8b`

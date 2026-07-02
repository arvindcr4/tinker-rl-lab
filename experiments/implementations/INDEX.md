# experiments/implementations/ — INDEX

**Purpose:** The core Tinker recipes (Math RL, GSM8K, Chat SFT, DPO-shorter, distillation) re-implemented across many RL libraries, so results aren't tied to one stack. All use the same verifiable reward (1.0 correct / 0.0 wrong-valid / -0.1 invalid).

**Key files:**
- TRL (LLM training, best for these tasks): `trl_grpo_math.py`, `trl_gsm8k_math.py`, `trl_chat_sft.py`, `trl_dpo_shorter.py`, `trl_distillation.py`.
- Matched GSM8K baselines (isolate the RL algorithm): `trl_ppo_gsm8k_baseline.py`, `trl_sft_gsm8k_baseline.py`.
- Classic RL (agent libs, shown for pattern; ~0% on language tasks): `sb3_ppo_math.py`, `cleanrl_ppo_math.py`, `tianshou_ppo_math.py`.
- High-perf / offline RL configs: `pufferlib_math.py`, `rl_games_math.py`, `d3rlpy_offline.py`.
- `requirements.txt`, `run_all.sh` — deps + run-everything driver.
- `README.md` — library recommendation matrix + Tinker→lib hyperparameter mapping.

**Subfolders:**
- `collab/` — teammate Colab tool-calling experiments (SFT/GRPO/eval, Qwen2.5 0.5B/1.5B/3B). See its INDEX.md.

**Find it fast:**
- recommended starting point → `trl_grpo_math.py`
- apples-to-apples GRPO vs PPO vs SFT → the two `trl_*_gsm8k_baseline.py`
- why classic PPO fails on LLMs → `README.md` "Important Notes"

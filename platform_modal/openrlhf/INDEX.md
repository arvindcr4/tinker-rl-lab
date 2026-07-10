# platform_modal/openrlhf/ — INDEX

**Purpose:** Integration of **OpenRLHF** (Ray + vLLM distributed RL; PPO/DAPO/REINFORCE++) with tinker-rl-lab. Real GRPO runner for cross-framework comparison, with a seeded deterministic `dryrun` fallback when openrlhf isn't installed.

**Key files:**
- `trainer.py` — `OpenRLHFTrainer`: GRPO via OpenRLHF's `ppo_ray` config with `advantage_estimator=group_norm`; loop mirrors TRL/Tinker. Exposes `run_openrlhf_training`, `run`.
- `config.py` — `OpenRLHFConfig` (pydantic model/optimizer/algorithm settings).
- `__init__.py` — exports trainer/config/run; carries a TODO list of adversarial-review limitations (ZVF fragility, closed-source confound, generalization, multi-seed).

**Find it fast:**
- to run/compare OpenRLHF GRPO → `trainer.py`

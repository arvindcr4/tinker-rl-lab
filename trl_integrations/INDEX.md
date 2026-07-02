# trl_integrations/ — INDEX

**Purpose:** Integration of **HuggingFace TRL** (Transformer RL: GRPO/PPO/DPO/REINFORCE) with tinker-rl-lab. The reference "same-stack" runner other frameworks' loops are mirrored against.

**Key files:**
- `trainer.py` — `TRLTrainer`: unified interface over TRL GRPO/PPO/DPO/REINFORCE; single- or multi-GPU (DeepSpeed), LoRA or full-parameter; tracks reward/loss history.
- `config.py` — `TRLConfig` (model, algorithm, optimizer, LoRA settings).
- `__init__.py` — exports `TRLTrainer`, `TRLConfig`.

**Find it fast:**
- to run HF TRL GRPO/PPO/DPO → `trainer.py`
- to tweak LoRA / algorithm knobs → `config.py`

# trl_integrations/ — INDEX

**Purpose:** Local integration helpers for **HuggingFace TRL**. The unified launcher is a smoke-test scaffold; `generate_trl_train_script()` produces the runnable GRPO entry point.

**Key files:**
- `trainer.py` — trainer factories plus the validated GRPO script generator.
- `config.py` — `TRLConfig` (model, algorithm, optimizer, quantization, and tuning settings).
- `__init__.py` — exports `TRLTrainer`, `TRLConfig`.
- `../unified/peft_utils.py` — applies LoRA, prefix tuning, P-tuning, prompt tuning, or BitFit and writes compact BitFit checkpoints.

**Find it fast:**
- to generate a 4-bit LoRA/QLoRA-style GRPO script → `python -m platform_local.unified.launcher --framework trl --algorithm grpo --peft-method lora --load-in-4bit --train-data train.json --generate-script train_grpo.py`
- to select another supported tuning method → replace `lora` with `prefix_tuning`, `p_tuning`, `prompt_tuning`, or `bitfit`
- PPO/DPO factory helpers exist in `trainer.py`, but script generation intentionally rejects them until complete runnable templates are implemented

# atropos/configs/ — INDEX

**Purpose:** YAML training configs (Atropos format with `env`, `openai`, and `tinker` sections). One file per model×env×variant; consumed by `launch_training.py`, the env `serve` command, and the Unsloth/TRL trainers.

**Key files:**
- `default.yaml` — canonical template: Llama-3.1-8B on GSM8K, 50 steps, LoRA rank 32, lr 4e-5.
- `usage.md` — how the three config sections map to trainer/env/inference-server.
- `gsm8k_*.yaml` — GSM8K across models (qwen 0.6b/1.7b/4b/8b/14b/30b-moe, llama 3b/8b/8b-base) and ablations (`_lora8/16/64`, `_seed1/2/3`, `_100steps`, `_no_prefix`).
- `math_*.yaml`, `math_curriculum_qwen8b.yaml` — MATH competition configs; `humaneval_qwen_8b.yaml`, `tool_use_*.yaml` — code/tool-use tasks.
- `bootstrap_threshold_{easy,hardest}.yaml` — E5 seed-signal control; `moe_routing_temp0_3/1_0.yaml` — E6 routing-variance; `logp_steering_qwen3_30b.yaml` — self-distillation; `quick_test.yaml` — fast smoke test.

**Subfolders:**
- `sweep_results/` — exhaustive hyperparameter sweep configs + runner (see its INDEX.md)

**Find it fast:**
- to pick a training config → match filename `<env>_<model>[_<variant>].yaml`
- to understand config sections → `usage.md`
- to run a fast sanity check → `quick_test.yaml`

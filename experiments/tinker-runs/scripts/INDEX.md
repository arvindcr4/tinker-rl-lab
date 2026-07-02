# experiments/tinker-runs/scripts/ — INDEX

**Purpose:** Individual + parameterized Tinker GRPO run scripts (the building blocks the campaigns in `../` orchestrate). Most wrap the same GRPO loop with codecarbon emissions tracking.

**Key files:**
- `grpo_gsm8k_base.py` — parameterized GSM8K GRPO (`--model --seed --rank --steps`); core reusable script. `grpo_gsm8k_base_patched.py` = same + a triton-shim for Python 3.14.
- `grpo_100_{math,synthetic,xlam}.py` — 100-step GRPO on MATH / synthetic 5-tool / xlam-60k data.
- `grpo_exp_a_baseline.py` (LR=3e-5,G=8,temp=0.8,rank=32) / `grpo_exp_d_xlam.py` — the A–D ablation runs.
- `world_class_experiments.py` — full model-catalog × task × loss-function suite (multi-model scaling, MoE vs dense, loss comparison).
- `tinker_parallel_runner.py` — ThreadPoolExecutor parallel launcher w/ W&B + HF checkpointing.
- `run_llama33_70b_seeds.py` / `retry_llama33_70b_seeds.py` — Llama-3.3-70B multi-seed launch + failed-seed retry. `run_missing.sh` = shell helper.

**Find it fast:**
- run one GSM8K GRPO config → `grpo_gsm8k_base.py`
- launch the whole catalog → `world_class_experiments.py` / `tinker_parallel_runner.py`

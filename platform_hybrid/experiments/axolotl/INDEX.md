# experiments/axolotl/ — INDEX

**Purpose:** Axolotl configs for matched SFT / GRPO / Dr.GRPO baselines on Qwen3-8B + GSM8K, runnable on any GPU server (HF Spaces / RunPod / Lambda) — addresses "no matched baseline" and "GRPO is obsolete" reviewer criticisms. All share identical hyperparameters (rank 32, lr 4e-5, batch 128) for fair comparison.

**Key files:**
- `sft_gsm8k_qwen8b.yaml` — SFT (QLoRA, 24 GB). `sft_gsm8k_qwen8b_full_lora.yaml` — SFT (LoRA, 80 GB, no quant).
- `grpo_gsm8k_qwen8b.yaml` — GRPO via Axolotl+vLLM (2 GPU). `grpo_gsm8k_qwen8b_dr_grpo.yaml` — Dr.GRPO variant.
- `gsm8k_rewards.py` — binary boxed-answer correctness reward (matches Tinker GRPO).
- `gsm8k_sft_transform.py` — formats GSM8K as question → CoT + \boxed{answer}.
- `README.md` — install + run commands (`axolotl train ...`, vllm-serve), GPU/cost matrix.

**Find it fast:**
- GRPO-vs-SFT matched baseline → the two `*_gsm8k_qwen8b.yaml` pair
- reward definition → `gsm8k_rewards.py`

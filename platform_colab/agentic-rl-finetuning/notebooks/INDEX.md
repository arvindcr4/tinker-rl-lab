# agentic-rl-finetuning/notebooks/ — INDEX

**Purpose:** Two-stage Colab (T4/A100) fine-tuning pipeline via Axolotl for Qwen3-4B/14B (unsupported by Tinker RL).

**Key files:**
- `axolotl_sft_qlora.ipynb` — Stage 1: SFT with QLoRA (LR 3e-6–1e-5, LoRA rank 32, sequence packing, user-turn masking).
- `axolotl_dpo_experiment.ipynb` — Stage 2: DPO on the SFT checkpoint (DPO LR far below SFT, beta ~0.1). Requires Stage 1 output first.

**Find it fast:**
- to run SFT first → `axolotl_sft_qlora.ipynb`
- to run preference optimization → `axolotl_dpo_experiment.ipynb`

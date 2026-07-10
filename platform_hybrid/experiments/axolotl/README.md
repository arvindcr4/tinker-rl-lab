# Axolotl Experiment Configs

Matched baselines for the GRPO scaling study, runnable on HuggingFace Spaces,
RunPod, or any GPU server. These address reviewer criticism #8 ("no matched
baseline showing GRPO is the reason") and #15 ("GRPO is obsolete").

## Configs

| Config | Method | Model | GPU Req | Purpose |
|--------|--------|-------|---------|---------|
| `sft_gsm8k_qwen8b.yaml` | SFT (QLoRA) | Qwen3-8B | 24 GB (A10G) | SFT baseline |
| `sft_gsm8k_qwen8b_full_lora.yaml` | SFT (LoRA) | Qwen3-8B | 80 GB (A100) | SFT baseline (no quant) |
| `grpo_gsm8k_qwen8b.yaml` | GRPO (QLoRA) | Qwen3-8B | 2x 24 GB | GRPO via Axolotl (not Tinker) |
| `grpo_gsm8k_qwen8b_dr_grpo.yaml` | Dr. GRPO | Qwen3-8B | 2x 24 GB | Newer GRPO variant |

## Quick Start

### Install Axolotl

```bash
pip install axolotl[vllm]
```

### SFT Baseline (single GPU)

```bash
# QLoRA on A10G / T4 (24 GB)
axolotl train experiments/axolotl/sft_gsm8k_qwen8b.yaml

# Full LoRA on A100 (80 GB)
axolotl train experiments/axolotl/sft_gsm8k_qwen8b_full_lora.yaml
```

### GRPO (2 GPUs)

```bash
# Terminal 1: vLLM server on GPU 1
CUDA_VISIBLE_DEVICES=1 axolotl vllm-serve experiments/axolotl/grpo_gsm8k_qwen8b.yaml

# Terminal 2: Training on GPU 0
CUDA_VISIBLE_DEVICES=0 axolotl train experiments/axolotl/grpo_gsm8k_qwen8b.yaml
```

### Dr. GRPO variant (2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=1 axolotl vllm-serve experiments/axolotl/grpo_gsm8k_qwen8b_dr_grpo.yaml
CUDA_VISIBLE_DEVICES=0 axolotl train experiments/axolotl/grpo_gsm8k_qwen8b_dr_grpo.yaml
```

## Running on HuggingFace Spaces

1. Create a Space with **A100 GPU** ($2.50/hr) or **2x A10G** ($3.00/hr)
2. Install: `pip install axolotl[vllm]`
3. Clone this repo and run the commands above

## Running on RunPod / Lambda

1. Rent an A100 80GB ($1.39-1.48/hr) or 2x A10G
2. Install: `pip install axolotl[vllm]`
3. Clone and run

## Comparison Matrix

All experiments use identical hyperparameters for fair comparison:

| Param | Value | Notes |
|-------|-------|-------|
| Model | Qwen3-8B base | Same across all methods |
| LoRA rank | 32 | Same capacity budget |
| LR | 4e-5 | Same optimization |
| Effective batch | 128 | Same gradient signal |
| Data | GSM8K train | Same distribution |
| Eval | GSM8K test (1319) | Greedy decoding, \\boxed{} extraction |

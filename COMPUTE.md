# Compute Resource Documentation

> Required by NeurIPS Paper Checklist Item 8.
> Reference: Henderson et al., "Deep Reinforcement Learning that Matters" (2018)

## Hardware

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA A100 (40GB and 80GB variants) |
| CPU | AMD EPYC 7763 64-Core (or equivalent) |
| RAM | 256 GB |
| Storage | NVMe SSD |
| Interconnect | NVLink (for multi-GPU) |

## Per-Experiment Compute

| Experiment | Model | GPU Type | GPUs | Time/Seed | Seeds | Total GPU-hrs |
|-----------|-------|----------|------|-----------|-------|---------------|
| Math RL (Arithmetic) | Llama-3.2-1B | A100-40GB | 1 | ~30 min | 5 | 2.5 |
| Math RL (GSM8K) | Llama-3.1-8B | A100-80GB | 1 | ~4 hrs | 5 | 20 |
| Chat SFT | Llama-3.2-1B | A100-40GB | 1 | ~2 hrs | 5 | 10 |
| DPO Shorter | Qwen3-0.6B | A100-40GB | 1 | ~1 hr | 5 | 5 |
| Distillation (Off-Policy) | Llama-3.2-1B | A100-40GB | 1 | ~1.5 hrs | 5 | 7.5 |
| Distillation (On-Policy) | Llama-3.2-1B | A100-80GB | 1 | ~3 hrs | 5 | 15 |
| Atropos GSM8K (Llama 3B) | Llama-3.2-3B | A100-80GB | 1 | ~3 hrs | 5 | 15 |
| Atropos GSM8K (Llama 8B) | Llama-3.1-8B | A100-80GB | 1 | ~5 hrs | 5 | 25 |
| Atropos GSM8K (Qwen 4B) | Qwen3-4B | A100-80GB | 1 | ~3 hrs | 5 | 15 |
| Atropos GSM8K (Qwen 8B) | Qwen3-8B | A100-80GB | 1 | ~5 hrs | 5 | 25 |
| Atropos GSM8K (Qwen 14B) | Qwen3-14B | A100-80GB | 2 | ~6 hrs | 5 | 60 |
| Atropos GSM8K (Qwen 30B MoE) | Qwen3-30B-A3B | A100-80GB | 4 | ~8 hrs | 3 | 96 |

## Summary

| Category | GPU-Hours |
|----------|-----------|
| **Reported experiments (in paper)** | ~296 |
| **Preliminary / failed experiments** | ~100 |
| **Hyperparameter sweeps** | ~50 |
| **Total project compute** | **~446 A100 GPU-hours** |

## Cloud Provider

- **Primary**: Tinker API (Thinking Machines) — handles GPU allocation for RL training
- **Baseline runs**: Google Cloud Platform (GCP) — a2-highgpu-1g (1x A100-40GB), a2-highgpu-2g (2x A100-80GB)
- **Atropos inference**: Self-hosted vLLM/SGLang on cloud VMs

## Cost Estimate

| Provider | Rate | Hours | Estimated Cost |
|----------|------|-------|---------------|
| Tinker API | See [rate card](https://tinker-console.thinkingmachines.ai/rate-card) | ~200 hrs | Variable |
| GCP A100-40GB | ~$3.67/hr | ~100 hrs | ~$367 |
| GCP A100-80GB | ~$5.00/hr | ~146 hrs | ~$730 |
| **Total estimated cost** | | | **~$1,100 + Tinker credits** |

## Carbon Footprint

Following [Strubell et al. (2019)](https://arxiv.org/abs/1906.02243) guidelines:

- Total compute: ~446 A100 GPU-hours
- A100 TDP: 400W
- PUE factor: ~1.1 (GCP data centers)
- Energy: 446 × 0.4 kW × 1.1 ≈ **196 kWh**
- CO₂ (US average grid): 196 × 0.39 kg/kWh ≈ **76 kg CO₂**

## Software Versions

See `requirements.txt` for exact versions. Key versions:

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.3.x–2.4.x |
| CUDA | 12.4 |
| Transformers | 4.46.x–4.49.x |
| TRL | 1.0.x–1.1.x |
| Accelerate | 1.0.x–1.1.x |

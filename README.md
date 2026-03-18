# Tinker RL Lab

A consolidated research repository for Reinforcement Learning experiments with Large Language Models, integrating the Tinker platform, Atropos environments, and RL Gym implementations.

## Overview

This repository consolidates multiple research projects focused on:
- **Tinker Platform Experiments**: RL training, SFT, preference learning, and distillation
- **Atropos Integration**: Connecting Atropos environments with Tinker API
- **RL Gym**: Google Drive and browser automation tasks for RL research

## Repository Structure

```
tinker-rl-lab/
├── grpo-results/         # GRPO RL experiment results and metrics
├── agentic-rl-finetuning/ # Agentic RL fine-tuning research
├── capstone-literature-survey/ # Literature Survey: RL for LLMs (GRPO Scaling)
├── experiments/           # Tinker RL Cookbook experiments
│   ├── notebooks/        # Jupyter notebooks for each experiment
│   ├── implementations/  # RL implementations (PPO, DPO, GRPO, etc.)
│   ├── results/          # Training metrics
│   └── logs/             # Training output logs
│
├── atropos/              # Tinker-Atropos integration
│   ├── tinker_atropos/   # Core package
│   │   ├── environments/ # GSM8K, Math, LogP steering
│   │   ├── trainer.py    # Tinker trainer
│   │   └── config.py     # Configuration management
│   ├── configs/          # YAML configurations
│   └── notebooks/        # Analysis notebooks
│
├── rl-gym/               # RL Gym (Google Drive & Browser tasks)
│   ├── gcloud_drive/     # Google Drive integration
│   └── docs/             # Task documentation
│
└── rl-master/            # RL Master (Task execution)
    ├── client_for_your_service.py
    └── gcloud_drive/
```

## Components

### 1. Experiments (Tinker RL Cookbook)

PES LLM Research Project experiments using the [Tinker](https://thinkingmachines.ai/tinker) platform.

| Recipe | Description | Status |
|--------|-------------|--------|
| Math RL (Arithmetic) | Train model to add numbers | Complete - 100% accuracy |
| Chat SL | Supervised fine-tuning on NoRobots | Complete |
| Preference Shorter | Train for concise responses | Complete |
| Distillation Off-Policy | SFT on OpenThoughts3 | Complete |
| Distillation On-Policy | KL minimization to teacher | Complete |
| Math RL (GSM8K) | Word problem solving | Complete |

**Key Results:**
- Arithmetic: 69.5% → 100% accuracy in ~20 steps
- Preference learning effectively shapes response style
- Distillation transfers knowledge efficiently

### 2. Atropos Integration

Integration layer connecting [Atropos](https://github.com/NousResearch/atropos) with the Tinker API.

Features:
- Use any Atropos environment with Tinker training
- Built-in GSM8K and Math environments
- LoRA-based fine-tuning with configurable parameters
- Checkpoint management and weight downloading

### 3. RL Gym

Google Drive and browser automation tasks for RL agent training.

Features:
- 200+ tasks for browser automation
- Google Drive API integration
- MCP tool integration
- Task generation and verification

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv tinker-env
source tinker-env/bin/activate

# Install dependencies
pip install tinker tinker-cookbook atropos
```

### Running Tinker Experiments

```bash
export TINKER_API_KEY="your-key-here"

# Math RL
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="meta-llama/Llama-3.2-1B" \
    env=arithmetic

# Chat SFT
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="meta-llama/Llama-3.2-1B"
```

### Running Atropos + Tinker

```bash
# Terminal 1: Start Atropos API
run-api

# Terminal 2: Start training
export TINKER_API_KEY="your-key"
python atropos/launch_training.py --config atropos/configs/default.yaml

# Terminal 3: Start environment
python atropos/tinker_atropos/environments/gsm8k_tinker.py serve \
    --config atropos/configs/default.yaml
```

## Source Repositories

This repository consolidates all PES LLM Research projects:

| Original Repo | Description | Created |
|--------------|-------------|---------|
| [tinker-experiments](https://github.com/arvindcr4/tinker-experiments) | Tinker RL Cookbook experiments | Jan 2026 |
| [tinker-atropos](https://github.com/arvindcr4/tinker-atropos) | Atropos + Tinker integration | Mar 2026 |
| [rl](https://github.com/arvindcr4/rl) | RL Gym tasks and documentation | Aug 2025 |
| [rl_master](https://github.com/arvindcr4/rl_master) | Task execution and MCP tools | Nov 2025 |
| [agentic-rl-finetuning](https://github.com/pes-llm-research/agentic-rl-finetuning) | Agentic RL fine-tuning | Mar 2026 |
| [capstone-literature-survey](https://github.com/arvindcr4/capstone-literature-survey) | GRPO Scaling Literature Survey | Mar 2026 |

## Documentation

- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai)
- [Atropos GitHub](https://github.com/NousResearch/atropos)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)

## References

- [DeepCoder Blog Post](https://thinkingmachines.ai/blog/deepcoder)
- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation)

## Authors

PES LLM Research Team

## License

See individual component directories for license information.

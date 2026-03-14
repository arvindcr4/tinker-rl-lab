# Tinker RL Cookbook Experiments

PES LLM Research Project - Reinforcement Learning for Language Models

## Overview

This repository contains experiments with the [Tinker](https://thinkingmachines.ai/tinker) platform for training LLMs using reinforcement learning, supervised fine-tuning, and knowledge distillation.

## Experiments

| Recipe | Description | Status |
|--------|-------------|--------|
| Math RL (Arithmetic) | Train model to add numbers | ✅ Complete - 100% accuracy |
| Chat SL | Supervised fine-tuning on NoRobots | ✅ Complete |
| Preference Shorter | Train for concise responses | ✅ Complete |
| Distillation Off-Policy | SFT on OpenThoughts3 | ✅ Complete |
| Distillation On-Policy | KL minimization to teacher | ✅ Complete |
| Math RL (GSM8K) | Word problem solving | ✅ Complete |

## Repository Structure

```
tinker-experiments/
├── notebooks/                    # Jupyter notebooks for each experiment
│   ├── 01_math_rl_arithmetic.ipynb
│   ├── 02_chat_sl_sft.ipynb
│   ├── 03_preference_shorter.ipynb
│   ├── 04_distillation_off_policy.ipynb
│   ├── 05_distillation_on_policy.ipynb
│   └── 06_math_rl_gsm8k.ipynb
├── results/                      # Training metrics (JSONL)
│   ├── arithmetic_metrics.jsonl
│   ├── distillation_off_metrics.jsonl
│   └── distillation_on_metrics.jsonl
├── logs/                         # Training output logs
├── create_presentation.py        # PowerPoint generator
├── Tinker_RL_Demo.pptx          # Presentation for demo
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv tinker-env
source tinker-env/bin/activate
```

2. Install dependencies:
```bash
pip install tinker tinker-cookbook
```

3. Set your API key:
```bash
export TINKER_API_KEY="your-key-here"
```

## Running Experiments

### Math RL (Arithmetic)
```bash
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="meta-llama/Llama-3.2-1B" \
    env=arithmetic \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-4
```

### Chat Supervised Learning
```bash
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="meta-llama/Llama-3.2-1B" \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=32
```

### Preference Learning (Shorter)
```bash
python -m tinker_cookbook.recipes.preference.shorter.train \
    model_name="Qwen/Qwen3-0.6B-Instruct" \
    group_size=4 \
    groups_per_batch=50
```

## Key Results

### Math RL (Arithmetic)
- Starting accuracy: 69.5%
- Final accuracy: 100%
- Reward: 0.676 → 1.0
- Steps to convergence: ~20

### Learning Curves

The experiments demonstrate that:
1. Simple tasks (arithmetic) converge quickly with RL
2. Preference learning effectively shapes response style
3. Distillation transfers knowledge efficiently

## Notebooks

Interactive Jupyter notebooks are provided for each experiment. They include:
- Configuration explanations
- Training commands
- Metric interpretation
- Result analysis

## Presentation

`Tinker_RL_Demo.pptx` contains slides summarizing:
- Tinker platform overview
- Training methods explored
- Experiment results
- Key findings and future work

## References

- [Tinker Documentation](https://thinkingmachines.ai/tinker)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)
- [DeepCoder Blog Post](https://thinkingmachines.ai/blog/deepcoder)
- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation)

## Authors

PES LLM Research Team

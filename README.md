# Tinker RL Lab

> **Academic reviewers / professors:** start at [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) for the semester boundary and ownership, then review [`sem 3 work/`](sem%203%20work/) (Group 6 capstone) and [`sem 4 work/`](sem%204%20work/) (solo continuation). Everything else in this repository is shared research infrastructure and evidence.

A consolidated research repository for Reinforcement Learning experiments with Large Language Models, integrating multiple RL frameworks and compute backends.

## Overview

This repository consolidates multiple research projects, spanning several RL frameworks (Tinker/SkyRL, verl, OpenRLHF, TRL from HuggingFace), multiple compute backends (Local GPU, vast.ai, Google Colab), and multiple environments (Atropos, GSM8K, Math, HumanEval, Tool Use).

## Semester and authorship map

This codebase spans two academic phases. They share infrastructure, but their deliverables and authorship are separated:

- [`sem 3 work/`](sem%203%20work/) — frozen six-student Group 6 report, capstone artifacts, and the NeurIPS main-track submission.
- [`sem 4 work/`](sem%204%20work/) — Arvind C R's solo continuation under project guide Ramesh Prakash Guledgudd, including the NeurIPS workshop submission, eight freshly compiled papers, and an experiment/evidence map.
- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — exact boundary, ownership explanation, and rationale for retaining one reproducible codebase.

## Repository Structure

```
tinker-rl-lab/
├── PROJECT_HISTORY.md    # Semester boundary & academic ownership
├── sem 3 work/           # Frozen Group 6 deliverables (Semester 3)
├── sem 4 work/           # Solo-continuation papers & submission (Semester 4)
│
├── platform_tinker/      # Cloud-based Tinker platform integration
│   ├── atropos/          # Tinker-Atropos environment integration
│   ├── tinkerrl/         # Tinker API bindings
│   └── reports/          # Final capstone report and paper
│
├── platform_modal/       # Modal serverless execution
│   ├── scripts/          # Research scripts and analysis
│   └── openrlhf/         # OpenRLHF on Modal
│
├── platform_local/       # Local GPU & unified execution
│   ├── unified/          # Unified launcher for all frameworks
│   ├── trl_integrations/ # HuggingFace TRL local integration
│   └── contexts/         # Local workspace context
│
├── platform_hybrid/      # Hybrid-cloud execution logic
│   ├── experiments/      # Tinker RL Cookbook & evaluation logs
│   ├── skyrl/            # SkyRL tx integration
│   └── registry/         # Hybrid asset registry
│
├── platform_colab/       # Google Colab notebooks
├── platform_vast/        # vast.ai run scripts
├── platform_gcp/         # Google Cloud execution config
├── platform_hf_spaces/   # HuggingFace Spaces deployments
│
├── agentic-rl-finetuning/ # Agentic RL fine-tuning research
└── capstone-literature-survey/ # Literature Survey
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

Key results from the experiments:
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

### 3. SkyRL (Local Tinker API)

SkyRL tx implements the Tinker API locally on your own GPUs. Any tinker-cookbook recipe works without cloud API.

Features:
- Local Tinker API server implementation
- GRPO, PPO, REINFORCE algorithms
- vLLM/SGLang inference
- vast.ai and Colab backends

### 4. verl (Volcano Engine RL)

Production-ready RL training framework with HybridFlow programming model.

Features:
- Multi-GPU and multi-node via Ray
- PPO, GRPO, REINFORCE
- High-throughput vLLM inference

### 5. OpenRLHF

Scalable agentic RL framework with Ray + vLLM distributed architecture.

Features:
- PPO, DAPO, REINFORCE++
- Async RL training
- Multi-GPU and multi-node support

### 6. TRL (HuggingFace)

HuggingFace's full-stack RL library with easy model integration.

Features:
- GRPO, PPO, DPO, Reward Modeling
- Works with any HF model
- Single GPU to multi-GPU (DeepSpeed)

### 7. AI Scientist Integrations

Integration with autonomous research agents (e.g., AI Scientist) providing runnable templates and tools for end-to-end RL exploration.

Features:
- `tinker_grpo_rl.py`: Remote GRPO experiments via Tinker API.
- `trl_local_grpo.py`: Local GRPO execution without cloud API constraints.
- Tool-use dense reward design templates.

### 8. Evaluation, Telemetry & Capabilities

Advanced benchmarking, evaluation, and training features ensuring statistical rigor.

Features:
- **Held-out Evaluator**: Automated robust validation on held-out test sets.
- **Variance Mitigation**: 9 implemented baseline methods including AReaL, Evolution Strategies (ES), MC-GRPO, and GIFT.
- **Telemetry & ZVF Diagnostics**: Zero-Variance Fraction (ZVF) telemetry and partial correlation tracking.
- **BFCLv4 Scaffold**: Integration with Berkeley Function Calling Leaderboard v4 for comprehensive tool-use evaluation.
- **Hyperparameter Sweeper**: Automated sweep utilities for configuration exploration.
- **Full Fine-Tuning (FT)**: Support for full weight updates alongside existing LoRA tracks.

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv tinker-env
source tinker-env/bin/activate

# Install dependencies
pip install tinker tinker-cookbook atropos
# Full dependency set (all integrations, audits, and analysis scripts):
pip install -r requirements.txt
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
python platform_tinker/atropos/launch_training.py --config platform_tinker/atropos/configs/default.yaml

# Terminal 3: Start environment
python platform_tinker/atropos/tinker_atropos/environments/gsm8k_tinker.py serve \
    --config platform_tinker/atropos/configs/default.yaml
```

### Running SkyRL (Local Tinker API)

```bash
# Start local Tinker API server (requires an external SkyRL checkout).
# Upstream layout changes over time — follow the official quickstart if the
# path below has moved: https://docs.skyrl.ai
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL/skyrl-train
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-1.5B-Instruct --port 8000

# In another terminal - run any tinker-cookbook recipe
export TINKER_API_KEY="tml-dummy"
export TINKER_BASE_URL="http://localhost:8000"
python -m tinker_cookbook.recipes.math_rl.train base_url=$TINKER_BASE_URL ...
```

### Running on vast.ai

```bash
# SkyRL on vast.ai
cd platform_vast/
./vast_run.sh --model Qwen/Qwen2.5-1.5B-Instruct
```

### Running with Unified Launcher

The unified launcher is a smoke-test scaffold: it validates framework dispatch and configuration and emits simulated metrics. Use the per-framework sections above for real training.

```bash
# Use any framework with unified launcher
export PYTHONPATH=.
python -m platform_local.unified.launcher --framework skyrl --model Qwen/Qwen2.5-1.5B-Instruct
python -m platform_local.unified.launcher --framework trl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo
python -m platform_local.unified.launcher --framework verl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm ppo
python -m platform_local.unified.launcher --framework openrlhf --model Qwen/Qwen2.5-1.5B-Instruct
```

### Running in Google Colab

Open the notebooks inside `platform_colab/` and run cells sequentially.

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

### Semester 4 — individual continuation

- Arvind C R (PES University) — student researcher and author
- Ramesh Prakash Guledgudd (PES University) — project guide

The current P1–P8 paper series belongs to this phase. See [`sem 4 work/`](sem%204%20work/) and the shared author source in [`paper/sections/_shared_author.tex`](paper/sections/_shared_author.tex).

### Semester 3 — Group 6

- Arvind C R (PES University) &mdash; equal contribution
- Sandhya Jeyaraj (PES University) &mdash; equal contribution
- Madhu Kumara L (PES University)
- Mohammad Rafi (PES University)
- Dhruva N Murthy (PES University)
- Arumugam Chetty K (PES University)
- Anwesh Reddy Paduri (Great Learning / PES University) &mdash; project guide
- Narayana Darapaneni (Northwestern University / Great Learning) &mdash; project guide

Corresponding author: Arvind C R &lt;arvindcr4@gmail.com&gt;. Equal contribution denotes equal
technical and writing contribution; author order among the student team is alphabetical by given name after the two equal-contribution leads.
The root [`CITATION.cff`](CITATION.cff) is the canonical citation record for the Semester 3 group release; a frozen copy is retained in [`sem 3 work/CITATION.cff`](sem%203%20work/CITATION.cff).

## License

See individual component directories for license information.

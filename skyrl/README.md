# SkyRL Integration for tinker-rl-lab

This module integrates **SkyRL** (NovaSky-AI/SkyRL) with tinker-rl-lab, providing:
- **SkyRL tx**: Local Tinker API server implementation for training on your own hardware
- **vast.ai backend**: Run SkyRL tx on vast.ai GPU instances
- **Colab notebooks**: Run SkyRL tx in Google Colab

## Quick Start

### 1. Clone and Setup SkyRL

```bash
# Clone SkyRL
git clone --depth 1 --branch skyrl_train-v0.4.0 https://github.com/NovaSky-AI/SkyRL.git SkyRL
cd SkyRL/skyrl-train

# Create virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra vllm
pip install wandb datasets math-verify latex2sympy2-extended
```

### 2. Start SkyRL tx (Local Tinker API Server)

```bash
# Start the Tinker API server on your local GPUs
cd SkyRL
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000
```

### 3. Run Training with Tinker Cookbook Scripts

Now any tinker-cookbook script can connect to your local server:

```bash
export TINKER_API_KEY="tml-dummy"  # Dummy key for local server
export TINKER_BASE_URL="http://localhost:8000"

# Run GRPO training on GSM8K
python -m tinker_cookbook.recipes.math_rl.train \
    base_url=$TINKER_BASE_URL \
    model_name=Qwen/Qwen2.5-1.5B-Instruct \
    lora_rank=32
```

## Backend Options

### Local (Default)

Run on your own GPUs using SkyRL tx:

```bash
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model <model> --port 8000
```

### vast.ai

Provision GPU instances and run SkyRL tx remotely:

```bash
# Using the vastai runner
cd skyrl
python -m skyrl.backends.vastai_runner \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --algorithm grpo \
    --epochs 20
```

### Google Colab

Run in Colab with T4/A100 GPUs:

1. Open `skyrl/notebooks/skyrl_colab_training.ipynb`
2. Set your API keys
3. Run cells sequentially

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    tinker-rl-lab                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  atropos/    │  │  experiments/ │  │    skyrl/    │    │
│  │  Tinker-     │  │  Tinker RL   │  │  SkyRL tx     │    │
│  │  Atropos     │  │  Cookbook     │  │  Integration │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │             │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Tinker API Layer (skyrl-tx)                │
│         Unified interface for training & inference          │
│              skyrl.tinker.api (local server)                │
└─────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Compute Backends                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    local     │  │   vast.ai     │  │    Colab     │    │
│  │  GPU(s)      │  │  GPU instances│  │   T4/A100    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## SkyRL tx Key Commands

### Start Server

```bash
# Single GPU
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-1.5B-Instruct

# Multi-GPU (Tensor Parallelism)
uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --backend-config '{"tensor_parallel_size": 2}'

# With external vLLM inference
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --external-inference-url "http://localhost:7999"
```

### Training Scripts

```python
import tinker
from tinker import types

# Connect to local SkyRL tx server
client = tinker.ServiceClient(
    base_url="http://localhost:8000",
    api_key="tml-dummy"
)

# Create training client
training_client = client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    rank=32
)

# Training loop
for step in range(100):
    # Get batch data (from environment)
    data = get_batch_data()
    
    # Forward-backward
    fwd_bwd = training_client.forward_backward(data, "grpo").result()
    
    # Optimizer step
    training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    
    print(f"Step {step}, loss={fwd_bwd.metrics['loss:sum']}")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TINKER_API_KEY` | API key for authentication | `tml-dummy` (local) |
| `TINKER_BASE_URL` | Base URL of Tinker API server | `http://localhost:8000` |
| `VAST_API_KEY` | vast.ai API key | - |
| `WANDB_API_KEY` | Weights & Biases API key | - |

## Files

```
skyrl/
├── README.md                    # This file
├── backends/
│   ├── vastai_runner.py        # Provision & run on vast.ai
│   └── vastai_launch.sh        # Launch script for vast.ai
├── notebooks/
│   └── skyrl_colab_training.ipynb  # Colab notebook
└── configs/
    ├── grpo_gsm8k.yaml         # GRPO config for GSM8K
    └── grpo_math.yaml          # GRPO config for Math
```

## References

- [SkyRL Documentation](https://docs.skyrl.ai/docs)
- [SkyRL tx GitHub](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx)
- [Tinker API Docs](https://tinker-docs.thinkingmachines.ai)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)

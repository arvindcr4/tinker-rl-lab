# =============================================================================
# TinkerRL Lab - Reproducible Environment
# =============================================================================
# Build:  docker build -t tinker-rl-lab .
# Run:    docker run --gpus all -it tinker-rl-lab bash
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create workspace
WORKDIR /workspace/tinker-rl-lab

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install the project in editable mode
RUN pip install -e atropos/ 2>/dev/null || true

# Default seed for reproducibility
ENV SEED=42
ENV WANDB_MODE=offline

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'Transformers {transformers.__version__}')" && \
    python -c "import trl; print(f'TRL {trl.__version__}')"

CMD ["bash"]

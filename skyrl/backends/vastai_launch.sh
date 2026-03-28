#!/bin/bash
# vastai_launch.sh - Launch SkyRL tx on vast.ai
# Usage: ./vastai_launch.sh --model Qwen/Qwen2.5-1.5B-Instruct

set -euo pipefail

# Default values
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ALGORITHM="${ALGORITHM:-grpo}"
EPOCHS="${EPOCHS:-20}"
INSTANCE_TYPE="${INSTANCE_TYPE:-a100-80gb}"
PORT="${PORT:-8000}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"; shift 2 ;;
        --algorithm)
            ALGORITHM="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --instance-type)
            INSTANCE_TYPE="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

echo "=== SkyRL vast.ai Launcher ==="
echo "Model: $MODEL"
echo "Algorithm: $ALGORITHM"
echo "Epochs: $EPOCHS"
echo "Instance Type: $INSTANCE_TYPE"
echo "Port: $PORT"
echo ""

# Check for vast.ai CLI
if ! command -v vast &> /dev/null; then
    echo "Installing vast.ai CLI..."
    pip install vast
fi

# Check for API key
if [[ -z "${VAST_API_KEY:-}" ]]; then
    echo "ERROR: VAST_API_KEY environment variable not set"
    echo "Set it with: export VAST_API_KEY='your-key'"
    exit 1
fi

# Search for instances
echo "Searching for $INSTANCE_TYPE instances..."
AVAILABLE=$(vast search instances "$INSTANCE_TYPE" --json 2>/dev/null | head -20)

if [[ -z "$AVAILABLE" ]]; then
    echo "No instances found. Trying broader search..."
    AVAILABLE=$(vast search instances a100 --json 2>/dev/null | head -20)
fi

if [[ -z "$AVAILABLE" ]]; then
    echo "ERROR: No instances available"
    exit 1
fi

# Get first available instance
INSTANCE_ID=$(echo "$AVAILABLE" | jq -r '.[0].id')

echo "Launching instance $INSTANCE_ID..."

# Launch instance
INSTANCE_INFO=$(vast create instance "$INSTANCE_ID" --json --api-key "$VAST_API_KEY")
echo "Instance info: $INSTANCE_INFO"

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 30

# Get SSH details
SSH_HOST=$(echo "$INSTANCE_INFO" | jq -r '.ssh_host')
SSH_PORT=$(echo "$INSTANCE_INFO" | jq -r '.ssh_port // 22')

echo "Instance ready at $SSH_HOST:$SSH_PORT"

# Generate setup script
cat > /tmp/setup_skyrl.sh << 'SETUP_EOF'
#!/bin/bash
set -euo pipefail

echo "=== Setting up SkyRL ==="

# Install system deps
apt-get update -qq
apt-get install -y -qq git curl wget build-essential

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone SkyRL
git clone --depth 1 --branch skyrl_train-v0.4.0 \
    https://github.com/NovaSky-AI/SkyRL.git /root/SkyRL

cd /root/SkyRL/skyrl-train

# Create venv and install
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra vllm --extra gpu --extra tinker

# Install additional deps
uv pip install wandb datasets math-verify latex2sympy2-extended trl peft accelerate

echo "=== Setup Complete ==="
SETUP_EOF

# Copy and run setup via SSH
echo "Running setup on remote instance..."
scp -P "$SSH_PORT" /tmp/setup_skyrl.sh root@"$SSH_HOST":/root/setup_skyrl.sh
ssh -p "$SSH_PORT" root@"$SSH_HOST" "bash /root/setup_skyrl.sh"

# Generate start script
cat > /tmp/start_skyrl.sh << START_EOF
#!/bin/bash
cd /root/SkyRL/skyrl-train
source .venv/bin/activate

# Start SkyRL tx server
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --extra tinker -m skyrl.tinker.api \
    --base-model "$MODEL" \
    --port $PORT \
    > /root/skyrl_server.log 2>&1 &

echo "SkyRL tx server starting..."
sleep 10
cat /root/skyrl_server.log
START_EOF

# Start server
echo "Starting SkyRL tx server..."
scp -P "$SSH_PORT" /tmp/start_skyrl.sh root@"$SSH_HOST":/root/start_skyrl.sh
ssh -p "$SSH_PORT" root@"$SSH_HOST" "bash /root/start_skyrl.sh &"

echo ""
echo "=== SkyRL tx Server Running ==="
echo "Connect with TINKER_BASE_URL=http://$SSH_HOST:$PORT"
echo "API Key: tml-dummy"
echo ""
echo "To run training:"
echo "  export TINKER_API_KEY='tml-dummy'"
echo "  export TINKER_BASE_URL='http://$SSH_HOST:$PORT'"
echo "  python -m tinker_cookbook.recipes.math_rl.train ..."
echo ""
echo "To stop: ssh -p $SSH_PORT root@$SSH_HOST 'pkill -f skyrl.tinker'"

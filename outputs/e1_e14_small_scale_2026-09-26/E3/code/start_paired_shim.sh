#!/bin/bash
# usage: start_paired_shim.sh <lane E3|E7|E12> <arm vllm_trained|vllm_base> <port> <budget> [max_tokens]
set -e
R=/Users/arvind/Developer/agentic_repos/tinker-rl-lab; D=$R/outputs/e1_e14_small_scale_2026-09-26
set -a; source $D/trained_actor/.env.local; set +a
if [ "$2" = vllm_trained ]; then U=https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run; M=pavlov-public-portfolio-bf16
else U=https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run; M=qwen36-base-bf16; fi
if lsof -Pan -iTCP:$3 -sTCP:LISTEN -t >/dev/null; then echo "port $3 busy"; exit 1; fi
mkdir -p $D/$1/$2/raw
SHIM_BACKEND=vllm VLLM_BASE_URL=$U VLLM_MODEL=$M SHIM_LOG=$D/$1/$2/raw/shim_calls.jsonl SHIM_BUDGET=$4 PORT=$3 \
SHIM_MAX_TOKENS=${5:-4096} nohup /Users/arvind/.local/share/uv/tools/tinker/bin/python $D/E3/code/tinker_shim.py > $D/$1/$2/raw/shim.out 2>&1 &
echo "$1 $2 port $3 pid $!"

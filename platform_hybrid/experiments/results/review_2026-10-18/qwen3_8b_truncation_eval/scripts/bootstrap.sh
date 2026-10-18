#!/bin/bash
# bootstrap on fresh VM: deps, model, Q1 adapter, then run_all
cd /content; mkdir -p /content/out /content/ad/tinker_s42 /content/ad/tinker_s123 /content/ad/tinker_s456
nvidia-smi --query-gpu=name,memory.total --format=csv > /content/out/gpu.txt
pip uninstall -y torchao > /dev/null 2>&1
(which uv || pip install -q uv) > /dev/null
uv venv /content/vv --python 3.12 && uv pip install --python /content/vv/bin/python vllm==0.32.0 datasets ninja > /content/out/vllm_install.log 2>&1
hf download Qwen/Qwen3-8B > /dev/null 2>&1
hf download arvindcr4/tinker-rl-scale_gsm8k_qwen3-8b-qwen3-8b --local-dir /content/ad/q1_scale8b > /dev/null 2>&1
echo BOOTDONE $(date -u +%T) >> /content/out/driver.log
GPU_UTIL=0.90 bash /content/run_all.sh

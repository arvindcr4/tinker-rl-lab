#!/bin/bash
# Paired arm vllm_trained: identical to run_e3.sh except shim port (vLLM backend), jobs dir, and -n 3 (endpoint concurrency cap)
cd "$(dirname "$0")/.."
export OPENAI_API_KEY=local-shim-no-key
harbor run -p code/tb2/terminal-bench -i build-cython-ext -i build-pmars -i cobol-modernization -i count-dataset-tokens -i hf-model-inference -i largest-eigenval -i pytorch-model-recovery -i query-optimize \
  -a terminus-2 -m openai/Qwen/Qwen3.6-35B-A3B \
  --ak api_base=http://127.0.0.1:8771/v1 --ak temperature=0 --ak max_turns=25 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  -e modal -n 3 -o vllm_trained/raw/jobs --job-name e3_tb2_vllm_trained

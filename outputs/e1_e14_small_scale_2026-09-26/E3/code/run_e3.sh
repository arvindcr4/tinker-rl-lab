#!/bin/bash
# Terminal-Bench 2.0 subset, terminus-2 agent -> local Tinker shim (:8763), Modal sandboxes (amd64)
cd "$(dirname "$0")/.."
export OPENAI_API_KEY=local-shim-no-key
harbor run -p code/tb2/terminal-bench -i build-cython-ext -i build-pmars -i cobol-modernization -i count-dataset-tokens -i hf-model-inference -i largest-eigenval -i pytorch-model-recovery -i query-optimize \
  -a terminus-2 -m openai/Qwen/Qwen3.6-35B-A3B \
  --ak api_base=http://127.0.0.1:8763/v1 --ak temperature=0 --ak max_turns=25 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  -e modal -n 4 -o raw/jobs --job-name e3_tb2_subset

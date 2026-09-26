#!/bin/bash
# BinaryAudit (public, upstream cbd86c7) 6-task subset; terminus-2 -> local Tinker shim (:8767); Modal sandboxes (amd64)
cd "$(dirname "$0")/.."
export OPENAI_API_KEY=local-shim-no-key
python3 code/make_modal_tasks.py
harbor run -p code/tasks_modal \
  -a terminus-2 -m openai/Qwen/Qwen3.6-35B-A3B \
  --ak api_base=http://127.0.0.1:8767/v1 --ak temperature=0 --ak max_turns=25 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  -e modal -n 3 -o raw/jobs --job-name e7_binaryaudit_subset

#!/bin/bash
# Paired arm vllm_base: same 6 tasks, same terminus-2 settings as run_e7.sh/run_e7_envfix2.sh; all tasks built from
# tasks_modal_paired (base stage inlined + debian:11 snapshot fix). Only the shim port (vLLM backend) differs.
cd "$(dirname "$0")/.."
export OPENAI_API_KEY=local-shim-no-key
[ -d code/tasks_modal_paired ] || TASKS_OUT=tasks_modal_paired python3 code/make_modal_tasks.py
harbor run -p code/tasks_modal_paired \
  -a terminus-2 -m openai/Qwen/Qwen3.6-35B-A3B \
  --ak api_base=http://127.0.0.1:18934/v1 --ak temperature=0 --ak max_turns=25 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  -e modal -n 3 -o vllm_base/raw/jobs --job-name e7_vllm_base

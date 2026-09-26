#!/bin/bash
# Second rerun (shim process was killed externally mid-run at ~14:35Z) of the 3 dnsmasq tasks whose Modal image build failed before any model call (debian:11 security-pool 404s).
# Same agent/model/caps as run_e7.sh; only the builder-stage apt sources differ (see make_modal_tasks.py).
cd "$(dirname "$0")/.."
export OPENAI_API_KEY=local-shim-no-key
TASKS_OUT=tasks_modal_envfix python3 code/make_modal_tasks.py dnsmasq-backdoor-detect dnsmasq-backdoor-detect-execvp-obfuscated dnsmasq-backdoor-detect-printf
harbor run -p code/tasks_modal_envfix \
  -a terminus-2 -m openai/Qwen/Qwen3.6-35B-A3B \
  --ak api_base=http://127.0.0.1:8767/v1 --ak temperature=0 --ak max_turns=25 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  -e modal -n 3 -o raw/jobs --job-name e7_envfix_rerun2

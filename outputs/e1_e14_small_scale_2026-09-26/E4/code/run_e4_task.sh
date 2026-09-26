#!/bin/bash
# usage: run_e4_task.sh <task_name>
S=/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad
D=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E4
export DOCKER_HOST=unix:///Users/arvind/.colima/default/docker.sock
export OPENAI_API_KEY="$SHIM_KEY"
export OPENAI_BASE_URL=http://host.docker.internal:18741/v1
export GEMINI_API_KEY="$(security find-generic-password -s GEMINI_API_KEY -w)"
cd /Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e4_banker_toolbench/official_repo_ff6db552
sed "s/task_names: \[\]/task_names: [$1]/; s/job_name: e4-small-base/job_name: e4-small-$1/" $D/code/job-e4-small.yaml > $D/code/job-$1.yaml
harbor run -c $D/code/job-$1.yaml -o $D/raw/jobs -n 1 -y 2>&1

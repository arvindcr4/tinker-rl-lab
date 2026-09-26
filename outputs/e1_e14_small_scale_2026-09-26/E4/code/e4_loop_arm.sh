#!/bin/bash
# usage: e4_loop_arm.sh <arm: vllm_trained|vllm_base> <shim_port> <task ids...>
# Identical to the Tinker arm (opencode + native verifier, 500k tokens/task cap) except the actor endpoint,
# and context 32768/output 8192 (vLLM max-model-len 32768), same in both vLLM arms.
S=${SCRATCH:-/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad}
D=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E4
arm=$1; port=$2; shift 2
export DOCKER_HOST=unix:///Users/arvind/.colima/default/docker.sock
export OPENAI_API_KEY=$(command cat $S/shimkey)
export OPENAI_BASE_URL=http://host.docker.internal:$port/v1
export GEMINI_API_KEY="$(security find-generic-password -s GEMINI_API_KEY -w)"
for t in "$@"; do
  L=$D/$arm/raw/shim_ledger.jsonl; touch $L
  USED=$(python3 -c "import json;print(sum(json.loads(l).get('prompt_tokens',0)+json.loads(l).get('completion_tokens',0) for l in open('$L')))")
  echo "{\"cap_total_tokens\": $((USED+500000))}" > $D/$arm/raw/cap.json
  echo "[loop] $t start $(date -u +%FT%TZ) used=$USED" >> $D/$arm/raw/loop.log
  sed "s/task_names: \[\]/task_names: [$t]/; s/job_name: e4-vllm/job_name: e4-$arm-$t/" $D/code/job-e4-vllm.yaml > $D/$arm/raw/job-$t.yaml
  (cd /Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e4_banker_toolbench/official_repo_ff6db552 && harbor run -c $D/$arm/raw/job-$t.yaml -o $D/$arm/raw/jobs -n 1 -y > $S/e4_${arm}_$t.log 2>&1)
  echo "[loop] $t end $(date -u +%FT%TZ)" >> $D/$arm/raw/loop.log
  docker images --format '{{.Repository}}' | grep '^btb-' | xargs -r docker rmi -f >/dev/null 2>&1; docker ps -aq --filter name=btb- | xargs -r docker rm -f >/dev/null 2>&1; docker builder prune -f >/dev/null 2>&1; timeout 120 colima ssh -- sudo fstrim -a >/dev/null 2>&1
done

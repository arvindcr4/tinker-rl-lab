#!/bin/bash
S=/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad
D=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E4
for t in "$@"; do
  USED=$(python3 -c "import json;print(sum(json.loads(l).get('prompt_tokens',0)+json.loads(l).get('completion_tokens',0) for l in open('$D/raw/shim_ledger.jsonl')))")
  echo "{\"cap_total_tokens\": $((USED+500000))}" > $D/raw/cap.json
  echo "[e4loop] $t start $(date -u +%FT%TZ) used=$USED" >> $D/raw/e4_loop.log
  $S/run_e4_task.sh $t > $S/e4_$t.log 2>&1
  echo "[e4loop] $t end $(date -u +%FT%TZ) rc=$?" >> $D/raw/e4_loop.log
  DOCKER_HOST=unix:///Users/arvind/.colima/default/docker.sock docker images --format '{{.Repository}}' | grep '^btb-' | xargs -r env DOCKER_HOST=unix:///Users/arvind/.colima/default/docker.sock docker rmi -f >/dev/null 2>&1; timeout 120 colima ssh -- sudo fstrim -a >/dev/null 2>&1
  [ -f $S/stop_e4 ] && break
done

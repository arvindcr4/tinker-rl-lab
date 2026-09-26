#!/bin/bash
# usage: shim_loop.sh <lane E4|E5> <port>
S=/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad
D=/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/$1
set -a; source /Users/arvind/Developer/agentic_repos/tinker-rl-lab/.env; set +a
export SHIM_KEY=$(command cat $S/shimkey)
while true; do
  echo "[loop] start $(date -u +%FT%TZ)" >> $S/shim_$1.log
  $S/shimvenv/bin/python /Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26/E4/code/tinker_shim.py --port $2 --ledger $D/raw/shim_ledger.jsonl --cap-file $D/raw/cap.json >> $S/shim_$1.log 2>&1
  echo "[loop] exit $? $(date -u +%FT%TZ)" >> $S/shim_$1.log
  [ -f $S/stop_shim_$1 ] && exit 0
  sleep 3
done

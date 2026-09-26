#!/bin/bash
# usage: shim_run.sh <name> <port> <ledger> <capfile> [extra shim args...]   (restarts on crash; touch $SCRATCH/stop_<name> to end)
S=${SCRATCH:-/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/f6a17e89-20dd-4b9b-b647-18831abd53b1/scratchpad}
R=/Users/arvind/Developer/agentic_repos/tinker-rl-lab
set -a; source $R/.env; source $R/outputs/e1_e14_small_scale_2026-09-26/trained_actor/.env.local; set +a
export SHIM_KEY=$(command cat $S/shimkey)
name=$1; port=$2; ledger=$3; cap=$4; shift 4
rm -f $S/stop_$name
while true; do
  echo "[loop] start $(date -u +%FT%TZ)" >> $S/shim_$name.log
  $S/shimvenv/bin/python $R/outputs/e1_e14_small_scale_2026-09-26/E4/code/tinker_shim.py --port $port --ledger $ledger --cap-file $cap "$@" >> $S/shim_$name.log 2>&1
  echo "[loop] exit $? $(date -u +%FT%TZ)" >> $S/shim_$name.log
  [ -f $S/stop_$name ] && exit 0
  sleep 3
done

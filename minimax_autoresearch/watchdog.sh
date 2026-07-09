#!/usr/bin/env bash
# L0 resident guardian (deli §7). Restarts the orchestrator if its heartbeat goes
# stale while inside the time budget. Run alongside the main loop:
#   nohup ./watchdog.sh 35 > logs/watchdog.$(date +%s).out 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_HOURS="${1:-35}"
STALE_SECS="${STALE_SECS:-1800}"     # 30 min without a heartbeat = dead
CHECK_SECS="${CHECK_SECS:-300}"
STATE_DIR="${STATE_DIR:-state}"      # e.g. state_berkeley for the Berkeley run
RUN_SH="${RUN_SH:-$HERE/run.sh}"     # launcher used for restarts
HB="$HERE/$STATE_DIR/heartbeat.json"
START=$(date +%s)
DEADLINE=$(( START + $(printf '%.0f' "$(echo "$MAX_HOURS*3600" | bc -l)") ))

log(){ echo "{\"ts\":\"$(date -Is)\",\"source\":\"watchdog\",\"level\":\"$1\",\"event\":\"$2\"}" \
        | tee -a "$HERE/logs/heartbeat.jsonl"; }

log info started
while :; do
  NOW=$(date +%s)
  [[ -f "$HERE/$STATE_DIR/STOP" ]] && { log info stop_seen; break; }
  (( NOW >= DEADLINE )) && { log info budget_reached; break; }
  # is orchestrator alive?
  if ! pgrep -f "orchestrator.py" >/dev/null 2>&1; then
    if [[ ! -f "$HERE/$STATE_DIR/STOP" ]] && (( NOW < DEADLINE )); then
      REMAIN_H=$(echo "scale=2; ($DEADLINE-$NOW)/3600" | bc -l)
      log warn restarting_dead_loop
      nohup "$RUN_SH" "$REMAIN_H" all > "$HERE/logs/run.restart.$NOW.out" 2>&1 &
    fi
  else
    # alive but heartbeat stale?
    if [[ -f "$HB" ]]; then
      HB_AGE=$(( NOW - $(stat -c %Y "$HB") ))
      if (( HB_AGE > STALE_SECS )); then
        log warn heartbeat_stale_killing
        pkill -f "orchestrator.py" 2>/dev/null || true   # watchdog restarts next tick
      fi
    fi
  fi
  sleep "$CHECK_SECS"
done
log info exited

#!/usr/bin/env bash
# Detached restart of the orchestrator+watchdog at 8h so new orchestrator.py code loads.
# Run as: setsid bash restart_8h.sh >/dev/null 2>&1 &  (pkill here won't hit the caller's shell)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
pkill -f "watchdog.sh" 2>/dev/null
for p in $(pgrep -f "orchestrator.py" 2>/dev/null); do
  grep -q "venv/bin/python" "/proc/$p/cmdline" 2>/dev/null && kill "$p" 2>/dev/null
done
sleep 3
rm -f state/STOP
TS=$(date +%s)
nohup ./run.sh 8 all > "logs/run.$TS.out" 2>&1 &
sleep 8
nohup ./watchdog.sh 8 > "logs/watchdog.$TS.out" 2>&1 &
sleep 2
echo "restarted ts=$TS orch=$(pgrep -f orchestrator.py | wc -l) wd=$(pgrep -f watchdog.sh | wc -l)" > logs/restart.marker

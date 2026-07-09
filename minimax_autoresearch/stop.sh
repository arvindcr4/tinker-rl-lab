#!/usr/bin/env bash
# Graceful stop: the orchestrator finishes the current iteration, then exits.
# Add --now to also kill immediately.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
touch "$HERE/state/STOP"
echo "STOP requested (orchestrator will exit after the current iteration)."
if [[ "${1:-}" == "--now" ]]; then
  pkill -f "orchestrator.py" 2>/dev/null && echo "killed orchestrator" || echo "no orchestrator process"
  pkill -f "watchdog.sh" 2>/dev/null && echo "killed watchdog" || true
fi
echo "To resume later: rm $HERE/state/STOP && ./run.sh <hours> all"

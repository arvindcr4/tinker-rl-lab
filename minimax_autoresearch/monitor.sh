#!/usr/bin/env bash
# Snapshot of run health. Usage: ./monitor.sh [N_last_iters]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-8}"
echo "=== process ==="; pgrep -af "orchestrator.py" || echo "(orchestrator not running)"
echo; echo "=== progress.json ==="; cat "$HERE/state/progress.json" 2>/dev/null || echo "(none)"
echo; echo "=== heartbeat age ==="
if [[ -f "$HERE/state/heartbeat.json" ]]; then
  echo "$(( $(date +%s) - $(stat -c %Y "$HERE/state/heartbeat.json") ))s ago"
else echo "(no heartbeat)"; fi
echo; echo "=== last $N iterations ==="
tail -n "$N" "$HERE/state/iteration_log.jsonl" 2>/dev/null || echo "(none yet)"
echo; echo "=== findings count ==="
wc -l /home/claude/tinker-rl-lab-minimax/AUTORESEARCH_FINDINGS.jsonl 2>/dev/null || echo "0"
echo; echo "=== worktree changes ==="
git -C /home/claude/tinker-rl-lab-minimax status --porcelain 2>/dev/null | wc -l

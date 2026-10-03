#!/usr/bin/env bash
#
# run-tmux.sh — launch ralph.sh in a detached tmux session (survives
# terminal drops and network disconnects).
#
# Usage: tools/harness/run-tmux.sh <session-name> [ralph args...]
#   e.g. tools/harness/run-tmux.sh grind --max-iters 20
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: run-tmux.sh <session-name> [ralph args...]" >&2; exit 2; }
NAME="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
command -v tmux >/dev/null || { echo "error: tmux not found" >&2; exit 1; }
tmux has-session -t "$NAME" 2>/dev/null && { echo "session $NAME already exists (tmux attach -t $NAME)" >&2; exit 1; }
tmux new-session -d -s "$NAME" -c "$ROOT" "tools/harness/ralph.sh $*; echo '[exit $?] press enter'; read -r"
echo "started tmux session: $NAME"
echo "  attach:  tmux attach -t $NAME"
echo "  logs:    tail -f tools/harness/logs/*.jsonl"
echo "  stop:    touch tools/harness/STOP   (or tmux kill-session -t $NAME)"

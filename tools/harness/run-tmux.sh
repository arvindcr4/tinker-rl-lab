#!/usr/bin/env bash
#
# run-tmux.sh — launch ralph.sh in a detached tmux session (survives
# terminal drops and network disconnects).
#
# Usage: tools/harness/run-tmux.sh <session-name> [ralph args...]
#   e.g. tools/harness/run-tmux.sh grind --max-iters 20
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: run-tmux.sh <session-name> [ralph args...]" >&2; exit 2; }
NAME="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
command -v tmux >/dev/null || { echo "error: tmux not found" >&2; exit 1; }
tmux has-session -t "$NAME" 2>/dev/null && { echo "session $NAME already exists (tmux attach -t $NAME)" >&2; exit 1; }
printf -v COMMAND '%q ' "$ROOT/tools/harness/ralph.sh" "$@"
# Status is evaluated by the child, after ralph finishes, never while composing.
COMMAND+='; rc=$?; printf "[exit %s] press enter\n" "$rc"; read -r; exit "$rc"'
printf -v WRAPPED 'bash -c %q' "$COMMAND"
tmux new-session -d -s "$NAME" -c "$ROOT" "$WRAPPED"

echo "started tmux session: $NAME"
echo "  attach:  tmux attach -t $NAME"
echo "  logs:    tail -f <worktree-git-dir>/ralph-run/logs/*.jsonl"
echo "  stop:    touch <worktree-git-dir>/ralph-run/STOP   (or tmux kill-session -t $NAME)"

#!/usr/bin/env bash
# Launch the MiniMax-M3 autoresearch orchestrator.
#   ./run.sh [MAX_HOURS] [PILLARS]
#   MAX_HOURS : wall-clock budget (default 35)
#   PILLARS   : "all" or e.g. "1,2" (default all)
# For an unattended 35h run, background it:
#   nohup ./run.sh 35 all > logs/run.$(date +%s).out 2>&1 &
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORKTREE="${WORKTREE:-/home/claude/tinker-rl-lab-minimax}"
MAX_HOURS="${1:-35}"
PILLARS="${2:-all}"
ROUND_CAP="${ROUND_CAP:-60}"

# --- load MiniMax credentials/env (never printed) ---
if [[ ! -f "$REPO/.env.minimax" ]]; then
  echo "FATAL: $REPO/.env.minimax not found" >&2; exit 1
fi
set -a; source "$REPO/.env.minimax"; set +a
: "${ANTHROPIC_AUTH_TOKEN:?MiniMax token not loaded}"

# --- sanity: worktree present ---
if [[ ! -d "$WORKTREE/.git" && ! -f "$WORKTREE/.git" ]]; then
  echo "FATAL: worktree $WORKTREE missing (git worktree add ...)" >&2; exit 1
fi

mkdir -p "$HERE/state" "$HERE/logs"
echo "[run.sh] model=$ANTHROPIC_MODEL endpoint=$ANTHROPIC_BASE_URL"
echo "[run.sh] worktree=$WORKTREE max_hours=$MAX_HOURS pillars=$PILLARS round_cap=$ROUND_CAP"

exec "$HERE/.venv/bin/python" "$HERE/orchestrator.py" \
  --worktree "$WORKTREE" \
  --state-dir "$HERE/state" \
  --task-spec "$HERE/state/task_spec.md" \
  --max-hours "$MAX_HOURS" \
  --pillars "$PILLARS" \
  --round-cap "$ROUND_CAP"

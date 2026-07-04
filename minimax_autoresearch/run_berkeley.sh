#!/usr/bin/env bash
# Launch the Berkeley-curriculum improvement-mining run (separate state dir so the
# 4-pillar paper run's state is untouched).
#   ./run_berkeley.sh [MAX_HOURS] [PILLARS]     (PILLARS: "all" or e.g. "1,3")
# Unattended:
#   nohup ./run_berkeley.sh 8 all > logs/run_berkeley.$(date +%s).out 2>&1 &
#   STATE_DIR=state_berkeley RUN_SH=./run_berkeley.sh \
#     nohup ./watchdog.sh 8 > logs/watchdog_berkeley.$(date +%s).out 2>&1 &
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORKTREE="${WORKTREE:-/home/claude/tinker-rl-lab-minimax}"
MAX_HOURS="${1:-8}"
PILLARS="${2:-all}"
ROUND_CAP="${ROUND_CAP:-50}"

if [[ ! -f "$REPO/.env.minimax" ]]; then
  echo "FATAL: $REPO/.env.minimax not found" >&2; exit 1
fi
set -a; source "$REPO/.env.minimax"; set +a
: "${ANTHROPIC_AUTH_TOKEN:?MiniMax token not loaded}"

if [[ ! -d "$WORKTREE/.git" && ! -f "$WORKTREE/.git" ]]; then
  echo "FATAL: worktree $WORKTREE missing (git worktree add ...)" >&2; exit 1
fi
if [[ ! -f "$WORKTREE/BERKELEY_IMPROVEMENT_BRIEF.md" ]]; then
  echo "FATAL: BERKELEY_IMPROVEMENT_BRIEF.md missing from worktree" >&2; exit 1
fi

mkdir -p "$HERE/state_berkeley" "$HERE/logs"
echo "[run_berkeley.sh] model=$ANTHROPIC_MODEL endpoint=$ANTHROPIC_BASE_URL"
echo "[run_berkeley.sh] worktree=$WORKTREE max_hours=$MAX_HOURS pillars=$PILLARS round_cap=$ROUND_CAP"

exec "$HERE/.venv/bin/python" "$HERE/orchestrator.py" \
  --worktree "$WORKTREE" \
  --state-dir "$HERE/state_berkeley" \
  --task-spec "$HERE/state_berkeley/task_spec_berkeley.md" \
  --pillars-file "$HERE/state_berkeley/pillars_berkeley.json" \
  --max-hours "$MAX_HOURS" \
  --pillars "$PILLARS" \
  --round-cap "$ROUND_CAP"

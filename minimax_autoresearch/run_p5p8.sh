#!/usr/bin/env bash
# Launch the P5-P8 improvement-mining run (separate state dir so other runs'
# state is untouched).
#   ./run_p5p8.sh [MAX_HOURS] [PILLARS]     (PILLARS: "all" or e.g. "1,3")
# Unattended:
#   nohup ./run_p5p8.sh 35 all > logs/run_p5p8.$(date +%s).out 2>&1 &
#   STATE_DIR=state_p5p8 RUN_SH=./run_p5p8.sh \
#     nohup ./watchdog.sh 35 > logs/watchdog_p5p8.$(date +%s).out 2>&1 &
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORKTREE="${WORKTREE:-/home/claude/tinker-rl-lab-minimax}"
MAX_HOURS="${1:-35}"
PILLARS="${2:-all}"
ROUND_CAP="${ROUND_CAP:-100}"
MAX_COST_USD="${MAX_COST_USD:-0}"   # 0 = no cost ceiling (wall-clock bound only)

if [[ ! -f "$REPO/.env.minimax" ]]; then
  echo "FATAL: $REPO/.env.minimax not found" >&2; exit 1
fi
set -a; source "$REPO/.env.minimax"; set +a
: "${ANTHROPIC_AUTH_TOKEN:?MiniMax token not loaded}"

if [[ ! -d "$WORKTREE/.git" && ! -f "$WORKTREE/.git" ]]; then
  echo "FATAL: worktree $WORKTREE missing (git worktree add ...)" >&2; exit 1
fi
if [[ ! -f "$WORKTREE/P5P8_IMPROVEMENT_BRIEF.md" ]]; then
  echo "FATAL: P5P8_IMPROVEMENT_BRIEF.md missing from worktree" >&2; exit 1
fi

mkdir -p "$HERE/state_p5p8" "$HERE/logs"
echo "[run_p5p8.sh] model=$ANTHROPIC_MODEL endpoint=$ANTHROPIC_BASE_URL"
echo "[run_p5p8.sh] worktree=$WORKTREE max_hours=$MAX_HOURS pillars=$PILLARS round_cap=$ROUND_CAP max_cost_usd=$MAX_COST_USD"

exec "$HERE/.venv/bin/python" "$HERE/orchestrator.py" \
  --worktree "$WORKTREE" \
  --state-dir "$HERE/state_p5p8" \
  --task-spec "$HERE/state_p5p8/task_spec_p5p8.md" \
  --pillars-file "$HERE/state_p5p8/pillars_p5p8.json" \
  --max-hours "$MAX_HOURS" \
  --pillars "$PILLARS" \
  --round-cap "$ROUND_CAP" \
  --max-cost-usd "$MAX_COST_USD"

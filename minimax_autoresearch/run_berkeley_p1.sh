#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" \u0026\u0026 pwd)
source $HERE/../.env.minimax
exec $HERE/.venv/bin/python $HERE/orchestrator.py \
  --worktree /home/claude/tinker-rl-lab-minimax-berkeley-p1 \
  --state-dir /home/claude/tinker-rl-lab/minimax_autoresearch/state_berkeley_p1 \
  --task-spec /home/claude/tinker-rl-lab/minimax_autoresearch/state_berkeley_p1/task_spec_berkeley.md \
  --pillars-file /home/claude/tinker-rl-lab/minimax_autoresearch/state_berkeley_p1/pillars_berkeley.json \
  --max-hours "${1:-8}" --pillars 1 --round-cap 50

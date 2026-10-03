#!/usr/bin/env bash
#
# worktree-new.sh — create an isolated git worktree + branch for one
# unattended run, so parallel/long runs never dirty your main checkout.
#
# Usage: tools/harness/worktree-new.sh <run-name> [base-ref]
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: worktree-new.sh <run-name> [base-ref]" >&2; exit 2; }
NAME="$1"; BASE="${2:-main}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WT_DIR="$(cd "$ROOT/.." && pwd)/worktrees/$NAME"
git -C "$ROOT" worktree add -b "run/$NAME" "$WT_DIR" "$BASE"
echo "worktree: $WT_DIR  (branch run/$NAME)"
echo "run:  WORKDIR=$WT_DIR tools/harness/ralph.sh"
echo "      tools/harness/run-tmux.sh $NAME --workdir $WT_DIR"
echo "cleanup: git worktree remove --force $WT_DIR && git branch -D run/$NAME"

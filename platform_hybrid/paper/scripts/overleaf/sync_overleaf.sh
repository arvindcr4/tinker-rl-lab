#!/usr/bin/env bash
# One-command re-sync of all 8 standalone papers to their Overleaf projects
# via the Overleaf git bridge (Premium). Each project is a separate self-contained
# repo; we mirror the current per-paper bundle into it and push.
#
# Auth: needs an Overleaf git authentication token (Account settings ->
# Git integration -> "Add another token"). Provide it either as:
#   export OL_TOKEN=olp_xxx           # env var, OR
#   echo -n olp_xxx > paper/scripts/overleaf/.ol_token   # gitignored file
#
# Usage:
#   ./sync_overleaf.sh            # sync all 8
#   ./sync_overleaf.sh P3 P8      # sync a subset
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HERE/.work"          # clones + staged bundles (gitignored)
STAGE="$WORK/stage"
mkdir -p "$WORK"

# --- token ---
if [[ -z "${OL_TOKEN:-}" ]]; then
  if [[ -f "$HERE/.ol_token" ]]; then OL_TOKEN="$(cat "$HERE/.ol_token")"; else
    echo "ERROR: set OL_TOKEN env var or create $HERE/.ol_token" >&2; exit 1
  fi
fi
export OL_TOKEN
HELPER='!f() { echo username=git; echo "password=$OL_TOKEN"; }; f'

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-Arvind CR}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-arvindcr4@gmail.com}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

# --- project map: PID  overleaf-project-id  human-name ---
declare -A IDS=(
  [P1]=6a48d49127ac77eca086d403
  [P2]=6a48d57c10c03b93d5d543db
  [P3]=6a48d5aa27ac77eca087438e
  [P4]=6a48d5cc27ac77eca0874ffd
  [P5]=6a48d5de10c03b93d5d560a6
  [P6]=6a48d60527ac77eca0875e10
  [P7]=6a48d62027ac77eca08767e7
  [P8]=6a48d63afe9a24b42d2c63aa
)

# --- build current bundles ---
echo ">> building bundles"
python3 "$HERE/build_bundles.py" "$STAGE"

TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(P1 P2 P3 P4 P5 P6 P7 P8)

for P in "${TARGETS[@]}"; do
  ID="${IDS[$P]:-}"; [[ -z "$ID" ]] && { echo "skip $P (no id)"; continue; }
  echo "==== $P -> $ID ===="
  CL="$WORK/ol_$P"; rm -rf "$CL"
  git -c credential.helper="$HELPER" clone --quiet "https://git.overleaf.com/$ID" "$CL"
  # NOTE: do NOT exclude '*.pdf' — figures/ contains real .pdf plots that must
  # ship. Only skip LaTeX build artifacts and root-level built paper PDFs.
  rsync -a --delete --exclude='.git/' \
    --exclude='*.aux' --exclude='*.log' --exclude='*.out' --exclude='*.blg' \
    --exclude='*.synctex.gz' --exclude='*.fls' --exclude='*.fdb_latexmk' \
    --exclude='build*.log' --exclude='/paper_*.pdf' \
    "$STAGE/$P/" "$CL/"
  ( cd "$CL"
    git add -A
    if git diff --cached --quiet; then
      echo "   no changes"
    else
      git commit -q -m "Sync $P from local paper/ ($(date -u +%Y-%m-%dT%H:%MZ))"
      git -c credential.helper="$HELPER" push --quiet origin HEAD
      echo "   pushed"
    fi )
done
echo ">> done"

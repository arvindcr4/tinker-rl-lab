#!/usr/bin/env bash
# Prepare MiniWoB++ assets for BrowserGym without vendoring them into this repo.

set -euo pipefail

TARGET_DIR="${MINIWOB_REPO_DIR:-$HOME/.cache/tinker-rl-lab/miniwob-plusplus}"
PINNED_COMMIT="7fd85d71a4b60325c6585396ec4f48377d049838"

mkdir -p "$(dirname "$TARGET_DIR")"

if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone https://github.com/Farama-Foundation/miniwob-plusplus.git "$TARGET_DIR"
fi

git -C "$TARGET_DIR" fetch --depth 1 origin "$PINNED_COMMIT"
git -C "$TARGET_DIR" reset --hard "$PINNED_COMMIT"

MINIWOB_HTML_DIR="$TARGET_DIR/miniwob/html/miniwob"

cat <<EOF
MiniWoB++ is ready.

Run this before BrowserGym MiniWoB experiments:

export MINIWOB_URL="file://$MINIWOB_HTML_DIR/"
EOF

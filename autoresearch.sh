#!/usr/bin/env bash
set -euo pipefail

# Count trash/unrelated files still tracked in git
# Categories:
#   1. doc/ — auto-generated HTML mirror (should not be tracked)
#   2. 0xsero_tweets.json — unrelated Twitter data
#   3. experiments/jarvis_config.ini — unrelated config
#   4. experiments/dropbox_uploader.sh — unrelated utility
#   5. autoresearch.jsonl — previous session state (not project code)

TRASH=0

# Count doc/ files (auto-generated HTML mirror)
DOC_COUNT=$(git ls-files -- doc/ 2>/dev/null | wc -l)
TRASH=$((TRASH + DOC_COUNT))

# Count specific unrelated files
for f in 0xsero_tweets.json experiments/jarvis_config.ini experiments/dropbox_uploader.sh; do
    if git ls-files --error-unmatch "$f" &>/dev/null; then
        TRASH=$((TRASH + 1))
    fi
done

# Total tracked files
TOTAL=$(git ls-files | wc -l)

# Repo size (tracked files only, in KB)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC trash_files_remaining=$TRASH"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

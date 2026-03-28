#!/usr/bin/env bash
set -euo pipefail

ISSUES=0

# 1. Leftover generated doc mirror still present in the workspace
if [ -d doc ]; then
    ISSUES=$((ISSUES + 1))
fi

# 2. Nested git repositories inside the generated mirror
NESTED_GITS=$(find doc -type d -name .git 2>/dev/null | wc -l || true)
ISSUES=$((ISSUES + NESTED_GITS))

# 3. Generated mirror still contains its own virtualenv
if [ -d doc/.venv-axolotl ]; then
    ISSUES=$((ISSUES + 1))
fi

TOTAL=$(git ls-files | wc -l)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC workspace_cleanup_issues=$ISSUES"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

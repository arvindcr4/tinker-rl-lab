#!/usr/bin/env bash
set -euo pipefail

ISSUES=0

# Count project-local Python bytecode caches outside ignored virtualenvs / git metadata.
PYCACHE_COUNT=$(find . \
  -path './.git' -prune -o \
  -path './.venv-axolotl' -prune -o \
  -type d -name '__pycache__' -print | wc -l)
ISSUES=$((ISSUES + PYCACHE_COUNT))

TOTAL=$(git ls-files | wc -l)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC workspace_bytecode_issues=$ISSUES"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

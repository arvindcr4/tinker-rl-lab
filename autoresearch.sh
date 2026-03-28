#!/usr/bin/env bash
set -euo pipefail

ISSUES=0

# Session docs should accurately reflect the finished cleanup state.
if grep -q '(Nothing yet' autoresearch.md 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi
if grep -q 'starting fresh' autoresearch.md 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi

# Dashboard should mention the final cleanup state and current branch head.
if ! grep -q 'repo_size_kb' autoresearch.jsonl 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi
if ! grep -q 'workspace_cleanup_issues' autoresearch.jsonl 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi
if ! grep -q 'workspace_bytecode_issues' autoresearch.jsonl 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi

TOTAL=$(git ls-files | wc -l)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC session_doc_issues=$ISSUES"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

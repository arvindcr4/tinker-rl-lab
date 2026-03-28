#!/usr/bin/env bash
set -euo pipefail

ISSUES=0

# 1. doc/ files still tracked
DOC_COUNT=$(git ls-files -- doc/ 2>/dev/null | wc -l)
ISSUES=$((ISSUES + DOC_COUNT))

# 2. Unrelated files
for f in 0xsero_tweets.json experiments/jarvis_config.ini experiments/dropbox_uploader.sh; do
    if git ls-files --error-unmatch "$f" &>/dev/null; then
        ISSUES=$((ISSUES + 1))
    fi
done

# 3. PDFs tracked despite *.pdf in .gitignore
PDF_COUNT=$(git ls-files -- '*.pdf' 2>/dev/null | wc -l)
ISSUES=$((ISSUES + PDF_COUNT))

# 4. Missing .gitignore entries
if ! grep -qE '\.venv-(\*|axolotl)' .gitignore 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi

# 5. README references non-existent directories
for dir in grpo-results rl-gym rl-master; do
    if grep -q "$dir/" README.md 2>/dev/null && ! [ -d "$dir" ]; then
        ISSUES=$((ISSUES + 1))
    fi
done

# 6. Duplicate grpo-results/ tracked
GRPO_COUNT=$(git ls-files -- grpo-results/ 2>/dev/null | wc -l)
ISSUES=$((ISSUES + GRPO_COUNT))

# Totals
TOTAL=$(git ls-files | wc -l)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC quality_issues=$ISSUES"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

#!/usr/bin/env bash
set -euo pipefail

# Count quality issues in the repo
ISSUES=0

# 1. doc/ files still tracked (auto-generated HTML)
DOC_COUNT=$(git ls-files -- doc/ 2>/dev/null | wc -l)
ISSUES=$((ISSUES + DOC_COUNT))

# 2. Specific unrelated files
for f in 0xsero_tweets.json experiments/jarvis_config.ini experiments/dropbox_uploader.sh; do
    if git ls-files --error-unmatch "$f" &>/dev/null; then
        ISSUES=$((ISSUES + 1))
    fi
done

# 3. Duplicated notebooks (grpo-results/ duplicates of atropos/notebooks/)
for f in grpo-results/*.ipynb; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    if git ls-files --error-unmatch "atropos/notebooks/$base" &>/dev/null 2>&1; then
        if git ls-files --error-unmatch "$f" &>/dev/null 2>&1; then
            ISSUES=$((ISSUES + 1))
        fi
    fi
done

# 4. PDFs tracked despite *.pdf in .gitignore (force-added compiled artifacts)
PDF_COUNT=$(git ls-files -- '*.pdf' 2>/dev/null | wc -l)
ISSUES=$((ISSUES + PDF_COUNT))

# 5. Missing .gitignore entries (check if .venv-axolotl pattern exists)
if ! grep -qE '\.venv-(\*|axolotl)' .gitignore 2>/dev/null; then
    ISSUES=$((ISSUES + 1))
fi

# Total tracked files
TOTAL=$(git ls-files | wc -l)

# Repo size (tracked files only, in KB)
REPO_SIZE_KB=$(git ls-files -z | xargs -0 du -sk 2>/dev/null | awk '{s+=$1} END {print s}')

echo "METRIC trash_files_remaining=$ISSUES"
echo "METRIC repo_tracked_files=$TOTAL"
echo "METRIC repo_size_kb=$REPO_SIZE_KB"

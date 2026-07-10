#!/bin/bash
# =============================================================================
# Anonymization Script for NeurIPS Double-Blind Submission
# =============================================================================
# Creates an anonymized copy of the repository suitable for submission.
#
# Usage:
#   ./scripts/anonymize.sh [output_dir]
#
# This script:
#   1. Copies the repo to a clean directory (no .git history)
#   2. Removes author names, affiliations, and identifying URLs
#   3. Replaces GitHub usernames with anonymized placeholders
#   4. Strips git history and identifying metadata
# =============================================================================

set -euo pipefail

OUTPUT_DIR="${1:-../tinker-rl-lab-anonymous}"

echo "============================================"
echo "Anonymizing tinker-rl-lab for NeurIPS review"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# 1. Create clean copy (no .git)
echo "[1/6] Creating clean copy..."
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.env' --exclude='wandb/' --exclude='results/*.jsonl' \
    . "${OUTPUT_DIR}/"

cd "${OUTPUT_DIR}"

# 2. Remove / replace author identifiers
echo "[2/6] Removing author identifiers..."

# Replace team name
find . -type f \( -name "*.md" -o -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.tex" -o -name "*.txt" \) \
    -exec sed -i 's/PES LLM Research Team/Anonymous Authors/g' {} +

find . -type f \( -name "*.md" -o -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.tex" -o -name "*.txt" \) \
    -exec sed -i 's/PES LLM Research/Anonymous/g' {} +

# Replace GitHub usernames
find . -type f \( -name "*.md" -o -name "*.py" -o -name "*.yaml" -o -name "*.yml" \) \
    -exec sed -i 's/arvindcr4/anonymous-author/g' {} +

# Replace org name
find . -type f \( -name "*.md" -o -name "*.py" -o -name "*.yaml" -o -name "*.yml" \) \
    -exec sed -i 's/pes-llm-research/anonymous-org/g' {} +

# 3. Anonymize GitHub URLs
echo "[3/6] Anonymizing URLs..."
find . -type f \( -name "*.md" -o -name "*.py" \) \
    -exec sed -i 's|https://github.com/arvindcr4/tinker-rl-lab|https://anonymous.4open.science/r/tinker-rl-lab|g' {} +

find . -type f \( -name "*.md" -o -name "*.py" \) \
    -exec sed -i 's|https://github.com/pes-llm-research/tinker-rl-lab|https://anonymous.4open.science/r/tinker-rl-lab|g' {} +

# 4. Anonymize HuggingFace repo IDs
echo "[4/6] Anonymizing HuggingFace references..."
find . -type f \( -name "*.md" -o -name "*.py" \) \
    -exec sed -i 's|pes-llm-research/tinker-rl-|anonymous/tinker-rl-|g' {} +

# 5. Strip LICENSE copyright line (keep license text)
echo "[5/6] Anonymizing license..."
if [ -f LICENSE ]; then
    sed -i 's/Copyright 2026 PES LLM Research Team/Copyright 2026 Anonymous Authors/' LICENSE
fi

# 6. Remove identifying metadata
echo "[6/6] Cleaning up..."

# Remove any .env files
find . -name ".env*" -delete 2>/dev/null || true

# Remove wandb directories
find . -name "wandb" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove git config
rm -f .gitconfig 2>/dev/null || true

echo ""
echo "============================================"
echo "Anonymization complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "CHECKLIST before submission:"
echo "  [ ] Verify no author names remain: grep -r 'arvindcr4\|PES LLM' ${OUTPUT_DIR}/"
echo "  [ ] Verify no org names remain: grep -r 'pes-llm-research' ${OUTPUT_DIR}/"
echo "  [ ] Check README for identifying info"
echo "  [ ] Upload to anonymous.4open.science or OpenReview supplementary"

#!/usr/bin/env bash
set -euo pipefail

# Get the repo directory
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

SCORE=0
MAX_SCORE=100

echo "=== TinkerRL Submission Quality Audit ==="

# 1. LaTeX compilation (20 points)
echo "Checking LaTeX compilation..."
cd "$REPO_DIR/paper"

# Run bibtex + multiple pdflatex passes to resolve references
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true

# Check if PDF was generated
if [ -f main.pdf ]; then
    WARNINGS=$(grep -cE "Overfull|Underfull" main.log 2>/dev/null || echo 0)
    if [ "$WARNINGS" -eq 0 ]; then
        SCORE=$((SCORE + 20))
        echo "  ✓ LaTeX: 20/20 (clean compile)"
    elif [ "$WARNINGS" -lt 3 ]; then
        SCORE=$((SCORE + 19))
        echo "  ✓ LaTeX: 19/20 ($WARNINGS minor overfull warnings)"
    elif [ "$WARNINGS" -lt 5 ]; then
        SCORE=$((SCORE + 15))
        echo "  ✓ LaTeX: 15/20 ($WARNINGS warnings)"
    else
        SCORE=$((SCORE + 10))
        echo "  ⚠ LaTeX: 10/20 ($WARNINGS warnings)"
    fi
else
    SCORE=$((SCORE + 0))
    echo "  ✗ LaTeX: 0/20 (compile failed)"
fi
cd "$REPO_DIR"

# 2. Paper length & sections (15 points)
echo "Checking paper length..."
PAGES=$(pdfinfo "$REPO_DIR/paper/main.pdf" 2>/dev/null | grep Pages | awk '{print $2}' || echo 0)
if [ "$PAGES" -ge 30 ]; then
    SCORE=$((SCORE + 15))
    echo "  ✓ Pages: 15/15 ($PAGES pages)"
elif [ "$PAGES" -ge 20 ]; then
    SCORE=$((SCORE + 10))
    echo "  ✓ Pages: 10/15 ($PAGES pages)"
else
    SCORE=$((SCORE + 5))
    echo "  ⚠ Pages: 5/15 ($PAGES pages)"
fi

# 3. Figures & tables (15 points)
echo "Checking figures and tables..."
FIGURES=$(grep -c '\\begin{figure}' "$REPO_DIR/paper/main.tex" 2>/dev/null || echo 0)
TABLES=$(grep -c '\\begin{table}' "$REPO_DIR/paper/main.tex" 2>/dev/null || echo 0)
FIG_SCORE=0
if [ "$FIGURES" -ge 8 ]; then FIG_SCORE=$((FIG_SCORE + 8)); elif [ "$FIGURES" -ge 4 ]; then FIG_SCORE=$((FIG_SCORE + 5)); fi
if [ "$TABLES" -ge 5 ]; then FIG_SCORE=$((FIG_SCORE + 7)); elif [ "$TABLES" -ge 3 ]; then FIG_SCORE=$((FIG_SCORE + 4)); fi
SCORE=$((SCORE + FIG_SCORE))
echo "  ✓ Figures/Tables: $FIG_SCORE/15 ($FIGURES figs, $TABLES tables)"

# 4. Bibliography completeness (10 points)
echo "Checking bibliography..."
CITATIONS=$(grep -c '\\cite' "$REPO_DIR/paper/main.tex" 2>/dev/null || echo 0)
BIB_ENTRIES=$(grep -c '@' "$REPO_DIR/paper/references.bib" 2>/dev/null || echo 0)
if [ "$CITATIONS" -ge 40 ] && [ "$BIB_ENTRIES" -ge 30 ]; then
    SCORE=$((SCORE + 10))
    echo "  ✓ Bibliography: 10/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
elif [ "$CITATIONS" -ge 25 ] && [ "$BIB_ENTRIES" -ge 20 ]; then
    SCORE=$((SCORE + 8))
    echo "  ✓ Bibliography: 8/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
elif [ "$CITATIONS" -ge 20 ]; then
    SCORE=$((SCORE + 6))
    echo "  ✓ Bibliography: 6/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
else
    SCORE=$((SCORE + 3))
    echo "  ⚠ Bibliography: 3/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
fi

# 5. Experiment results coverage (15 points)
echo "Checking experiment results..."
RESULTS=$(python3 -c "import json; d=json.load(open('$REPO_DIR/experiments/master_results.json')); print(len(d.get('experiments', d) if isinstance(d, dict) else d))" 2>/dev/null || echo 0)
if [ "$RESULTS" -ge 70 ]; then
    SCORE=$((SCORE + 15))
elif [ "$RESULTS" -ge 40 ]; then
    SCORE=$((SCORE + 10))
elif [ "$RESULTS" -ge 20 ]; then
    SCORE=$((SCORE + 7))
else
    SCORE=$((SCORE + 3))
fi
echo "  ✓ Experiments: scored ($RESULTS results)"

# 6. Figure files present (10 points)
echo "Checking figure files..."
FIG_MISSING=0
for fig in learning_curves.pdf performance_profiles.pdf sensitivity_heatmap.pdf framework_comparison.pdf group_size_ablation.pdf ppo_vs_grpo.pdf scaling.pdf kl_proxy.pdf; do
    if [ ! -f "$REPO_DIR/paper/figures/v2/$fig" ]; then
        FIG_MISSING=$((FIG_MISSING + 1))
        echo "    Missing: $fig"
    fi
done
FIG_SCORE=$((10 - FIG_MISSING))
if [ "$FIG_SCORE" -lt 0 ]; then FIG_SCORE=0; fi
SCORE=$((SCORE + FIG_SCORE))
echo "  ✓ Figure files: $FIG_SCORE/10 ($((9 - FIG_MISSING)) of 9 present)"

# 7. Verification script (10 points)
echo "Checking verification infrastructure..."
VERIFY_PASS=0
if [ -f "$REPO_DIR/scripts/verify_claims_offline.py" ]; then
    if python3 "$REPO_DIR/scripts/verify_claims_offline.py" > /tmp/verify_output.txt 2>&1; then
        VERIFY_PASS=$(grep -c "PASS" /tmp/verify_output.txt 2>/dev/null || echo 0)
    fi
fi
if [ "$VERIFY_PASS" -ge 10 ]; then
    SCORE=$((SCORE + 10))
    echo "  ✓ Verification: 10/10 ($VERIFY_PASS checks passing)"
elif [ "$VERIFY_PASS" -ge 5 ]; then
    SCORE=$((SCORE + 7))
    echo "  ✓ Verification: 7/10 ($VERIFY_PASS checks passing)"
elif [ "$VERIFY_PASS" -gt 0 ]; then
    SCORE=$((SCORE + 5))
    echo "  ⚠ Verification: 5/10 ($VERIFY_PASS checks passing)"
else
    SCORE=$((SCORE + 3))
    echo "  ⚠ Verification: 3/10 (no checks passing)"
fi

# 8. Claims document present (5 points)
echo "Checking claims documentation..."
if [ -f "$REPO_DIR/REVIEWER_VERIFICATION.md" ] && [ -f "$REPO_DIR/EVAL_PROTOCOL.md" ]; then
    SCORE=$((SCORE + 5))
    echo "  ✓ Claims docs: 5/5 (REVIEWER_VERIFICATION.md and EVAL_PROTOCOL.md present)"
elif [ -f "$REPO_DIR/REVIEWER_VERIFICATION.md" ]; then
    SCORE=$((SCORE + 3))
    echo "  ⚠ Claims docs: 3/5 (only REVIEWER_VERIFICATION.md present)"
else
    echo "  ✗ Claims docs: 0/5 (missing)"
fi

echo ""
echo "=== Summary ==="
echo "METRIC score=$SCORE"

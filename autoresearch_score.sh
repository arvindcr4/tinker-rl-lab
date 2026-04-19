#!/bin/bash
set -euo pipefail

SCORE=0
MAX_SCORE=100
DETAILS=""

cd /home/user/workspace/tinker-rl-lab

# 1. LaTeX compilation (20 points)
cd paper
if pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/latex_compile.log 2>&1; then
    WARNINGS=$(grep -ciE "warning|overfull|underfull" /tmp/latex_compile.log 2>/dev/null || echo 0)
    WARNINGS=${WARNINGS##*$'\n'}
    WARNINGS=$((WARNINGS + 0))
    if [ "$WARNINGS" -lt 5 ]; then
        SCORE=$((SCORE + 20))
        DETAILS="$DETAILS\nLatex: 20/20 (clean compile)"
    elif [ "$WARNINGS" -lt 15 ]; then
        SCORE=$((SCORE + 15))
        DETAILS="$DETAILS\nLatex: 15/20 ($WARNINGS warnings)"
    else
        SCORE=$((SCORE + 10))
        DETAILS="$DETAILS\nLatex: 10/20 ($WARNINGS warnings)"
    fi
    # Run bibtex + second pass
    bibtex main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
else
    SCORE=$((SCORE + 0))
    DETAILS="$DETAILS\nLatex: 0/20 (compile failed)"
fi
cd ..

# 2. Paper length & sections (15 points)
PAGES=$(pdfinfo paper/main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo 0)
if [ "$PAGES" -ge 30 ]; then
    SCORE=$((SCORE + 15))
    DETAILS="$DETAILS\nPages: 15/15 ($PAGES pages)"
elif [ "$PAGES" -ge 20 ]; then
    SCORE=$((SCORE + 10))
    DETAILS="$DETAILS\nPages: 10/15 ($PAGES pages)"
else
    SCORE=$((SCORE + 5))
    DETAILS="$DETAILS\nPages: 5/15 ($PAGES pages)"
fi

# 3. Figures & tables (15 points)
FIGURES=$(grep -c '\\begin{figure' paper/main.tex 2>/dev/null || echo 0)
TABLES=$(grep -c '\\begin{table' paper/main.tex 2>/dev/null || echo 0)
FIG_SCORE=0
if [ "$FIGURES" -ge 8 ]; then FIG_SCORE=$((FIG_SCORE + 8)); elif [ "$FIGURES" -ge 4 ]; then FIG_SCORE=$((FIG_SCORE + 5)); fi
if [ "$TABLES" -ge 5 ]; then FIG_SCORE=$((FIG_SCORE + 7)); elif [ "$TABLES" -ge 3 ]; then FIG_SCORE=$((FIG_SCORE + 4)); fi
SCORE=$((SCORE + FIG_SCORE))
DETAILS="$DETAILS\nFigures/Tables: $FIG_SCORE/15 ($FIGURES figs, $TABLES tables)"

# 4. Bibliography completeness (10 points)
CITATIONS=$(grep -c '\\cite' paper/main.tex 2>/dev/null || echo 0)
BIB_ENTRIES=$(grep -c '@' paper/references.bib 2>/dev/null || echo 0)
if [ "$CITATIONS" -ge 40 ] && [ "$BIB_ENTRIES" -ge 30 ]; then
    SCORE=$((SCORE + 10))
    DETAILS="$DETAILS\nBibliography: 10/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
elif [ "$CITATIONS" -ge 20 ]; then
    SCORE=$((SCORE + 6))
    DETAILS="$DETAILS\nBibliography: 6/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
else
    SCORE=$((SCORE + 3))
    DETAILS="$DETAILS\nBibliography: 3/10 ($CITATIONS citations, $BIB_ENTRIES entries)"
fi

# 5. Experiment results coverage (15 points)
RESULTS=$(python3 -c "import json; d=json.load(open('experiments/master_results.json')); print(len(d.get('experiments', d) if isinstance(d, dict) else d))" 2>/dev/null || echo 0)
if [ "$RESULTS" -ge 70 ]; then
    SCORE=$((SCORE + 15))
elif [ "$RESULTS" -ge 40 ]; then
    SCORE=$((SCORE + 10))
elif [ "$RESULTS" -ge 20 ]; then
    SCORE=$((SCORE + 7))
else
    SCORE=$((SCORE + 3))
fi
DETAILS="$DETAILS\nExperiments: scored ($RESULTS results in master_results.json)"

# 6. Code quality (10 points)
PYTHON_FILES=$(find experiments -name "*.py" | wc -l)
# Check for docstrings
DOCSTRINGS=$(grep -r '"""' experiments/*.py 2>/dev/null | wc -l || echo 0)
if [ "$DOCSTRINGS" -ge 10 ]; then
    SCORE=$((SCORE + 10))
elif [ "$DOCSTRINGS" -ge 5 ]; then
    SCORE=$((SCORE + 7))
else
    SCORE=$((SCORE + 3))
fi
DETAILS="$DETAILS\nCode quality: scored ($PYTHON_FILES py files, $DOCSTRINGS docstrings)"

# 7. Report completeness (15 points)
if [ -f "reports/final/capstone_final_report.md" ]; then
    REPORT_LINES=$(wc -l < reports/final/capstone_final_report.md)
    if [ "$REPORT_LINES" -ge 1500 ]; then
        SCORE=$((SCORE + 15))
    elif [ "$REPORT_LINES" -ge 1000 ]; then
        SCORE=$((SCORE + 10))
    else
        SCORE=$((SCORE + 5))
    fi
    DETAILS="$DETAILS\nReport: scored ($REPORT_LINES lines)"
else
    DETAILS="$DETAILS\nReport: 0/15 (missing)"
fi

echo "METRIC score=$SCORE"
echo -e "$DETAILS"

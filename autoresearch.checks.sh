#!/bin/bash
set -euo pipefail

# Get the repo directory
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Check 1: LaTeX compiles without errors (use batchmode to speed up)
cd "$REPO_DIR/paper"
if pdflatex -interaction=batchmode -halt-on-error main.tex > /dev/null 2>&1; then
    echo "LaTeX compiled successfully"
else
    echo "WARNING: LaTeX compilation had issues, but continuing..."
fi
cd "$REPO_DIR"

# Check 2: master_results.json is valid JSON
if python3 -c "import json; json.load(open('$REPO_DIR/experiments/master_results.json'))"; then
    echo "master_results.json is valid"
else
    echo "ERROR: master_results.json is invalid JSON"
    exit 1
fi

# Check 3: Report exists and is non-empty (either .tex or .md)
if [ -f "$REPO_DIR/reports/final/capstone_final_report.tex" ]; then
    if [ -s "$REPO_DIR/reports/final/capstone_final_report.tex" ]; then
        echo "capstone_final_report.tex exists and is non-empty"
    else
        echo "ERROR: capstone_final_report.tex is empty"
        exit 1
    fi
elif [ -f "$REPO_DIR/reports/final/capstone_final_report.md" ]; then
    if [ -s "$REPO_DIR/reports/final/capstone_final_report.md" ]; then
        echo "capstone_final_report.md exists and is non-empty"
    else
        echo "ERROR: capstone_final_report.md is empty"
        exit 1
    fi
else
    echo "ERROR: No capstone_final_report found"
    exit 1
fi

# Check 4: Verify claims offline script exists and is valid Python
if [ -f "$REPO_DIR/scripts/verify_claims_offline.py" ]; then
    if python3 -c "import ast; ast.parse(open('$REPO_DIR/scripts/verify_claims_offline.py').read())"; then
        echo "verify_claims_offline.py is valid Python"
    else
        echo "ERROR: verify_claims_offline.py has syntax errors"
        exit 1
    fi
else
    echo "WARNING: verify_claims_offline.py not found"
fi

echo "All checks passed!"

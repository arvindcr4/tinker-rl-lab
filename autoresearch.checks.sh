#!/bin/bash
set -euo pipefail
cd /home/user/workspace/tinker-rl-lab

# Check 1: LaTeX compiles without errors
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1
cd ..

# Check 2: master_results.json is valid JSON
python3 -c "import json; json.load(open('experiments/master_results.json'))"

# Check 3: Report exists and is non-empty
test -s reports/final/capstone_final_report.md

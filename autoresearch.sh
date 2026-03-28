#!/usr/bin/env bash
set -euo pipefail

SCORE=0
FILE_COUNT=0
GUIDE_SECTIONS=0

# 1. Check required template files exist and are non-empty
REQUIRED_FILES=(
    "ai-scientist-template/experiment.py"
    "ai-scientist-template/plot.py"
    "ai-scientist-template/prompt.json"
    "ai-scientist-template/seed_ideas.json"
    "ai-scientist-template/latex/template.tex"
    "ai-scientist-template/run_0/final_info.json"
    "ai-scientist-template/README.md"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        FILE_COUNT=$((FILE_COUNT + 1))
        SCORE=$((SCORE + 3))  # 3 points per file, max 21
    fi
done

# 2. Check experiment.py has required interface
if grep -q "argparse" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi
if grep -q "final_info.json" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi
if grep -q "means" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi

# 3. Check final_info.json has correct structure
if python3 -c "
import json, sys
with open('ai-scientist-template/run_0/final_info.json') as f:
    d = json.load(f)
assert 'gsm8k_training' in d
assert 'means' in d['gsm8k_training']
assert 'stderrs' in d['gsm8k_training']
print('OK')
" 2>/dev/null | grep -q "OK"; then
    SCORE=$((SCORE + 5))
fi

# 4. Check prompt.json structure
if python3 -c "
import json
with open('ai-scientist-template/prompt.json') as f:
    d = json.load(f)
assert 'system' in d and 'task_description' in d
print('OK')
" 2>/dev/null | grep -q "OK"; then
    SCORE=$((SCORE + 5))
fi

# 5. Check seed_ideas.json structure
if python3 -c "
import json
with open('ai-scientist-template/seed_ideas.json') as f:
    d = json.load(f)
assert isinstance(d, list) and len(d) >= 1
assert all(k in d[0] for k in ['Name','Title','Experiment','Interestingness','Feasibility','Novelty'])
print('OK')
" 2>/dev/null | grep -q "OK"; then
    SCORE=$((SCORE + 5))
fi

# 6. Check LaTeX template has required sections
if grep -q "ABSTRACT_PLACEHOLDER\|\\\\begin{abstract}" ai-scientist-template/latex/template.tex 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi
if grep -q "references.bib\|filecontents" ai-scientist-template/latex/template.tex 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi
if grep -q "graphicspath" ai-scientist-template/latex/template.tex 2>/dev/null; then
    SCORE=$((SCORE + 2))
fi

# 7. Check experiment.py Python syntax
if python3 -c "import ast; ast.parse(open('ai-scientist-template/experiment.py').read())" 2>/dev/null; then
    SCORE=$((SCORE + 5))
fi

# 8. Check plot.py Python syntax
if python3 -c "import ast; ast.parse(open('ai-scientist-template/plot.py').read())" 2>/dev/null; then
    SCORE=$((SCORE + 5))
fi

# 9. Check README has key sections
for section in "Setup" "Launch" "Template File Reference" "Cloud GPU" "Thesis"; do
    if grep -qi "$section" ai-scientist-template/README.md 2>/dev/null; then
        GUIDE_SECTIONS=$((GUIDE_SECTIONS + 1))
        SCORE=$((SCORE + 2))
    fi
done

# 10. Check no hardcoded API keys
if ! grep -qE '(sk-ant-|sk-|tml-|AAAA)' ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 5))
fi

# 11. experiment.py has multi-seed support
if grep -q "num_seeds" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi

# 12. experiment.py has configurable model selection
if grep -q "MODELS" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi

# 13. seed_ideas.json has 2+ ideas
if python3 -c "import json; d=json.load(open('ai-scientist-template/seed_ideas.json')); assert len(d)>=2; print('OK')" 2>/dev/null | grep -q "OK"; then
    SCORE=$((SCORE + 3))
fi

# 14. LaTeX has at least 5 BibTeX entries
if python3 -c "
import re
tex = open('ai-scientist-template/latex/template.tex').read()
entries = re.findall(r'@\w+\{', tex)
assert len(entries) >= 5, f'only {len(entries)} entries'
print('OK')
" 2>/dev/null | grep -q "OK"; then
    SCORE=$((SCORE + 3))
fi

# 15. README has installation commands (pip, conda, git clone)
if grep -q "pip install" ai-scientist-template/README.md 2>/dev/null && \
   grep -q "git clone" ai-scientist-template/README.md 2>/dev/null; then
    SCORE=$((SCORE + 3))
fi

# 16. experiment.py handles GPU/CPU gracefully
if grep -q "device_map\|cuda\|torch.device" ai-scientist-template/experiment.py 2>/dev/null; then
    SCORE=$((SCORE + 2))
fi

# 17. plot.py generates multiple plot types
PLOT_COUNT=$(grep -c "savefig" ai-scientist-template/plot.py 2>/dev/null || echo 0)
if [ "$PLOT_COUNT" -ge 3 ]; then
    SCORE=$((SCORE + 2))
fi

# 18. No hardcoded API keys in the ENTIRE repo (not just template)
if ! grep -rn 'tml-[A-Za-z0-9]\{20,\}' --include='*.py' --include='*.sh' . 2>/dev/null | grep -v '.git/' | grep -v '.venv' | grep -q .; then
    SCORE=$((SCORE + 3))
fi

# Cap at 100
if [ "$SCORE" -gt 100 ]; then
    SCORE=100
fi

echo "METRIC integration_completeness=$SCORE"
echo "METRIC template_file_count=$FILE_COUNT"
echo "METRIC guide_sections=$GUIDE_SECTIONS"

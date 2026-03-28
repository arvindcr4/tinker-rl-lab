# Autoresearch Dashboard: AI Scientist Integration

**Runs:** 5 | **Kept:** 5 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** integration_completeness: 78 (#1)
**Best:** integration_completeness: 100 (#3-5, +28.2%)

| # | commit | integration_completeness | status | description |
|---|--------|------------------------|--------|-------------|
| 1 | b2c5d3a | 78 | keep | initial template: all 7 files, valid structure |
| 2 | e068f24 | 97 (+24.4%) | keep | expanded scoring, all quality checks pass |
| 3 | 4a846ec | 100 (+28.2%) | keep | 3rd seed idea, requirements.txt, fixed 8 hardcoded API keys |
| 4 | 90ebf86 | 100 (+28.2%) | keep | fixed experiment.py to match current TRL GRPOTrainer API |
| 5 | b2fed25 | 100 (+28.2%) | keep | added WRITEUP.md with prior results context |

## Template Files (9 total)
- `experiment.py` — TRL GRPOTrainer on GSM8K with Qwen 0.5B-3B (current API)
- `plot.py` — 3 plot types: accuracy bars, training curves, metric summary
- `prompt.json` — GRPO math reasoning domain prompt
- `seed_ideas.json` — 3 seed ideas: partial credit, curriculum, group size scaling
- `latex/template.tex` — ICLR-format paper with 8 embedded references
- `run_0/final_info.json` — Placeholder baseline (regenerate on GPU)
- `requirements.txt` — Python dependencies
- `README.md` — Comprehensive setup guide (local, vast.ai, Colab)
- `WRITEUP.md` — Research context for paper writing

## Security Fix
Removed hardcoded Tinker API keys from 8 files (replaced with env var lookups).

## Quick Start
```bash
cp -r ai-scientist-template ~/AI-Scientist/templates/grpo_gsm8k
cd ~/AI-Scientist
python launch_scientist.py --model "claude-opus-4-6" --experiment grpo_gsm8k --num-ideas 2
```

# Autoresearch Dashboard: AI Scientist Integration

**Runs:** 3 | **Kept:** 3 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** integration_completeness: 78 (#1)
**Best:** integration_completeness: 100 (#3, +28.2%)

| # | commit | integration_completeness | status | description |
|---|--------|------------------------|--------|-------------|
| 1 | b2c5d3a | 78 | keep | initial template: all 7 files, valid structure |
| 2 | e068f24 | 97 (+24.4%) | keep | expanded scoring, all quality checks pass |
| 3 | 4a846ec | 100 (+28.2%) | keep | 3rd seed idea, requirements.txt, fixed 8 hardcoded API keys |

## Template Files Created
- `ai-scientist-template/experiment.py` — TRL GRPOTrainer on GSM8K with Qwen 0.5B-3B
- `ai-scientist-template/plot.py` — 3 plot types: accuracy bars, training curves, metric summary
- `ai-scientist-template/prompt.json` — GRPO math reasoning domain prompt
- `ai-scientist-template/seed_ideas.json` — 3 seed ideas: partial credit, curriculum, group size scaling
- `ai-scientist-template/latex/template.tex` — ICLR-format paper with 8 embedded references
- `ai-scientist-template/run_0/final_info.json` — Placeholder baseline (regenerate on GPU)
- `ai-scientist-template/requirements.txt` — Python dependencies
- `ai-scientist-template/README.md` — Comprehensive setup guide

## Security Fix
Removed hardcoded Tinker API keys from 8 files (replaced with env var lookups).

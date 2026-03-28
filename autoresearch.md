# Autoresearch: AI Scientist Integration

## Objective
Create a complete, working Sakana AI Scientist template for the GRPO math reasoning research in this repo. The template should let AI Scientist autonomously generate experiment ideas, run them, write papers, and self-review — producing results that feed back into the capstone thesis and conference paper.

## Metrics
- **Primary**: integration_completeness (score 0-100, higher is better)
  - 0-20: Template files exist but are stubs
  - 20-40: Template files are complete but untested
  - 40-60: Template passes syntax/import checks
  - 60-80: Baseline runs successfully, AI Scientist can load template
  - 80-100: End-to-end pipeline tested with at least one idea
- **Secondary**: template_file_count, guide_sections

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `ai-scientist-template/experiment.py` — GRPO training script (TRL-based, local GPU)
- `ai-scientist-template/plot.py` — Visualization script
- `ai-scientist-template/prompt.json` — System prompt and task description
- `ai-scientist-template/seed_ideas.json` — Example ideas for few-shot
- `ai-scientist-template/latex/template.tex` — LaTeX paper template
- `ai-scientist-template/run_0/final_info.json` — Baseline results
- `ai-scientist-template/README.md` — Setup and usage guide

## Off Limits
- `atropos/` — core project code
- `experiments/` — existing experiment data
- `reports/` — existing papers and thesis

## Constraints
- Template must work with AI Scientist's expected interface (experiment.py --out_dir, final_info.json format)
- experiment.py must be self-contained and runnable on a single GPU
- Must not require Tinker API (runs locally with TRL)
- Must not expose API keys

## What's Been Tried
- Run 1: Created complete template with all 6 required files + README guide
  - experiment.py: TRL GRPOTrainer with Qwen 0.5B-3B, LoRA, 3-seed
  - plot.py: accuracy comparison, training curves, metric summary
  - prompt.json: tailored to GRPO math reasoning domain
  - seed_ideas.json: partial credit reward + curriculum learning
  - LaTeX template with domain-relevant embedded references
  - Placeholder baseline in run_0/final_info.json
  - Comprehensive README with setup steps, cloud GPU instructions, thesis integration

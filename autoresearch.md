# Autoresearch: Codebase Cleanup

## Objective
Remove trash, unrelated, and auto-generated files from the tinker-rl-lab repository. The repo is an RL research project for LLM fine-tuning (GRPO, PPO, DPO etc.) with Tinker/Atropos. It has accumulated bloat: a 337M auto-generated HTML doc/ mirror, unrelated data files, and missing gitignore entries.

## Metrics
- **Primary**: trash_files_remaining (count, lower is better)
  - Counts git-tracked files that are unrelated to the RL research project OR are auto-generated artifacts that shouldn't be version-controlled
- **Secondary**: repo_tracked_files (total tracked file count), repo_size_kb (git ls-files size)

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `.gitignore` — update to exclude generated/unrelated patterns
- `doc/` — entire directory is auto-generated HTML; 235 tracked files, 337M
- `0xsero_tweets.json` — unrelated Twitter data scrape (1.6MB)
- `experiments/jarvis_config.ini` — unrelated Jarvis AI agent config
- `experiments/dropbox_uploader.sh` — unrelated Dropbox utility
- `autoresearch.jsonl` — previous autoresearch session state (not project code)

## Off Limits
- `atropos/` — core project code
- `experiments/implementations/` — RL implementations
- `experiments/notebooks/` — experiment notebooks
- `experiments/tinker-runs/` — training logs and scripts
- `experiments/results/` — training metrics
- `grpo-results/` — GRPO experiment results
- `agentic-rl-finetuning/` — research notebooks
- `capstone-literature-survey/` — literature survey
- `reports/` — final reports and papers
- `README.md` — project readme
- `scientific_audit.py` — audit tool
- `run_coding.sh`, `run_one.sh`, `vast_run.sh` — launcher scripts

## Constraints
- Do NOT remove any file that is part of the actual RL research
- Remove files from git tracking (git rm) not just delete from disk
- Update .gitignore so removed patterns don't get re-added
- Each experiment = one logical batch of removals
- Verify no breakage after each removal

## What's Been Tried
- Removed the tracked `doc/` HTML mirror and unrelated files (`0xsero_tweets.json`, `experiments/jarvis_config.ini`, `experiments/dropbox_uploader.sh`).
- Removed generated HTML exports from `atropos/notebooks/html/`, `experiments/html/`, and the duplicate `grpo-results/` results mirror.
- Updated `.gitignore` to cover generated exports and local virtualenv artifacts.
- Cleaned leftover untracked workspace artifacts (`doc/` mirror remnants and project-local `__pycache__/` directories).

## Current State
- Tracked trash/unrelated artifacts: 0
- Tracked files: ~204
- Tracked repo size: ~4 MB
- README structure references corrected to match the real repository layout.

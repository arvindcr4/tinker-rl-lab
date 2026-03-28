# Worklog: Codebase Cleanup

## Session: 2026-03-28

**Goal:** Remove trash, unrelated, and auto-generated files from tinker-rl-lab.

### Data Summary
- Total tracked files: 468
- `doc/` directory: 235 files (auto-generated HTML), 337MB
- Unrelated files: `0xsero_tweets.json`, `experiments/jarvis_config.ini`, `experiments/dropbox_uploader.sh`
- Missing .gitignore entries: `.venv-axolotl/`, `doc/`

---

## Key Insights
(Updated as experiments progress)

## Next Ideas
- Remove doc/ directory from tracking
- Remove 0xsero_tweets.json
- Remove jarvis_config.ini and dropbox_uploader.sh
- Update .gitignore with proper exclusions
- Check for other bloat (HTML exports in grpo-results/html/, experiments/html/)

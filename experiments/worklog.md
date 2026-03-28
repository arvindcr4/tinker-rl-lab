# Worklog: Codebase Cleanup

## Session: 2026-03-28

**Goal:** Remove trash, unrelated, and auto-generated files from tinker-rl-lab.

### Data Summary (Before)
- Total tracked files: 657
- `doc/` directory: 421 files (auto-generated HTML mirror), ~337MB
- Unrelated files: `0xsero_tweets.json`, `experiments/jarvis_config.ini`, `experiments/dropbox_uploader.sh`
- Duplicate notebooks: `grpo-results/` was identical copy of `atropos/notebooks/`
- Missing .gitignore entries: `.venv-axolotl/`
- Stale README references to non-existent dirs

---

### Run 1: baseline — quality_issues=424 (KEEP)
- Timestamp: 2026-03-28 15:45
- What changed: nothing (baseline measurement)
- Result: 424 trash files, 657 tracked, 358MB
- Insight: Most bloat is the auto-generated doc/ HTML mirror (421 files)
- Next: remove doc/ from tracking

### Run 2: remove doc/ + unrelated files — quality_issues=0 (KEEP)
- Timestamp: 2026-03-28 15:48
- What changed: `git rm -r --cached doc/`, plus prior session already removed unrelated files
- Result: 0 trash, 233 tracked, 16MB
- Insight: doc/ had nested doc/doc/ with its own .git — recursive HTML generation gone wrong
- Next: check for duplicates and gitignore gaps

### Run 3: remove duplicates, fix gitignore — quality_issues=0 (KEEP)
- Timestamp: 2026-03-28 15:52
- What changed: Removed grpo-results/ (7 identical notebooks + 2 files), untracked PDF, added .venv-*/ to .gitignore
- Result: 0 issues, 202 tracked, 4MB
- Insight: grpo-results/ was a full duplicate of atropos/notebooks/
- Next: fix README stale references

### Run 4: update README — quality_issues=0 (KEEP)
- Timestamp: 2026-03-28 15:55
- What changed: Removed references to non-existent rl-gym/, rl-master/, grpo-results/ from README
- Result: 0 issues, 202 tracked, 4MB
- Insight: README had accumulated references to directories that were merged into other locations

---

## Key Insights
- The doc/ directory was an auto-generated HTML mirror that recursively included itself (doc/doc/doc/...)
- grpo-results/ was a complete duplicate of atropos/notebooks/
- foundation_chapter.pdf was force-added despite *.pdf in .gitignore
- Training log files (experiments/tinker-runs/logs/) are intentionally tracked research data despite *.log gitignore

## Summary
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Tracked files | 657 | 202 | -69% |
| Repo size | 358MB | 4MB | -99% |
| Quality issues | 424 | 0 | -100% |

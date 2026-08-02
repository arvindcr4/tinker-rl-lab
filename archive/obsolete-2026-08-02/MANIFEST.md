# Obsolete-File Archive — 2026-08-02

Files moved here because they are obsolete (finished intermediate artifacts, superseded, or
orphaned scratch) and had **zero references from any live/canonical path**. Verified by
`grep` across the whole repo before moving. Nothing here is deleted — restore with
`git mv <path> <original>`.

## What was moved (80 files)

### `./` — orphaned "50 ideas" review pipeline (42 files)
A closed, self-referential ZAI review pipeline over `50_research_ideas_catalog.md`.
Nothing in canonical docs, `zvf-program/`, or `submission/` cites it; active work only
pursues the ZVF idea (category 1). All dated 2026-07-29.
- `50_research_ideas_catalog.md`
- `survey_grounding_cat{1..10}.md`, `adversarial_review_cat{1..10}.md`,
  `proofreading_cat{1..10}.md`, `final_proofread_cat{1..10}.md`
- `ideagent_50_research_ideas.json` (4.4 MB — raw generation data behind the catalog)

### `./` — superseded root docs (2 files)
- `breakthrough.md` — superseded by `BREAKTHROUGH_CHASE_18_ARTIFACTS.md`
- `experiment_design_comparison.md` — 0 external references

### `zvf-program/` — orphaned per-slide one-off scripts (10 files)
Not wired into `apply_all.sh` and not imported as modules by any live script.
Live deck scripts (`build_lightning_deck.py`, `enrich_title_slide.py`,
`enrich_slide_{2,3,4,5,6,8}.py`) remain in `zvf-program/`.
- `inspect_slide.py`, `inspect_slide_2.py`, `read_slide13.py`
- `enrich_slide.py` (v1), `enrich_slide_12.py`, `enrich_slide15.py`, `enrich_slide_17.py`
- `add_title_images.py`, `extract_text.py`, `build_progress_deck.py`

### `autoresearch/` — finished one-shot runs (7 dirs)
1-commit finished runs with no external references.
- `reason-260729-1515`, `reason-260729-1558`
- `orchestrator-260729-1505`, `orchestrator-260730-1855`, `orchestrator-260730-2213`
- `fix-260729-1524`, `learn-260714-1741`

### `docs/` — one-off (1 file)
- `advisor-brief-2026-07-12.md` — defense-prep one-off, 0 references

## Flagged obsolete but KEPT (live references found during verification)

These looked obsolete but are referenced by canonical/active paths, so were **not** moved:
- `FINAL_HANDOFF.md` — cited by live `autoresearch/deli-neurips-tmlr-260802/`
- `INTEGRATION_LOG.md` — referenced by `CHANGELOG.md`
- `BREAKTHROUGH_CHASE_18_ARTIFACTS.md` — referenced by `execution-notes.md`
- `test.json` — loaded by `train_grpo_test.py`, `platform_hybrid/.../p1a_completion_mask_test.py`
- `AGENTS.md` — agentic-tool convention file (also in `.hyperresearch/config.toml` exclude list)
- `platform_gcp/`, `platform_vast/` — referenced by root `README.md`
- `verl/` — referenced by `pyproject.toml`, `execution-notes.md`, `gameplan.md`
- `autoresearch/reason-260727-2155`, `reason-260728-0744` — cited by live deli-neurips campaign
- `autoresearch/reason-260730-1257`, `orchestrator-260730-1818` — referenced by today's active `outputs/build_progress_update_deck.py`
- `platform_hybrid/experiments/results/berkeley/iso_g_cdh_echo.tsv` — read by active `platform_modal/.../iso_g_dynamic_grouping.py`

## Still candidates for a later pass (UNCERTAIN — needs human judgment)
- Prior TinkerRL-Bench submission doc set: `ARTIFACT.md`, `NEURIPS_CHECKLIST_FINAL.md`,
  `ACM_CHECKLIST.md`, `BASELINES.md`, `FRONTIER_INSIGHTS.md` (batch-obsolete if that
  submission's paperwork is closed).
- `output/` — 2,144-PNG render dump (3 weeks old, mostly unreferenced); cold-storage
  candidate after confirming nothing in `thesis/`/`submission/` embeds them.
- `platform_local/` (33 files), `platform_tinker/` (383 files) — 3+ weeks quiet but
  referenced; "paused/done" not abandoned.
- `platform_hybrid/docs/p5p8_improvements/` — 246 auto-generated numbered docs with many
  `100_` version collisions; worth a dedup pass.

## Local-only junk (untracked, gitignored — NOT moved, safe to `rm` locally)
`wandb/` (210 stale run dirs), `tinkerrl.egg-info/`, `.pytest_cache/`, all `__pycache__/`.

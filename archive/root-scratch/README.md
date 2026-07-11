# Root scratch archive

One-off working files moved out of the repository root during the 2026-07-10 professor-facing cleanup. Preserved for provenance; nothing here is a deliverable.

- `team-analysis.pplx.md`, `team-links-audit.pplx.md`, `verify_links_entities.txt` — internal team memos and link audits. Contain real names/handles; the whole `archive/` tree is excluded from anonymized bundles by `blind_review/anonymize_code.py`.
- `patch.py`, `patch.diff`, `inject_patch.py` — one-off codemods with machine-specific paths (see root `INDEX.md` caveats).
- `_ideation_context.md` — ideation-prompt context referenced by `FRONTIER_INSIGHTS.md`.
- `patch_trainer.py`, `patch_wandb.py`, `patch_wandb_imports.py` — one-off codemods for external checkouts (machine-specific paths).
- `update_remaining.py` — imports a since-removed `scratch.refactor` module; kept for history only.
- `plot_monitor.py` — remote-machine training monitor with hardcoded host paths.
- `download_berkeley_transcripts.py` — one-off transcript fetcher with hardcoded origin-machine paths and personal credential lookups; unreferenced by the research code.

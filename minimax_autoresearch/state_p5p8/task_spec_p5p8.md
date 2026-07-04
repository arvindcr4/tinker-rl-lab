# Task Spec — P5–P8 Improvement Mining (MiniMax M3 Autoresearch)

Mission: raise papers **P5 (minreport), P6 (registry), P7 (zvf_controller),
P8 (fraud)** to the same evidence standard as P1–P4. Full brief (paper map,
fresh-data inventory, target classes T1–T5, deliverable conventions) lives in
the worktree at `P5P8_IMPROVEMENT_BRIEF.md` — the worker reads it every
iteration.

Threads (round-robin): 1=P5 minreport, 2=P6 registry, 3=P7 controller,
4=P8 fraud + cross-paper synthesis.

Success = `P5P8_IMPROVEMENTS.md` ledger rows moving proposed → prototyped →
validated, each backed by a script run on real repo data (N2 tensors, mega
manifests, N10 seeds, zvf_iter TSVs, fraud CSVs) + verified citations, with
validated paper-facing items integrated into the affected `paper_P{5..8}_*.tex`
at **0 build errors / 0 undefined citations** (current state — do not regress).

No new GPU training: analysis of existing/streaming data only. Driver model:
MiniMax M3. Protocol: deli-autoresearch (fresh session per iteration, state in
files, stall → pivot, watchdog).

Getting papers (MCP): `mcp__firecrawl__firecrawl_research_search_papers` /
`_read_paper` / `_related_papers` for arXiv; `brave-search` / `serper` for web.
Fall back to built-in WebSearch/WebFetch. VERIFY every citation before use.

Same guardrails as always: worktree-only writes, no push/gh/secrets/uploads,
files ≤300 lines, zero interaction, commit locally each iteration.

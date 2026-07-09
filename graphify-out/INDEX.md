# graphify-out/ — INDEX

**Purpose:** Output of the `graphify` tool run over the whole repo (2026-04-20): an AST + semantic knowledge graph of code/docs, with an Obsidian-style report.

**Key files:**
- `GRAPH_REPORT.md` — human-readable graph summary: 2924 nodes · 4503 edges · 332 communities over 329 files; community hubs as `[[wikilinks]]` for navigation.
- `graph.json` (~2.9MB) — full node/edge graph data.
- `graph.html` (~2.4MB) — interactive graph visualization.
- `manifest.json` — per-file mtime manifest of everything graphed.
- `cost.json` — token/cost accounting per run (0 tokens; AST-only run).
- `.graphify_python` — path to the graphify pipx venv python.

**Subfolders:**
- `cache/` — per-file extraction cache (~171 hash-named JSONs) (see its INDEX.md).

**Find it fast:**
- to browse the code graph → open `graph.html`
- to query nodes/edges programmatically → `graph.json`
- to read the summary → `GRAPH_REPORT.md`

# graphify-out/cache/ — INDEX

**Purpose:** Per-file extraction cache for the graphify run — do not edit by hand.

**Contents (dump):** ~171 JSON files named `<sha256>.json`, one cache entry per graphed source file. Each holds that file's extracted graph fragment: `{"nodes": [...], "edges": [...], "raw_calls": [...]}` (nodes carry `id/label/file_type/source_file`; edges carry `source/target/relation/confidence`). The filename hash keys the cached extraction so re-runs skip unchanged files. Consume the merged result via `../graph.json`, not these shards.

**Find it fast:**
- to read the assembled graph → `../graph.json` (not individual cache files)

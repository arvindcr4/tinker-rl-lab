"""Read-only lookup of the W&B run inventory for Sem3 NeurIPS claim tracing.

Usage: python3 tools/review_ledger/Sem3_NeurIPS_wandb_lookup.py <wandb_runs.json> <substring> [project]
Prints project, id, name, state, created date and scalar summary fields for runs
whose name or config contains <substring> (case-insensitive).
"""
import json
import sys

runs = json.load(open(sys.argv[1]))
pat = sys.argv[2].lower()
proj = sys.argv[3] if len(sys.argv) > 3 else None
for r in runs:
    blob = (r.get("name", "") + " " + json.dumps(r.get("config", {}))).lower()
    if pat in blob and (proj is None or r["project"] == proj):
        s = {k: v for k, v in r.get("summary", {}).items()
             if not str(k).startswith("_") and isinstance(v, (int, float, str))}
        print(r["project"], r["id"], r["name"], r["state"], r["created"][:10], json.dumps(s)[:300])

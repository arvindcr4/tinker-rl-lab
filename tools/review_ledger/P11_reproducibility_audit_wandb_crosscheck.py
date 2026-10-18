#!/usr/bin/env python3
"""Cross-check E1 seed records against a read-only W&B run export.

Usage: P11_reproducibility_audit_wandb_crosscheck.py <wandb_runs.json>
For each zvf-program/audit/results/full/<arm>-seed-<s>.json, look up its W&B run id
and print state plus any summary keys that look like held-out accuracy / rollouts.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runs = {
    r["id"]: r
    for r in json.loads(Path(sys.argv[1]).read_text())
    if isinstance(r, dict) and "id" in r
}
rows, mism = [], 0
for p in sorted((ROOT / "zvf-program/audit/results/full").glob("*-seed-*.json")):
    rec = json.loads(p.read_text())
    rid = rec["remote_verification"]["wandb"]["run_id"]
    w = runs.get(rid)
    if w is None:
        rows.append((p.name, rid, "MISSING", None))
        continue
    s = w.get("summary") or {}
    cand = {
        k: v
        for k, v in s.items()
        if any(t in k.lower() for t in ("heldout", "held_out", "eval/acc", "rollout"))
    }
    hs = next(
        (
            v
            for k, v in cand.items()
            if "heldout" in k.lower() and isinstance(v, (int, float)) and 0 <= v <= 1
        ),
        None,
    )
    ok = hs is None or abs(hs - rec["heldout_score"]) < 1e-9
    mism += not ok
    rows.append(
        (
            p.name,
            rid,
            w.get("state"),
            {"record": rec["heldout_score"], "wandb": hs, "match": ok, "keys": sorted(cand)[:6]},
        )
    )
for r in rows:
    print(*r)
print("mismatches:", mism, "of", len(rows))

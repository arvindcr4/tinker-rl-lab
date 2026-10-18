#!/usr/bin/env python3
"""Propose review17-v2 CI-method shape migration without inventing provenance.
Original method names are preserved; unknown metadata remains null.
Canonical schema validation precedes publication; original entries are untouched."""

import csv
import json
import pathlib

import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from review17_contract import validate_proposals
from _paths import REPO_ROOT

ROOT = REPO_ROOT
REG = ROOT / "platform_hybrid/registry/entries"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8/review17_v2"


def migrate_method(method):
    return {"method": method, "n_boot": None, "seed": None, "ci_level": None, "source": None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not list(REG.glob("*.json")):
        raise ValueError("Empty registry")
    patch_log = []
    proposed = {}
    for p in sorted(REG.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("record_type") != "variant_delta":
            continue
        rows = d.get("measured") or []
        changed = False
        for i, m in enumerate(rows):
            cm = m.get("ci_method")
            if isinstance(cm, str):
                patch_log.append(
                    {
                        "id": d["id"],
                        "row_idx": i,
                        "metric": m.get("metric", ""),
                        "panel": m.get("panel", ""),
                        "before": cm,
                        "after": json.dumps(migrate_method(cm)),
                    }
                )
                m["ci_method"] = migrate_method(cm)
                changed = True
        if changed:
            proposed[p.stem] = d
    if proposed:
        validate_proposals(proposed, ROOT / "platform_hybrid/registry/schema.json")
    (OUT / "p6_iter134_proposed_entries.json").write_text(
        json.dumps(proposed, indent=2, allow_nan=False) + "\n"
    )
    with (OUT / "p6_iter134_patch_log.tsv").open("w", newline="") as f:
        if patch_log:
            w = csv.DictWriter(f, fieldnames=list(patch_log[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(patch_log)
    print(f"patched {len(patch_log)} ci_method strings to canonical objects")
    for r in patch_log:
        print(f"  {r['id']:>14s}  row{r['row_idx']}  metric={r['metric']:>12s}  panel={r['panel']}")


if __name__ == "__main__":
    main()

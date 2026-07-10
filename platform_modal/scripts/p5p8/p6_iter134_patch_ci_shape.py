#!/usr/bin/env python3
"""P6 iter-134 — patch the 8 ci_method string-shape violations.

The schema's $defs/ci_method is {type: object|null}; 8 measured rows have
`ci_method: "bootstrap_paired_5seed"` (string). Iter-130 patched the
mag_mean stale CI but did not promote the string-typed ci_method to the
canonical object shape. This script:

  - Promotes each string to a full {method, n_boot, seed, ci_level, source}
    object using the iter-130 paired-seed bootstrap provenance.
  - Writes a patch log to platform_hybrid/experiments/results/p5p8/p6_iter134_patch_log.tsv
  - Leaves all other rows untouched.

Stdlib only.
"""
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
REG = ROOT / "registry/entries"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8"

CANON = {
    "method": "bootstrap_paired_5seed",
    "n_boot": 5000,
    "seed": 20260705,
    "ci_level": 0.95,
    "source": "platform_modal/scripts/p5p8/p6_iter130_patch_stale_mag.py",
}


def main():
    patch_log = []
    for p in sorted(REG.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("record_type") != "variant_delta":
            continue
        rows = d.get("measured") or []
        changed = False
        for i, m in enumerate(rows):
            cm = m.get("ci_method")
            if isinstance(cm, str):
                patch_log.append({
                    "id": d["id"],
                    "row_idx": i,
                    "metric": m.get("metric", ""),
                    "panel": m.get("panel", ""),
                    "before": cm,
                    "after": json.dumps(CANON),
                })
                m["ci_method"] = dict(CANON)
                changed = True
        if changed:
            p.write_text(json.dumps(d, indent=2) + "\n")
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
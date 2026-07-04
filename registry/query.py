#!/usr/bin/env python3
"""GRPO-Registry reference CLI: list / query / badge / stackdiff.

Stdlib only. Field convention (see schema.json): JSON null = unreported;
explicit false / 0.0 / "none" = reported-as-absent. The MIN-REPORT badge
scores reporting coverage, not configuration virtue.

Usage:
  python3 registry/query.py list
  python3 registry/query.py query --item reference_kl [--full]
  python3 registry/query.py badge [entry_id]
  python3 registry/query.py stackdiff <entry_id_a> <entry_id_b>
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]
# stackdiff flip-risk ladder (R0 lowest, R5 highest)
RISK = {0: "R0 identical manifests",
        1: "R1 nuisance-only differences (metadata, logging)",
        2: "R2 runtime differences (backend, precision, placement)",
        3: "R3 loss-form or KL-handling differences",
        4: "R4 variant-delta set differs (a claimed technique is absent or surrogate on one side)",
        5: "R5 unauditable: a differing item is unreported (null) on at least one side"}


def load(kind=None):
    recs = {}
    for p in sorted((HERE / "entries").glob("*.json")):
        r = json.loads(p.read_text())
        if kind is None or r["record_type"] == kind:
            recs[r["id"]] = r
    return recs


def item_score(d):
    """Fraction of leaf fields in one MIN-REPORT item that are non-null."""
    leaves = list(d.values())
    return sum(v is not None for v in leaves) / len(leaves) if leaves else 0.0


def badge(rec):
    scores = {it: item_score(rec["min_report"][it]) for it in ITEMS}
    return round(100 * sum(scores.values()) / len(ITEMS)), scores


def cmd_list(_):
    for r in load("stack").values():
        print(f"{r['id']:40s} label={r['label_claimed']:14s} "
              f"framework={r['framework']['name']}/{r['framework']['openness']}")


def cmd_badge(args):
    recs = load("stack")
    if args.entry:
        recs = {args.entry: recs[args.entry]}
    for rid, r in recs.items():
        total, scores = badge(r)
        detail = " ".join(f"{k}={v:.2f}" for k, v in scores.items())
        print(f"{rid:40s} badge={total:3d}  {detail}")


def cmd_query(args):
    """Which stacks fully report a MIN-REPORT item? (e.g. --item reference_kl)"""
    for rid, r in load("stack").items():
        s = item_score(r["min_report"][args.item])
        status = "FULL" if s == 1.0 else ("PARTIAL" if s > 0 else "UNREPORTED")
        line = f"{rid:40s} {args.item}={status} ({s:.2f})"
        if args.full and status != "UNREPORTED":
            line += "  " + json.dumps(r["min_report"][args.item])
        print(line)


def diff_leaves(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_leaves(va, vb, prefix + k + ".")
        elif va != vb:
            out.append((prefix + k, va, vb))
    return out


def cmd_stackdiff(args):
    recs = load("stack")
    a, b = recs[args.a], recs[args.b]
    level = 0
    print(f"stackdiff {a['id']} vs {b['id']}")
    if a["label_claimed"] == b["label_claimed"]:
        print(f"  shared claimed label: '{a['label_claimed']}'")
    # runtime / backend
    for f, va, vb in diff_leaves(a["min_report"]["sampler_backend"],
                                 b["min_report"]["sampler_backend"]):
        level = max(level, 2)
        print(f"  [R2] sampler_backend.{f}: {va!r} -> {vb!r}")
    # loss form + KL
    for it in ("loss_form", "reference_kl"):
        for f, va, vb in diff_leaves(a["min_report"][it], b["min_report"][it]):
            lv = 5 if (va is None or vb is None) else 3
            level = max(level, lv)
            print(f"  [R{lv}] {it}.{f}: {va!r} -> {vb!r}")
    # variant-delta sets
    da = {(d["delta_id"], d["component"]): d["status"]
          for d in a.get("variant_deltas_applied", [])}
    db = {(d["delta_id"], d["component"]): d["status"]
          for d in b.get("variant_deltas_applied", [])}
    for key in sorted(set(da) | set(db)):
        sa, sb = da.get(key, "absent"), db.get(key, "absent")
        if sa != sb:
            lv = 5 if "unknown" in (sa, sb) else 4
            level = max(level, lv)
            print(f"  [R{lv}] delta {key[0]}:{key[1]}: {sa} -> {sb}")
    print(f"verdict: {RISK[level]}")
    if level >= 4:
        print("verdict: comparisons across this pair are at risk of a LABEL FLIP;"
              "\n         do not attribute outcome differences to the shared label.")
    return level


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pb = sub.add_parser("badge")
    pb.add_argument("entry", nargs="?", default=None)
    pq = sub.add_parser("query")
    pq.add_argument("--item", choices=ITEMS, required=True)
    pq.add_argument("--full", action="store_true")
    pd = sub.add_parser("stackdiff")
    pd.add_argument("a")
    pd.add_argument("b")
    args = ap.parse_args()
    {"list": cmd_list, "badge": cmd_badge,
     "query": cmd_query, "stackdiff": cmd_stackdiff}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

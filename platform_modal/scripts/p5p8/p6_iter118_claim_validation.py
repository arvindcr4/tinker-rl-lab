#!/usr/bin/env python3
"""P6 iter-118 claim-vs-measurement cross-reference.

For each ``delta_*.json`` record that has at least one
``expected_effects`` entry, join it against every corresponding
``measured[*]`` row matching the (metric, panel) key. Produce a single
TSV row per (delta, expected_effect, measured) join with a verdict:
    SUPPORTS     -- sign matches predicted, CI excludes 0
    CONTRADICTS  -- sign contradicts predicted, CI excludes 0
    NEUTRAL      -- sign matches predicted, CI contains 0 (or sign
                    matches but CI exclusions are unclear)
    NEUTRAL_MISMATCH -- sign contradicts predicted, CI contains 0
    UNCLAIMED    -- no measured entry for this (metric, panel) key
    MISSING      -- expected_effect has no measured and no panel

Sign convention: the registry uses (variant - baseline) for
``measured[*].delta`` (e.g. aero - grpo = -0.025 means aero is 0.025
BELOW grpo). This script adopts that convention and emits
``predicted_sign`` as parsed from the entry verbatim.

Output: platform_hybrid/experiments/results/p5p8/p6_iter118_claim_validation.tsv

The script also prints a summary table: per delta, total SUPPORTS /
CONTRADICTS / NEUTRAL / UNCLAIMED counts.
"""
import argparse
import csv
import json
import pathlib
import sys

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
REGISTRY = WORKTREE / "registry"
RESULTS = WORKTREE / "experiments" / "results" / "p5p8"

PRED_MAP = {
    ">0": lambda x: x > 0,
    "<0": lambda x: x < 0,
    ">=0": lambda x: x >= 0,
    "<=0": lambda x: x <= 0,
    "==0": lambda x: x == 0,
}


def parse_pred(pred):
    """Map a predicted-sign string to a function, or None when unparsed."""
    if pred is None:
        return None
    return PRED_MAP.get(pred)


def verdict(sign_matches, ci_excludes_0):
    """Map (sign_matches, ci_excludes_0) to a verdict tag.

    The four outcomes follow the standard registry convention used in
    delta_*.json ``claim_validation`` blocks.
    """
    if ci_excludes_0 and sign_matches:
        return "SUPPORTS"
    if ci_excludes_0 and not sign_matches:
        return "CONTRADICTS"
    if not ci_excludes_0 and sign_matches:
        return "NEUTRAL"
    return "NEUTRAL_MISMATCH"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {}

    def ensure(did):
        s = summary.setdefault(did, {})
        for k in ("SUPPORTS", "CONTRADICTS", "NEUTRAL",
                  "NEUTRAL_MISMATCH", "UNCLAIMED", "total"):
            s.setdefault(k, 0)
        return s

    deltas = sorted(REGISTRY.glob("entries/delta_*.json"))
    for p in deltas:
        rec = json.loads(p.read_text())
        delta_id = rec["id"]
        name = rec.get("name", delta_id)
        measured_by_key = {}
        for m in rec.get("measured", []):
            key = (m.get("metric"), m.get("panel"))
            measured_by_key.setdefault(key, []).append(m)
        for ee in rec.get("expected_effects", []):
            metric = ee.get("metric")
            panel = ee.get("panel")
            pred = ee.get("predicted_sign")
            rationale = ee.get("rationale", "")
            matched = measured_by_key.get((metric, panel), [])
            if not matched:
                rows.append({
                    "delta_id": delta_id,
                    "name": name,
                    "metric": metric,
                    "panel": panel,
                    "predicted_sign": pred,
                    "rationale": rationale,
                    "observed_delta": "",
                    "ci_lo": "",
                    "ci_hi": "",
                    "n": "",
                    "significant": "",
                    "verdict": "UNCLAIMED",
                    "note": "no measured entry for this (metric, panel)",
                })
                s = ensure(delta_id)
                s["UNCLAIMED"] += 1
                s["total"] += 1
                continue
            for m in matched:
                delta = m.get("delta")
                lo = m.get("ci_low")
                hi = m.get("ci_high")
                sig = m.get("significant")
                ci_excludes_0 = False
                try:
                    if lo is not None and hi is not None:
                        ci_excludes_0 = not (lo <= 0 <= hi)
                except TypeError:
                    ci_excludes_0 = False
                pred_fn = parse_pred(pred)
                sign_matches = None
                if pred_fn is not None and delta is not None:
                    try:
                        sign_matches = bool(pred_fn(float(delta)))
                    except Exception:
                        sign_matches = None
                tag = ""
                if sign_matches is not None:
                    tag = verdict(sign_matches, ci_excludes_0)
                rows.append({
                    "delta_id": delta_id,
                    "name": name,
                    "metric": metric,
                    "panel": panel,
                    "predicted_sign": pred,
                    "rationale": rationale[:80],
                    "observed_delta": f"{float(delta):+.6f}" if delta is not None else "",
                    "ci_lo": f"{float(lo):+.6f}" if lo is not None else "",
                    "ci_hi": f"{float(hi):+.6f}" if hi is not None else "",
                    "n": m.get("n", ""),
                    "significant": "yes" if sig else ("no" if sig is False else ""),
                    "verdict": tag,
                    "note": m.get("note", "")[:80],
                })
                s = ensure(delta_id)
                s[tag] = s.get(tag, 0) + 1
                s["total"] += 1

    cols = ["delta_id", "name", "metric", "panel", "predicted_sign",
            "rationale", "observed_delta", "ci_lo", "ci_hi", "n",
            "significant", "verdict", "note"]
    if args.write:
        with (RESULTS / "p6_iter118_claim_validation.tsv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # print summary
    print(f"\n# Claim-validation summary across {len(deltas)} delta records")
    print(f"{'delta':18s} {'tot':>4s}  {'SUPPORT':>7s} {'CONTR':>7s} {'NEUT':>5s} {'NEUT_M':>7s} {'UNCL':>5s}")
    grand = {"SUPPORTS": 0, "CONTRADICTS": 0, "NEUTRAL": 0, "NEUTRAL_MISMATCH": 0, "UNCLAIMED": 0}
    for did in sorted(summary):
        s = summary[did]
        for k in grand:
            grand[k] += s.get(k, 0)
        print(f"{did:18s} {s['total']:>4d}  {s.get('SUPPORTS',0):>7d} "
              f"{s.get('CONTRADICTS',0):>7d} {s.get('NEUTRAL',0):>5d} "
              f"{s.get('NEUTRAL_MISMATCH',0):>7d} {s.get('UNCLAIMED',0):>5d}")
    print(f"{'GRAND TOTAL':18s} {sum(grand.values()):>4d}  "
          f"{grand['SUPPORTS']:>7d} {grand['CONTRADICTS']:>7d} "
          f"{grand['NEUTRAL']:>5d} {grand['NEUTRAL_MISMATCH']:>7d} {grand['UNCLAIMED']:>5d}")

    if args.write:
        print(f"\nwrote {RESULTS}/p6_iter118_claim_validation.tsv ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GRPO-Registry reference CLI: list / query / badge / stackdiff / drift.

Stdlib only. Field convention (see schema.json): JSON null = unreported;
explicit false / 0.0 / "none" = reported-as-absent. The MIN-REPORT badge
scores reporting coverage, not configuration virtue.

Usage:
  python3 registry/query.py list
  python3 registry/query.py query --item reference_kl [--full]
  python3 registry/query.py badge [entry_id]
  python3 registry/query.py stackdiff <entry_id_a> <entry_id_b>
  python3 registry/query.py drift  # iter-42: schema-anchored `field:` claims
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


def cmd_implementations(args):
    """Cross-reference: which stacks claim to implement a (delta, component)?

    Usage:
      python3 registry/query.py implementations --delta delta_dapo
      python3 registry/query.py implementations --status implemented
      python3 registry/query.py implementations --status unknown --framework worktree-zvf130-batch
    """
    recs = load("variant_delta") if args.delta else {}
    stacks = load("stack")
    # Build reverse index: (did, comp) -> [(stack_id, status)]
    by_pair = {}
    for sid, s in stacks.items():
        for vd in s.get("variant_deltas_applied", []):
            by_pair.setdefault((vd["delta_id"], vd["component"]), []).append(
                (sid, vd["status"]))
    if args.delta:
        if args.delta not in recs:
            print(f"unknown delta_id: {args.delta}"); return 1
        d = recs[args.delta]
        print(f"delta_id={args.delta} name='{d.get('name')}'")
        for comp in d["deltas"]:
            key = (args.delta, comp["component"])
            claims = by_pair.get(key, [])
            if args.status:
                claims = [(s, st) for s, st in claims if st == args.status]
            if args.framework:
                claims = [(s, st) for s, st in claims
                          if stacks[s]["framework"]["name"] == args.framework]
            print(f"  {comp['component']:30s} {len(claims)} claim(s): "
                  + ", ".join(f"{s}={st}" for s, st in claims) if claims
                  else f"  {comp['component']:30s} 0 claims")
    else:
        # Aggregate: print per-delta, per-component counts
        from collections import Counter
        ctr = Counter()
        for (did, comp), claims in by_pair.items():
            for _, st in claims:
                ctr[(did, comp, st)] += 1
        for (did, comp, st), n in sorted(ctr.items()):
            line = f"{did:18s} {comp:30s} {st:12s} {n}"
            if args.status and st != args.status:
                continue
            print(line)
    return 0


def cmd_drift(args):
    """Iter-42: walk every delta_*.json component's `field:` claim, classify
    it against the actual schema MIN-REPORT surface. Verdict counts and
    per-row table mirror scripts/p5p8/p6_delta_field_drift_audit.py."""
    import importlib.util
    audit = pathlib.Path(__file__).resolve().parents[1] / "scripts/p5p8" / "p6_delta_field_drift_audit.py"
    spec = importlib.util.spec_from_file_location("audit", audit)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schema = json.load(open(HERE / "schema.json"))
    defs = schema.get("$defs", {})

    def deref(node):
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                if node["$ref"].startswith("#/$defs/"):
                    return deref(defs[node["$ref"].split("/")[-1]])
            return {k: deref(v) for k, v in node.items() if k != "$ref"}
        if isinstance(node, list):
            return [deref(x) for x in node]
        return node

    stack_def = deref(schema["$defs"]["stack_record"])
    min_report = stack_def["properties"]["min_report"]["properties"]
    block_leaves = {}
    for item_name, item_def in min_report.items():
        out = set()
        def walk(o):
            if not isinstance(o, dict):
                return
            for k, v in o.items():
                if k == "properties" and isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, dict) and "properties" in vv:
                            walk(vv)
                        elif isinstance(vv, dict):
                            out.add(kk)
        walk(item_def)
        block_leaves[item_name] = out

    counts = {"OK": 0, "SEE_CITATION": 0,
              "BLOCK_NOT_IN_MIN_REPORT": 0,
              "LEAF_NOT_IN_SCHEMA": 0, "AMBIGUOUS_REFERENCE": 0}
    n_rows = 0
    drifts = []
    for p in sorted((HERE / "entries").glob("delta_*.json")):
        d = json.load(open(p))
        for c in d.get("deltas", []):
            n_rows += 1
            field = c.get("field", "")
            comp = c["component"]
            if not field:
                v, r = "EMPTY", "no field declared"
            elif field == "see delta-list and citation":
                v, r = "SEE_CITATION", "deferred to source paper"
            elif "." not in field:
                v, r = "AMBIGUOUS_REFERENCE", f"block-only reference: {field!r}"
            else:
                block, leaf = field.split(".", 1)
                if block not in block_leaves:
                    v, r = "BLOCK_NOT_IN_MIN_REPORT", f"block {block!r} not in MIN-REPORT"
                elif leaf not in block_leaves[block]:
                    v, r = "LEAF_NOT_IN_SCHEMA", f"leaf {leaf!r} missing from MIN-REPORT.{block}"
                else:
                    v, r = "OK", f"{block}.{leaf}"
            counts[v] = counts.get(v, 0) + 1
            if v not in ("OK", "SEE_CITATION"):
                drifts.append((d["id"], comp, field, v, r))
    print(f"delta-schema drift audit (iter-42): {n_rows} (delta, component) pairs")
    drift_n = sum(c for k, c in counts.items() if k not in ("OK", "SEE_CITATION"))
    print(f"  drift rate: {drift_n/max(1,n_rows):.3f}  ({drift_n}/{n_rows})")
    print()
    print(f"  {'verdict':24s} count")
    for k, n in sorted(counts.items()):
        print(f"  {k:24s} {n}")
    if drifts:
        print()
        print(f"  drift rows ({len(drifts)}):")
        for d, c, f, v, r in drifts:
            print(f"    {d:14s} {c:30s} field={f!r:40s} -> {v}")
    return 0


def cmd_claim_validation(args):
    """Iter-46: print the (delta, metric, panel) claim-validation verdict table
    for every variant_delta record. Optional --delta filters to one entry.
    Verdict codes: SUPPORTS / NEUTRAL / CONTRADICTS / UNCLAIMED."""
    from collections import Counter
    counts = Counter()
    for p in sorted((HERE / "entries").glob("delta_*.json")):
        rec = json.load(open(p))
        rows = rec.get("claim_validation", []) or []
        if args.delta and rec["id"] != args.delta:
            continue
        for r in rows:
            counts[r["verdict"]] += 1
            if args.delta:
                print(f"  {rec['id']:18s} {r['metric']:18s} {r['panel']:24s} "
                      f"verdict={r['verdict']:11s} "
                      f"observed={r['observed_delta']:+.4f} "
                      f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
    if not args.delta:
        print(f"=== Iter 46 P6 — claim-validation verdict counts (across all {len(list((HERE / 'entries').glob('delta_*.json')))} delta_*.json) ===")
        for k, n in sorted(counts.items()):
            print(f"  {k:12s} {n}")
    return 0


def cmd_validate(_):
    """Iter-50: CI-style schema validation pass. Returns exit code 0 iff every
    entry parses against registry/schema.json. Stdout is one PASS/FAIL line per
    entry; stderr is empty on full pass."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("FATAL: jsonschema not installed", file=sys.stderr)
        return 2
    schema = json.load(open(HERE / "schema.json"))
    fails = 0
    for p in sorted((HERE / "entries").glob("*.json")):
        rec = json.loads(p.read_text())
        try:
            jsonschema.validate(rec, schema)
            print(f"PASS  {p.name}")
        except jsonschema.ValidationError as e:
            fails += 1
            print(f"FAIL  {p.name}  {str(e.message)[:140]}")
    print(f"--- {len(list((HERE / 'entries').glob('*.json'))) - fails}/{len(list((HERE / 'entries').glob('*.json')))} pass ---")
    return 1 if fails else 0


def cmd_health(_):
    """Iter-50: shell out to scripts/p5p8/p6_registry_health.py for the
    full coverage + null-rate + verdict-signature audit. Passes the exit
    code through unchanged so CI sees schema failures."""
    import subprocess
    target = (HERE / ".." / "scripts" / "p5p8" / "p6_registry_health.py").resolve()
    rc = subprocess.run([sys.executable, str(target)]).returncode
    return rc


def cmd_measured_coverage(args):
    """Iter-58: print the measured-block provenance & coverage audit.

    For every variant_delta record, audit the measured / expected_effects /
    claim_validation block presence, source path resolution on disk, and the
    cross-panel sign agreement (N2 zvf vs ZVF130 risk for entries with both).
    Exit 0 always — this is a reporting audit, not a gate."""
    audit_path = HERE / "measured_block_audit.json"
    if not audit_path.exists():
        # Self-rerun if no cache: shell out to the script.
        import subprocess
        target = (HERE / ".." / "scripts" / "p5p8" / "p6_measured_coverage.py").resolve()
        subprocess.run([sys.executable, str(target)], check=True)
    audit = json.load(open(audit_path))
    if args.delta:
        rows = [r for r in audit["per_entry"] if r["delta_id"] == args.delta]
        if not rows:
            print(f"no such delta_id: {args.delta}", file=sys.stderr)
            return 1
        for r in rows:
            print(f"=== {r['delta_id']}  ({r['name']}) ===")
            for k, v in r.items():
                if k in ("delta_id", "name"):
                    continue
                print(f"  {k:24s} {v}")
        # Print cross-panel agreement if available
        for c in audit.get("cross_panel_agreement", []):
            if c["delta_id"] == args.delta:
                print(f"  cross_panel_agreement:")
                for k, v in c.items():
                    if k == "delta_id":
                        continue
                    print(f"    {k:18s} {v}")
        return 0
    headline = {
        "n_deltas": audit["n_entries"],
        "with_measured": sum(1 for r in audit["per_entry"]
                             if r["measured_count"] > 0),
        "empty_measured": sum(1 for r in audit["per_entry"]
                              if r["measured_count"] == 0),
        "verdict_totals": audit["verdict_totals"],
        "n_cross_panel_pairs": len(audit.get("cross_panel_agreement", [])),
        "n_cross_panel_same_sign": sum(
            1 for r in audit.get("cross_panel_agreement", [])
            if r["same_sign"]),
        "n_missing_sources": audit.get("n_mismatch_audit", 0),
    }
    print(f"=== Iter 58 P6 — measured-block coverage audit ({headline['n_deltas']} deltas) ===")
    for k, v in headline.items():
        print(f"  {k:24s} {v}")
    print("--- per-entry ---")
    for r in audit["per_entry"]:
        print(f"  {r['delta_id']:22s} m={r['measured_count']:2d} sig={r['n_significant']:2d} "
              f"S={r['supports']} C={r['contradicts']} N={r['neutral']} U={r['unclaimed']}")
    if audit.get("cross_panel_agreement"):
        print("--- cross-panel (N2 zvf  vs  ZVF130 risk) ---")
        for c in audit["cross_panel_agreement"]:
            flag = "✓" if c["same_sign"] else "✗"
            print(f"  {flag} {c['delta_id']:14s}  N2 zvf={c['n2_zvf_delta']:+.4f}  "
                  f"Z130 risk={c['zvf130_risk_delta']:+.4f}")
    return 0


def cmd_coverage(args):
    """Iter-62: print the outcomes.coverage self-report block per stack entry.

    Reads the audit TSV cached by scripts/p5p8/p6_outcomes_coverage_block.py
    if present, otherwise shells out to regenerate it. Exit 0 always.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_outcomes_coverage_audit.tsv"
    audit_path = audit_path.resolve()
    if not audit_path.exists():
        import subprocess
        target = (HERE / ".." / "scripts" / "p5p8" / "p6_outcomes_coverage_block.py").resolve()
        subprocess.run([sys.executable, str(target)], check=True)
    # parse the TSV
    with audit_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if args.entry:
        rows = [r for r in rows if r[idx["entry_id"]] == args.entry]
        if not rows:
            print(f"no such entry_id: {args.entry}", file=sys.stderr)
            return 1
    # headline
    n_total = len(rows)
    n_stack = sum(1 for r in rows if r[idx["record_type"]] == "stack")
    n_with_cov_block = 0
    recs = load()
    for r in rows:
        eid = r[idx["entry_id"]]
        if eid not in recs:
            continue
        cov = recs[eid].get("outcomes", {})
        if isinstance(cov, dict) and "coverage" in cov:
            n_with_cov_block += 1
    print(f"=== Iter 62 P6 — outcomes.coverage self-report ({n_total} entries) ===")
    print(f"  n_with_coverage_block: {n_with_cov_block}/{n_total}")
    print(f"  n_stack: {n_stack}, n_delta: {n_total - n_stack}")
    print("--- per entry ---")
    if args.entry:
        # full dict for one entry
        eid = args.entry
        rec = recs.get(eid)
        if rec:
            out = rec.get("outcomes", {}) or {}
            cov = out.get("coverage") or {}
            print(f"  {eid}")
            for k, v in cov.items():
                print(f"    {k:30s} {v}")
        return 0
    for r in rows:
        eid = r[idx["entry_id"]]
        mr = float(r[idx["min_report_coverage"]])
        dd = float(r[idx["declared_deltas_coverage"]])
        me = float(r[idx["measured_coverage"]])
        ci = r[idx["ci_method_present"]]
        rt = r[idx["record_type"]]
        print(f"  {eid:38s} {rt:14s} min={mr:.3f} decl={dd:.3f} meas={me:.3f} ci={ci}")
    return 0


def cmd_antiherding(args):
    """Iter-66: print outcomes.zvf_antiherding per stack + measured_yield_residual per delta.

    Reads experiments/results/p5p8/p6_zvf_antiherding_summary.tsv (regenerated
    by scripts/p5p8/p6_zvf_antiherding.py if missing). Exit 0 always.
    """
    audit_path = HERE / ".." / "experiments" / "results" / "p5p8" / "p6_zvf_antiherding_summary.tsv"
    audit_path = audit_path.resolve()
    if not audit_path.exists():
        import subprocess
        target = (HERE / ".." / "scripts" / "p5p8" / "p6_zvf_antiherding.py").resolve()
        subprocess.run([sys.executable, str(target)], check=True)
    with audit_path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        rows = [line.rstrip("\n").split("\t") for line in fh]
    if args.method:
        rows = [r for r in rows if r[idx["method"]] == args.method]
        if not rows:
            print(f"no such method: {args.method}", file=sys.stderr)
            return 1
    print(f"  {'method':8s} {'G':>3s} {'zvf_obs':>8s} {'zvf_iid':>8s} {'delta_div':>9s} {'Y_obs':>7s}  vs_grpo [lo,hi]  sig  p")
    for r in rows:
        m = r[idx["method"]]
        G = r[idx["G"]]
        zo = float(r[idx["zvf_obs_mean"]])
        zi = float(r[idx["zvf_iid_mean"]])
        dd = float(r[idx["delta_div_mean"]])
        yo = float(r[idx["y_obs_mean"]])
        dv = float(r[idx["delta_div_vs_grpo"]])
        lo = float(r[idx["ci_low"]])
        hi = float(r[idx["ci_high"]])
        sig = r[idx["significant"]]
        p = float(r[idx["p_two_sided"]])
        print(f"  {m:8s} {G:>3s} {zo:>8.4f} {zi:>8.4f} {dd:>9.4f} {yo:>7.4f}  "
              f"{dv:+.4f} [{lo:+.4f},{hi:+.4f}]  {sig[:1]}   {p:.3f}")
    return 0


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
    pi = sub.add_parser("implementations")
    pi.add_argument("--delta", default=None,
                    help="Show stacks claiming this delta_id (e.g. delta_dapo).")
    pi.add_argument("--status", choices=("implemented", "surrogate", "absent",
                                          "unknown"), default=None)
    pi.add_argument("--framework", default=None,
                    help="Filter to stacks whose framework.name == X.")
    sub.add_parser("drift")
    pcv = sub.add_parser("claim-validation",
                          help="Iter-46: print (delta, metric, panel) claim-validation verdicts.")
    pcv.add_argument("--delta", default=None,
                     help="Filter to one delta_id (e.g. delta_aero).")
    sub.add_parser("validate",
                   help="Iter-50: CI-style schema validation pass")
    sub.add_parser("health",
                   help="Iter-50: full registry health audit")
    pmc = sub.add_parser("measured-coverage",
                         help="Iter-58: measured-block provenance & coverage audit.")
    pmc.add_argument("--delta", default=None,
                     help="Filter to one delta_id (e.g. delta_aero).")
    pcov = sub.add_parser("coverage",
                          help="Iter-62: print outcomes.coverage self-report block per entry.")
    pcov.add_argument("--entry", default=None,
                      help="Filter to one entry_id.")
    pah = sub.add_parser("antiherding",
                         help="Iter-66: print Contrastive Yield / anti-herding residual per N2 method.")
    pah.add_argument("--method", default=None,
                     help="Filter to one method (grpo, aero, areal, gift).")
    args = ap.parse_args()
    rc = {"list": cmd_list, "badge": cmd_badge,
          "query": cmd_query, "stackdiff": cmd_stackdiff,
          "implementations": cmd_implementations, "drift": cmd_drift,
          "claim-validation": cmd_claim_validation,
          "validate": cmd_validate, "health": cmd_health,
          "measured-coverage": cmd_measured_coverage,
          "coverage": cmd_coverage,
          "antiherding": cmd_antiherding}[args.cmd](args)
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main() or 0)

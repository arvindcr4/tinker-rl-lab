#!/usr/bin/env python3
"""P6 (Pillar 2 — GRPO-Registry) cross-reference integrity + coverage audit (iter 102).

The zvf_iter130 risk-index is the ground truth for the 9 real 5-seed GRPO-family
methods.  The registry stores each method's risk number in TWO independent
representations:
  (A) the zvf130_<method> STACK entry   -> outcomes.{zvf_risk_mean, delta_vs_grpo_*}
  (B) the delta_<method> VARIANT entry  -> measured[] block (panel=zvf130_5seed)
Until now nothing checked (A) against the TSV, (B) against the TSV, or (A)==(B).
This script is that CI-style regression guard.  It also audits COVERAGE: which of
the 9 real methods actually have a stack entry.

Outputs (platform_hybrid/experiments/results/p5p8/):
  p6_crossref_integrity.tsv     one row per (method, check)
  p6_crossref_summary.json      pass/fail counts + coverage gap list
No third-party deps (stdlib only).
"""
import json, glob, os, math, csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV = os.path.join(ROOT, "platform_hybrid/experiments/results/zvf_iter130_method_risk.tsv")
ENTRIES = os.path.join(ROOT, "registry/entries")
OUTDIR = os.path.join(ROOT, "platform_hybrid/experiments/results/p5p8")
os.makedirs(OUTDIR, exist_ok=True)
TOL = 5e-4  # tolerance for stored-vs-recomputed floats

# The 9 methods with real 5-seed statistics (n_seeds==5). Placeholder rows
# (scaling_law_*, tool_use_* with n_seeds==1) are excluded from the panel.
BASE = "grpo"


def load_tsv():
    d = {}
    with open(TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if not row.get("n_seeds") or int(float(row["n_seeds"])) != 5:
                continue
            d[row["method"]] = {
                "zvf_risk_mean": float(row["zvf_risk_mean"]),
                "zvf_risk_sd": float(row["zvf_risk_sd"]),
                "mag_mean": float(row["mag_mean"]),
                "csd_mean": float(row["csd_mean"]),
                "drift_mean": float(row["drift_mean"]),
                "failure_rate": float(row["failure_rate"]),
                "n": 5,
            }
    return d


def welch(m1, s1, n1, m0, s0, n0):
    """Welch two-sample t on (method - base). Returns delta, ci_lo, ci_hi, sig."""
    diff = m1 - m0
    se = math.sqrt(s1 * s1 / n1 + s0 * s0 / n0)
    if se == 0:
        return diff, diff, diff, False, None
    num = (s1 * s1 / n1 + s0 * s0 / n0) ** 2
    den = (s1 * s1 / n1) ** 2 / (n1 - 1) + (s0 * s0 / n0) ** 2 / (n0 - 1)
    df = num / den if den > 0 else (n1 + n0 - 2)
    # two-sided 95% critical t by df (small table + interpolation on common dfs)
    tcrit = t_crit_95(df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    tstat = diff / se
    sig = not (lo <= 0 <= hi)
    return diff, lo, hi, sig, tstat


def t_crit_95(df):
    tbl = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
           20: 2.086, 30: 2.042, 1e9: 1.960}
    keys = sorted(tbl)
    if df <= keys[0]:
        return tbl[keys[0]]
    for a, b in zip(keys, keys[1:]):
        if a <= df <= b:
            fr = (df - a) / (b - a)
            return tbl[a] + fr * (tbl[b] - tbl[a])
    return tbl[keys[-1]]


def load_entries():
    stacks, deltas = {}, {}
    for f in glob.glob(os.path.join(ENTRIES, "*.json")):
        d = json.load(open(f))
        if d.get("record_type") == "stack" and os.path.basename(f).startswith("zvf130_"):
            stacks[d["label_claimed"]] = (os.path.basename(f), d)
        elif d.get("record_type") == "variant_delta":
            nm = d.get("id", "").replace("delta_", "")
            deltas[nm] = (os.path.basename(f), d)
    return stacks, deltas


def delta_z130(dentry):
    for m in dentry.get("measured", []) or []:
        if m.get("panel") == "zvf130_5seed" and m.get("metric") == "zvf_risk_mean":
            return m
    return None


def approx(a, b, tol=TOL):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def main():
    tsv = load_tsv()
    stacks, deltas = load_entries()
    methods = sorted(tsv)
    base = tsv[BASE]
    rows = []
    checks = {"pass": 0, "fail": 0, "sig_diverge": 0}

    def rec(method, check, status, detail):
        rows.append({"method": method, "check": check, "status": status, "detail": detail})
        if status == "PASS":
            checks["pass"] += 1
        elif status == "SIG_DIVERGE":
            checks["sig_diverge"] += 1
        else:
            checks["fail"] += 1

    coverage = {"stack_present": [], "stack_missing": [], "delta_present": [], "delta_missing": []}
    for mth in methods:
        gt = tsv[mth]
        # recomputed Welch delta vs grpo from TSV (ground truth)
        if mth == BASE:
            g_delta, g_lo, g_hi, g_sig = 0.0, 0.0, 0.0, False
        else:
            g_delta, g_lo, g_hi, g_sig, _ = welch(
                gt["zvf_risk_mean"], gt["zvf_risk_sd"], gt["n"],
                base["zvf_risk_mean"], base["zvf_risk_sd"], base["n"])

        # ---- STACK-entry checks ----
        if mth in stacks:
            coverage["stack_present"].append(mth)
            _, se = stacks[mth]
            oc = se.get("outcomes", {}) or {}
            rec(mth, "stack.zvf_risk_mean_vs_tsv",
                "PASS" if approx(oc.get("zvf_risk_mean"), gt["zvf_risk_mean"]) else "FAIL",
                f"stored={oc.get('zvf_risk_mean')} tsv={round(gt['zvf_risk_mean'],6)}")
            rec(mth, "stack.delta_vs_grpo_vs_recompute",
                "PASS" if approx(oc.get("delta_vs_grpo_mean"), g_delta) else "FAIL",
                f"stored={oc.get('delta_vs_grpo_mean')} recomputed={round(g_delta,6)}")
            if mth != BASE:
                # stored sig comes from paired Gaussian-residual bootstrap (iter90);
                # g_sig here is the CONSERVATIVE Welch two-sample verdict.  A divergence
                # is NOT a data bug -- it is a CI-method robustness signal, recorded as
                # sig_robust = (bootstrap_sig AND welch_sig).
                status = "PASS" if oc.get("delta_vs_grpo_sig") == g_sig else "SIG_DIVERGE"
                rec(mth, "stack.sig_robust_bootstrap_vs_welch", status,
                    f"bootstrap_sig={oc.get('delta_vs_grpo_sig')} welch_sig={g_sig}")
        else:
            coverage["stack_missing"].append(mth)
            rec(mth, "stack.coverage", "FAIL", "no zvf130_ stack entry for this real 5-seed method")

        # ---- DELTA-entry checks (base has no self-delta) ----
        if mth == BASE:
            continue
        dm = delta_z130(deltas[mth][1]) if mth in deltas else None
        if dm is not None:
            coverage["delta_present"].append(mth)
            rec(mth, "delta.z130_vs_recompute",
                "PASS" if approx(dm.get("delta"), g_delta) else "FAIL",
                f"stored={dm.get('delta')} recomputed={round(g_delta,6)}")
            # cross-representation: delta entry vs stack entry (if both exist)
            if mth in stacks:
                oc = stacks[mth][1].get("outcomes", {}) or {}
                rec(mth, "crossref.delta_eq_stack",
                    "PASS" if approx(dm.get("delta"), oc.get("delta_vs_grpo_mean")) else "FAIL",
                    f"delta_entry={dm.get('delta')} stack_entry={oc.get('delta_vs_grpo_mean')}")
        else:
            coverage["delta_missing"].append(mth)
            rec(mth, "delta.coverage", "FAIL", "no zvf130_5seed measured block in delta entry")

    # write tsv
    tpath = os.path.join(OUTDIR, "p6_crossref_integrity.tsv")
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "check", "status", "detail"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_real_methods": len(methods),
        "methods": methods,
        "checks_pass": checks["pass"],
        "checks_fail": checks["fail"],
        "stack_coverage": f"{len(coverage['stack_present'])}/{len(methods)}",
        "stack_present": coverage["stack_present"],
"stack_missing": coverage["stack_missing"],
        "delta_coverage": f"{len(coverage['delta_present'])}/{len(methods)-1}",
        "delta_missing": coverage["delta_missing"],
        "ground_truth": TSV,
        "sig_robustness": {
            m: {
                "delta_vs_grpo": round(welch(tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                                             base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[0], 6),
                "welch_lo": round(welch(tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                                        base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[1], 6),
                "welch_hi": round(welch(tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                                        base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[2], 6),
                "welch_sig": welch(tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                                   base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[3],
                "bootstrap_sig": bool((delta_z130(deltas[m][1]) or {}).get("significant"))
                if m in deltas else None,
            } for m in methods if m != BASE
        },
        "welch_recompute": {
            m: {
                "zvf_risk_mean": round(tsv[m]["zvf_risk_mean"], 6),
                "delta_vs_grpo": 0.0 if m == BASE else round(
                    welch(tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                          base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[0], 6),
                "sig": False if m == BASE else welch(
                    tsv[m]["zvf_risk_mean"], tsv[m]["zvf_risk_sd"], 5,
                    base["zvf_risk_mean"], base["zvf_risk_sd"], 5)[3],
            } for m in methods
        },
    }
    spath = os.path.join(OUTDIR, "p6_crossref_summary.json")
    json.dump(summary, open(spath, "w"), indent=2)
    print(f"[integrity] {checks['pass']} PASS / {checks['fail']} FAIL / {checks['sig_diverge']} SIG_DIVERGE")
    print(f"[coverage ] stack {summary['stack_coverage']}  missing={coverage['stack_missing']}")
    print(f"[coverage ] delta {summary['delta_coverage']}  missing={coverage['delta_missing']}")
    sr = summary["sig_robustness"]
    flips = [m for m, v in sr.items() if v["bootstrap_sig"] is not None and v["welch_sig"] != v["bootstrap_sig"]]
    robust = [m for m, v in sr.items() if v["welch_sig"] and v["bootstrap_sig"]]
    print(f"[sig-robust] {len(robust)}/{len(sr)} sig under BOTH bootstrap+Welch: {robust}")
    print(f"[sig-flip ] bootstrap-sig but Welch-NS: {flips}")
    print(f"[wrote] {tpath}\n[wrote] {spath}")
    hard = [r for r in rows if r["status"] == "FAIL" and "coverage" not in r["check"]]
    if hard:
        print("HARD INTEGRITY FAILURES (point-estimate mismatch vs ground truth):")
        for r in hard:
            print("  ", r["method"], r["check"], r["detail"])
    else:
        print("No hard integrity failures: all stored point estimates match ground truth.")

    # write sig-robustness table (one row per non-base method)
    srpath = os.path.join(OUTDIR, "p6_sig_robustness.tsv")
    with open(srpath, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "delta_vs_grpo", "welch_lo", "welch_hi",
                    "welch_sig", "bootstrap_sig", "sig_robust"])
        for m, v in sr.items():
            robust = bool(v["welch_sig"]) and bool(v["bootstrap_sig"])
            w.writerow([m, v["delta_vs_grpo"], v["welch_lo"], v["welch_hi"],
                        v["welch_sig"], v["bootstrap_sig"], robust])
    print(f"[wrote] {srpath}")

    # CI-style exit code: nonzero only on HARD (point-estimate) failures
    import sys
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()

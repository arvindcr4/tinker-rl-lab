#!/usr/bin/env python3
"""
P6 (Pillar 2) — iter174 — per-(entry, panel) metric-coverage audit
stratified by evidence tier (A/B/D from iter126).

Fresh vein (brief vein (a) at the tier-stratified metric-coverage layer):
  no prior P6 iter (134/138/142/146/150/154/158/162/166) has reported
  a tier-stratified SIGNIFICANCE-RATE and PANEL-COVERAGE comparison.
Iter-126 tier-classified; iter-134 audited per-row field completeness;
iter-158 audited 4-tuple join coverage; iter-166 audited provenance.
None stratified measured-evidence DEPLOYMENT by tier.

Hypotheses (falsifiable, set BEFORE measurement):
  H1: tier-A entries (aero/gift/areal) have HIGHER sig_rate (n_sig/n_total)
      than tier-B entries (cppo/drgrpo/es/mcgrpo/ngrpo/scafgrpo/adaptiveg).
  H2: tier-A entries have ≥2 distinct panels each; tier-B ≤1 each.
  H3: on n2_same_stack_last10, reward_mean has HIGHER sig_rate than zvf
      (reward_mean is the headline outcome; zvf is a derived diagnostic).
  H4: tier-A entries have NARROWER mean CI width than tier-B entries.
  H5: tier-D entries (label-only) represent a measurable fraction of
      entries; this iter quantifies that fraction.

Inputs:
  registry/entries/delta_*.json  (17 entries, all variant_delta records)

Outputs:
  experiments/results/p5p8/p6_iter174_per_entry.tsv       (17 rows)
  experiments/results/p5p8/p6_iter174_per_entry_panel.tsv (~26 rows)
  experiments/results/p5p8/p6_iter174_per_metric.tsv      (10 metrics)
  experiments/results/p5p8/p6_iter174_tier_summary.tsv    (3 tier rows)
  experiments/results/p5p8/p6_iter174_summary.json        (H1-H5 verdicts)

Stdlib only. Deterministic. Re-runnable.
"""
import json
import glob
import math
import os
import sys
import time
from collections import defaultdict

REG = "registry/entries/delta_*.json"
OUT_DIR = "experiments/results/p5p8"

# iter-126 tier rule (from ledger row 139): A=n_sig≥3 AND n_panels≥2;
# B=n_sig≥1; D=n_total=0.  We re-derive from current data; rule unchanged.


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI on a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify_tier(n_sig, n_panels, n_total):
    """Mirror iter-126 rule verbatim."""
    if n_total == 0:
        return "D"
    if n_sig >= 3 and n_panels >= 2:
        return "A"
    if n_sig >= 1:
        return "B"
    return "C"  # n_total>=1 AND n_sig=0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Pass 1: load all delta_*.json entries, walk measured[] arrays.
    # ------------------------------------------------------------------
    entries = []
    for path in sorted(glob.glob(REG)):
        with open(path) as fh:
            ej = json.load(fh)
        eid = ej.get("id", os.path.basename(path).replace(".json", ""))
        ename = ej.get("name", "?")
        rows = ej.get("measured", []) or []
        # Per-row aggregates
        per_row = []
        for r in rows:
            d = r.get("delta")
            cl = r.get("ci_low")
            ch = r.get("ci_high")
            sig = r.get("significant")
            mt = r.get("metric", "?")
            pn = r.get("panel", "?")
            if d is None or cl is None or ch is None:
                continue  # skip rows with missing numerics
            per_row.append({
                "metric": mt, "panel": pn, "delta": d,
                "ci_low": cl, "ci_high": ch,
                "ci_width": abs(ch - cl),
                "abs_delta": abs(d),
                "significant": bool(sig),
            })
        n_sig = sum(1 for r in per_row if r["significant"])
        panels = sorted({r["panel"] for r in per_row})
        metrics = sorted({r["metric"] for r in per_row})
        n_total = len(per_row)
        tier = classify_tier(n_sig, len(panels), n_total)
        entries.append({
            "id": eid, "name": ename, "tier": tier,
            "n_total": n_total, "n_sig": n_sig,
            "n_panels": len(panels), "n_metrics": len(metrics),
            "panels": panels, "metrics": metrics,
            "rows": per_row,
        })

    # ------------------------------------------------------------------
    # Output 1: per-entry summary
    # ------------------------------------------------------------------
    out_entry = os.path.join(OUT_DIR, "p6_iter174_per_entry.tsv")
    with open(out_entry, "w") as fh:
        fh.write("entry_id\tname\ttier\tn_total\tn_sig\tn_panels\t"
                 "n_metrics\tsig_rate\tmean_abs_delta\tmean_ci_width\t"
                 "panels\tmetrics\n")
        for e in entries:
            sig_rate = e["n_sig"] / e["n_total"] if e["n_total"] > 0 else 0.0
            mean_abs = (sum(r["abs_delta"] for r in e["rows"]) / e["n_total"]
                        if e["n_total"] > 0 else 0.0)
            mean_ciw = (sum(r["ci_width"] for r in e["rows"]) / e["n_total"]
                        if e["n_total"] > 0 else 0.0)
            fh.write(f'{e["id"]}\t{e["name"]}\t{e["tier"]}\t'
                     f'{e["n_total"]}\t{e["n_sig"]}\t{e["n_panels"]}\t'
                     f'{e["n_metrics"]}\t{sig_rate:.4f}\t{mean_abs:.6f}\t'
                     f'{mean_ciw:.6f}\t{";".join(e["panels"])}\t'
                     f'{";".join(e["metrics"])}\n')

    # ------------------------------------------------------------------
    # Output 2: per-(entry, panel) summary
    # ------------------------------------------------------------------
    out_ep = os.path.join(OUT_DIR, "p6_iter174_per_entry_panel.tsv")
    with open(out_ep, "w") as fh:
        fh.write("entry_id\tname\ttier\tpanel\tn_metrics\tn_sig\t"
                 "sig_rate\tmean_abs_delta\tmean_ci_width\tmetrics_list\n")
        ep_count = 0
        for e in entries:
            by_panel = defaultdict(list)
            for r in e["rows"]:
                by_panel[r["panel"]].append(r)
            for pn, rows in sorted(by_panel.items()):
                ep_count += 1
                n_met = len(rows)
                n_sig_p = sum(1 for r in rows if r["significant"])
                srate = n_sig_p / n_met if n_met > 0 else 0.0
                m_abs = sum(r["abs_delta"] for r in rows) / n_met
                m_ciw = sum(r["ci_width"] for r in rows) / n_met
                mets = sorted({r["metric"] for r in rows})
                fh.write(f'{e["id"]}\t{e["name"]}\t{e["tier"]}\t{pn}\t'
                         f'{n_met}\t{n_sig_p}\t{srate:.4f}\t{m_abs:.6f}\t'
                         f'{m_ciw:.6f}\t{";".join(mets)}\n')

    # ------------------------------------------------------------------
    # Output 3: per-metric summary (across all entries)
    # ------------------------------------------------------------------
    out_metric = os.path.join(OUT_DIR, "p6_iter174_per_metric.tsv")
    metric_agg = defaultdict(lambda: {"n_total": 0, "n_sig": 0,
                                      "abs_delta_sum": 0.0,
                                      "ci_width_sum": 0.0,
                                      "panels": set(), "entries": set()})
    for e in entries:
        for r in e["rows"]:
            a = metric_agg[r["metric"]]
            a["n_total"] += 1
            a["n_sig"] += 1 if r["significant"] else 0
            a["abs_delta_sum"] += r["abs_delta"]
            a["ci_width_sum"] += r["ci_width"]
            a["panels"].add(r["panel"])
            a["entries"].add(e["id"])
    with open(out_metric, "w") as fh:
        fh.write("metric\tn_total\tn_sig\tsig_rate\tmean_abs_delta\t"
                 "mean_ci_width\tn_panels\tn_entries\tentries_list\n")
        for mt, a in sorted(metric_agg.items(), key=lambda x: -x[1]["n_total"]):
            n = a["n_total"]
            sr = a["n_sig"] / n if n > 0 else 0.0
            ma = a["abs_delta_sum"] / n if n > 0 else 0.0
            mc = a["ci_width_sum"] / n if n > 0 else 0.0
            fh.write(f'{mt}\t{n}\t{a["n_sig"]}\t{sr:.4f}\t{ma:.6f}\t'
                     f'{mc:.6f}\t{len(a["panels"])}\t{len(a["entries"])}\t'
                     f'{";".join(sorted(a["entries"]))}\n')

    # ------------------------------------------------------------------
    # Output 4: tier summary
    # ------------------------------------------------------------------
    out_tier = os.path.join(OUT_DIR, "p6_iter174_tier_summary.tsv")
    tier_agg = defaultdict(lambda: {"n_entries": 0, "n_total": 0,
                                    "n_sig": 0, "abs_sum": 0.0,
                                    "ciw_sum": 0.0, "entries": []})
    for e in entries:
        t = e["tier"]
        tier_agg[t]["n_entries"] += 1
        tier_agg[t]["n_total"] += e["n_total"]
        tier_agg[t]["n_sig"] += e["n_sig"]
        for r in e["rows"]:
            tier_agg[t]["abs_sum"] += r["abs_delta"]
            tier_agg[t]["ciw_sum"] += r["ci_width"]
        tier_agg[t]["entries"].append(e["id"])
    with open(out_tier, "w") as fh:
        fh.write("tier\tn_entries\tn_total\tn_sig\tsig_rate\t"
                 "mean_abs_delta\tmean_ci_width\tentries\n")
        for tier in ["A", "B", "C", "D"]:
            a = tier_agg.get(tier)
            if a is None or a["n_total"] == 0:
                if a is None:
                    continue
                fh.write(f'{tier}\t{a["n_entries"]}\t0\t0\tNA\tNA\tNA\t'
                         f'{";".join(a["entries"])}\n')
                continue
            n = a["n_total"]
            sr = a["n_sig"] / n
            ma = a["abs_sum"] / n
            mc = a["ciw_sum"] / n
            fh.write(f'{tier}\t{a["n_entries"]}\t{n}\t{a["n_sig"]}\t'
                     f'{sr:.4f}\t{ma:.6f}\t{mc:.6f}\t{";".join(a["entries"])}\n')

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------
    n_total_entries = len(entries)
    n_tier_d = tier_agg["D"]["n_entries"] if "D" in tier_agg else 0
    n_tier_a = tier_agg["A"]["n_entries"] if "A" in tier_agg else 0
    n_tier_b = tier_agg["B"]["n_entries"] if "B" in tier_agg else 0

    # H1: tier-A sig_rate > tier-B sig_rate
    tier_a_sr = (tier_agg["A"]["n_sig"] / tier_agg["A"]["n_total"]
                 if "A" in tier_agg and tier_agg["A"]["n_total"] > 0 else 0.0)
    tier_b_sr = (tier_agg["B"]["n_sig"] / tier_agg["B"]["n_total"]
                 if "B" in tier_agg and tier_agg["B"]["n_total"] > 0 else 0.0)
    h1_pass = tier_a_sr > tier_b_sr

    # H2: every tier-A entry has ≥2 panels; every tier-B has ≤1
    tier_a_entries = [e for e in entries if e["tier"] == "A"]
    tier_b_entries = [e for e in entries if e["tier"] == "B"]
    h2_pass = (all(e["n_panels"] >= 2 for e in tier_a_entries)
               and all(e["n_panels"] <= 1 for e in tier_b_entries))

    # H3: on n2_same_stack_last10, reward_mean sig_rate > zvf sig_rate
    rm_rows = [r for e in entries for r in e["rows"]
               if r["panel"] == "n2_same_stack_last10" and r["metric"] == "reward_mean"]
    zv_rows = [r for e in entries for r in e["rows"]
               if r["panel"] == "n2_same_stack_last10" and r["metric"] == "zvf"]
    rm_sr = sum(1 for r in rm_rows if r["significant"]) / len(rm_rows) if rm_rows else 0.0
    zv_sr = sum(1 for r in zv_rows if r["significant"]) / len(zv_rows) if zv_rows else 0.0
    h3_pass = rm_sr > zv_sr

    # H4: tier-A mean CI width < tier-B mean CI width
    tier_a_ciw = (tier_agg["A"]["ciw_sum"] / tier_agg["A"]["n_total"]
                  if "A" in tier_agg and tier_agg["A"]["n_total"] > 0 else 1.0)
    tier_b_ciw = (tier_agg["B"]["ciw_sum"] / tier_agg["B"]["n_total"]
                  if "B" in tier_agg and tier_agg["B"]["n_total"] > 0 else 1.0)
    h4_pass = tier_a_ciw < tier_b_ciw

    # H5: tier-D fraction > 0.0 (i.e. there ARE tier-D entries)
    h5_pass = n_tier_d > 0
    tier_d_frac = n_tier_d / n_total_entries if n_total_entries > 0 else 0.0
    tier_d_lo, tier_d_hi = wilson_ci(n_tier_d, n_total_entries)

    # ------------------------------------------------------------------
    # Output 5: structured summary
    # ------------------------------------------------------------------
    summary = {
        "iter": 174,
        "pillar": "P6",
        "vein": "tier-stratified metric-coverage audit",
        "n_entries": n_total_entries,
        "tier_counts": {"A": n_tier_a, "B": n_tier_b, "D": n_tier_d},
        "tier_a_sig_rate": tier_a_sr,
        "tier_b_sig_rate": tier_b_sr,
        "tier_a_mean_ciw": tier_a_ciw,
        "tier_b_mean_ciw": tier_b_ciw,
        "n2_reward_mean_sig_rate": rm_sr,
        "n2_zvf_sig_rate": zv_sr,
        "n2_reward_mean_rows": len(rm_rows),
        "n2_zvf_rows": len(zv_rows),
        "tier_d_fraction": tier_d_frac,
        "tier_d_wilson_ci95": [tier_d_lo, tier_d_hi],
        "hypotheses": {
            "H1_tierA_sig_rate_gt_tierB": {"pass": h1_pass,
                                           "tierA_sr": tier_a_sr,
                                           "tierB_sr": tier_b_sr},
            "H2_tierA_n_panels_ge_2_tierB_le_1": {
                "pass": h2_pass,
                "tierA_panels": [e["n_panels"] for e in tier_a_entries],
                "tierB_panels": [e["n_panels"] for e in tier_b_entries],
            },
            "H3_n2_reward_mean_sig_gt_zvf": {"pass": h3_pass,
                                             "reward_mean_sr": rm_sr,
                                             "zvf_sr": zv_sr},
            "H4_tierA_mean_ciw_lt_tierB": {"pass": h4_pass,
                                           "tierA_ciw": tier_a_ciw,
                                           "tierB_ciw": tier_b_ciw},
            "H5_tierD_exists": {"pass": h5_pass,
                                "tierD_count": n_tier_d,
                                "tierD_wilson_ci95": [tier_d_lo, tier_d_hi]},
        },
        "artifacts": {
            "per_entry": "experiments/results/p5p8/p6_iter174_per_entry.tsv",
            "per_entry_panel": "experiments/results/p5p8/p6_iter174_per_entry_panel.tsv",
            "per_metric": "experiments/results/p5p8/p6_iter174_per_metric.tsv",
            "tier_summary": "experiments/results/p5p8/p6_iter174_tier_summary.tsv",
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_sum = os.path.join(OUT_DIR, "p6_iter174_summary.json")
    with open(out_sum, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # Console banner
    print(f"iter174 P6 tier-stratified audit complete in "
          f"{summary['elapsed_sec']}s — entries={n_total_entries} "
          f"(A={n_tier_a}, B={n_tier_b}, D={n_tier_d})")
    print(f"  H1 tierA_sr {tier_a_sr:.3f} > tierB_sr {tier_b_sr:.3f}: "
          f"{'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 tierA panels≥2, tierB≤1: "
          f"{'PASS' if h2_pass else 'FAIL'}")
    print(f"  H3 n2 reward_mean sr {rm_sr:.3f} > zvf sr {zv_sr:.3f}: "
          f"{'PASS' if h3_pass else 'FAIL'}")
    print(f"  H4 tierA ciw {tier_a_ciw:.4f} < tierB ciw {tier_b_ciw:.4f}: "
          f"{'PASS' if h4_pass else 'FAIL'}")
    print(f"  H5 tier-D = {n_tier_d}/{n_total_entries} = "
          f"{tier_d_frac:.3f}: {'PASS' if h5_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

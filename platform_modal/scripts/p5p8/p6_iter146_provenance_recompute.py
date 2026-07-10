#!/usr/bin/env python3
"""
Iter-146 — P6 per-row provenance-to-claim recompute audit.

For every measured[] row in delta_*.json: infer source family from path/panel,
recompute (delta, ci_low, ci_high, n) deterministically (LCG B=2000, seed=20260705),
classify as MATCH / MATCH_POINT / POINT_MATCH_WRONG_SOURCE / DRIFT_SIGN / etc.

Inputs: registry/entries/delta_*.json, n2_metrics.tsv, zvf_iter130_*,
        length_bias_iter60, qp7_adaptive.tsv, registry/schema.json.
Outputs: experiments/results/p5p8/p6_iter146_{audit,per_entry,summary}.{tsv,json}
         and p6_iter146_fix_plan.tsv (rows where source path is misattributed).
"""
import csv
import json
import glob
import os
import math
import sys
from collections import defaultdict

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(WORKTREE, "registry", "entries")
OUT_DIR = os.path.join(WORKTREE, "experiments", "results", "p5p8")
BOOT_SEED = 20260705
B_BOOT = 2000


def lcg_next(state):
    """Deterministic LCG; returns float in [0,1)."""
    state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
    return state[0] / 0x7FFFFFFF


def paired_bootstrap(diffs, n_boot=B_BOOT, seed=BOOT_SEED):
    """Paired bootstrap on `diffs` array; returns (mean, ci_lo, ci_high)."""
    n = len(diffs)
    state = [seed]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            i = int(lcg_next(state) * n)
            s += diffs[i]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    point = sum(diffs) / n
    return point, lo, hi


def _read_tsv(path, group_by=None, drop_agg=False, cast_nums=False):
    """Generic TSV reader; returns list of dicts or dict-grouped-by-method."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if drop_agg:
        rows = [r for r in rows if str(r.get("seed", "")).lstrip("-").isdigit()]
    if cast_nums:
        rows = [{k: (float(v) if k not in ("method", "seed", "arm") else v)
                 for k, v in r.items()} for r in rows]
    if group_by is None:
        return rows
    out = defaultdict(list) if group_by == "list" else {}
    for r in rows:
        key = r["method"]
        if group_by == "list":
            out[key].append(r)
        else:
            if key not in out:
                out[key] = r
    return out


def load_n2_metrics():
    return _read_tsv(os.path.join(WORKTREE, "experiments", "results",
                                  "n2_reward_tensor_resume", "n2_metrics.tsv"),
                     group_by="list", cast_nums=True)


def load_zvf130():
    return _read_tsv(os.path.join(WORKTREE, "experiments", "results",
                                  "zvf_iter130_method_risk.tsv"),
                     group_by="first")


def load_zvf130_risk_index():
    return _read_tsv(os.path.join(WORKTREE, "experiments", "results",
                                  "zvf_iter130_risk_index.tsv"),
                     group_by="list", drop_agg=True)


def _tsv_cols(path):
    with open(path) as f:
        return next(csv.reader(f, delimiter="\t"))


def zvf130_columns():
    return _tsv_cols(os.path.join(WORKTREE, "experiments", "results",
                                  "zvf_iter130_method_risk.tsv"))


def zvf130_risk_index_columns():
    return _tsv_cols(os.path.join(WORKTREE, "experiments", "results",
                                  "zvf_iter130_risk_index.tsv"))


def recompute_n2_panel(panel, method_variant, metric, last_k):
    """Paired-step bootstrap on last K steps; variant minus grpo; one diff/step."""
    n2 = load_n2_metrics()
    if method_variant not in n2 or "grpo" not in n2:
        return None
    v, g = n2[method_variant][-last_k:], n2["grpo"][-last_k:]
    diffs = [rv[metric] - rg[metric] for rv, rg in zip(v, g)]
    point, lo, hi = paired_bootstrap(diffs)
    return point, lo, hi, len(diffs), (lo > 0) or (hi < 0)


def recompute_zvf130_risk_metric(metric, method_variant):
    """Paired-seed bootstrap on per-seed metric values."""
    zvf = load_zvf130_risk_index()
    if method_variant not in zvf or "grpo" not in zvf or metric not in zvf[method_variant][0]:
        return None
    vs = [float(r[metric]) for r in zvf[method_variant]]
    gs = [float(r[metric]) for r in zvf["grpo"]]
    n = min(len(vs), len(gs))
    diffs = [v - g for v, g in zip(vs[:n], gs[:n])]
    point, lo, hi = paired_bootstrap(diffs)
    return point, lo, hi, n, (lo > 0) or (hi < 0)


def load_length_bias():
    """Load length_bias_iter60 → dict metric → list of mean_diff values."""
    out = defaultdict(list)
    for row in csv.DictReader(open(os.path.join(WORKTREE, "experiments",
                                  "results", "length_bias_iter60_grpo_vs_drgrpo.tsv")),
                              delimiter="\t"):
        try:
            out[row["metric"]].append(float(row["mean_diff"]))
        except ValueError:
            continue
    return out


def load_qp7_adaptive():
    """Load qp7_adaptive.tsv → dict arm → list of step dicts."""
    out = defaultdict(list)
    for row in csv.DictReader(open(os.path.join(WORKTREE, "experiments",
                                  "results", "quick_20260704", "qp7_adaptive.tsv")),
                              delimiter="\t"):
        try:
            out[row["arm"]].append({"step": int(row["step"]), **row})
        except (ValueError, KeyError):
            continue
    return out


def _pack(point, lo, hi, n):
    return point, lo, hi, n, (lo > 0) or (hi < 0)


def recompute_length_bias_metric(metric):
    """Pooled mean across tasks of mean_diff for the metric (parametric normal-approx CI)."""
    diffs = load_length_bias().get(metric)
    if not diffs:
        return None
    n = len(diffs)
    point = sum(diffs) / n
    if n >= 2:
        var = sum((d - point) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
        return _pack(point, point - 1.96 * se, point + 1.96 * se, n)
    return _pack(point, point, point, n)


def recompute_qp7_adaptive_metric(metric):
    """Paired-step bootstrap on per-step metric, arm B minus arm A (aligned by step)."""
    qp = load_qp7_adaptive()
    if "A" not in qp or "B" not in qp or metric not in qp["A"][0]:
        return None
    a, b = {r["step"]: r for r in qp["A"]}, {r["step"]: r for r in qp["B"]}
    diffs = [float(b[s][metric]) - float(a[s][metric])
             for s in sorted(set(a) & set(b))]
    point, lo, hi = paired_bootstrap(diffs)
    return _pack(point, lo, hi, len(diffs))


def infer_recipe(measured_row):
    """Return (family, args) inferred from source path + panel + metric.

    Family keys: n2 (paired-step on n2_metrics.tsv), zvf130_risk_index
    (paired-seed on zvf_iter130_risk_index.tsv), zvf130_method_risk (point
    only on zvf_iter130_method_risk.tsv aggregate), length_bias (pooled
    across tasks), qp7_adaptive (paired-arm per-step).
    """
    src = measured_row.get("source", "")
    panel = measured_row.get("panel", "")
    metric = measured_row.get("metric", "")
    if "n2_reward_tensor_resume" in src and panel.startswith("n2_same_stack"):
        last_k = 10 if "last10" in panel else (20 if "last20" in panel
                  else (40 if "full40" in panel else 5))
        return ("n2", {"last_k": last_k, "metric": metric})
    if "zvf_iter130_risk_index.tsv" in src and panel == "zvf130_5seed":
        return ("zvf130_risk_index", {"metric": metric})
    if "zvf_iter130_method_risk.tsv" in src and panel == "zvf130_5seed":
        return ("zvf130_method_risk", {"metric": metric})
    if "length_bias_iter60" in src:
        return ("length_bias", {"metric": metric})
    if "qp7_adaptive" in src:
        return ("qp7_adaptive", {"metric": metric})
    return (None, None)


def check_source_field_mismatch(measured_row, family):
    """Return non-None verdict string if the metric column is missing from the
    cited source file (forward-pointing provenance gap)."""
    metric = measured_row.get("metric", "")
    src = measured_row.get("source", "")
    if family == "n2":
        cols = ["zvf", "reward_mean", "pcd", "mean_len", "loss", "cv_len"]
        if metric not in cols:
            return f"SOURCE_FIELD_MISMATCH (n2 has none of {cols!r})"
        return None
    if family == "zvf130_method_risk":
        cols = zvf130_columns()
        if metric not in cols:
            actual = "zvf_iter130_risk_index.tsv" if metric == "mean_zvf" else "(unknown)"
            return f"SOURCE_FIELD_MISMATCH ({os.path.basename(src)} lacks '{metric}')"
        return None
    if family == "zvf130_risk_index":
        cols = zvf130_risk_index_columns()
        if metric not in cols:
            actual = "zvf_iter130_method_risk.tsv" if metric == "mag_mean" else "(unknown)"
            return f"SOURCE_FIELD_MISMATCH ({os.path.basename(src)} lacks '{metric}'; candidate: {actual})"
        return None
    return None


def recompute_zvf130_method_risk_point(metric, method_variant):
    """For zvf130_method_risk.tsv (per-method aggregate file), derive point delta
    directly from the aggregate table. CI is not recoverable from this file."""
    zvf = load_zvf130()
    if method_variant not in zvf or "grpo" not in zvf:
        return None
    if metric not in zvf[method_variant]:
        return None
    v = float(zvf[method_variant][metric])
    g = float(zvf["grpo"][metric])
    n = int(zvf[method_variant].get("n_seeds", 5) or 5)
    return v - g, None, None, n, None  # CI not derivable here


def recompute(family, args, method_variant):
    if family == "n2":
        return recompute_n2_panel(None, method_variant, args["metric"], args["last_k"])
    if family == "zvf130_risk_index":
        return recompute_zvf130_risk_metric(args["metric"], method_variant)
    if family == "zvf130_method_risk":
        return recompute_zvf130_method_risk_point(args["metric"], method_variant)
    if family == "length_bias":
        return recompute_length_bias_metric(args["metric"])
    if family == "qp7_adaptive":
        return recompute_qp7_adaptive_metric(args["metric"])
    return None


def classify(stored, recomputed, abs_tol=0.01, rel_tol=0.05):
    """Classify drift between stored and recomputed. Returns (verdict, abs_drift).

    If `recomputed[1]` is None (CI not derivable from cited source — e.g. the
    per-method aggregate file), only the point delta and `n` are compared.
    """
    if recomputed is None:
        return "MISSING_SOURCE", None
    s_d, s_lo, s_hi, s_n, s_sig = stored
    r_d, r_lo, r_hi, r_n, r_sig = recomputed
    tol = max(abs_tol, rel_tol * max(abs(s_d), 1e-9))
    abs_drift = abs(r_d - s_d)
    if r_lo is None or r_hi is None:
        # Point-only recompute: match on (delta, n) only
        if abs_drift <= tol and s_n == r_n:
            return "MATCH_POINT", abs_drift
        return "DRIFT_POINT", abs_drift
    s_half = max(abs(s_hi - s_lo) / 2.0, 1e-9)
    r_half = max(abs(r_hi - r_lo) / 2.0, 1e-9)
    ci_overlap = (s_half + r_half) >= abs_drift
    sig_match = (bool(s_sig) == bool(r_sig))
    n_match = (s_n == r_n)
    if abs_drift <= tol and ci_overlap and sig_match and n_match:
        return "MATCH", abs_drift
    if not sig_match:
        return "DRIFT_SIGN", abs_drift
    if abs_drift > tol and not ci_overlap:
        return "DRIFT_MAG", abs_drift
    if not n_match:
        return "DRIFT_N", abs_drift
    return "DRIFT_OTHER", abs_drift


def main():
    rows = []
    per_entry = defaultdict(lambda: {"total": 0, "match": 0, "drift": 0,
                                       "missing": 0, "drift_kinds": defaultdict(int)})

    delta_files = sorted(glob.glob(os.path.join(REG_DIR, "delta_*.json")))
    for fpath in delta_files:
        d = json.load(open(fpath))
        eid = d.get("id", os.path.basename(fpath).replace(".json", ""))
        method_variant = eid.replace("delta_", "")
        for m in (d.get("measured") or []):
            family, args = infer_recipe(m)
            d_stored, lo_stored, hi_stored = m.get("delta"), m.get("ci_low"), m.get("ci_high")
            per_entry[eid]["total"] += 1
            if d_stored is None:
                # row carries no numeric delta (deferred / null CI)
                per_entry[eid]["missing"] += 1
                rows.append({
                    "entry_id": eid,
                    "metric": m.get("metric", ""),
                    "panel": m.get("panel", ""),
                    "stored_delta": "", "stored_ci_low": "", "stored_ci_high": "",
                    "stored_n": int(m.get("n", 0)),
                    "stored_sig": int(bool(m.get("significant", False))),
                    "verdict": "NULL_DELTA_SKIPPED",
                    "abs_drift": "",
                    "source_family": family or "",
                })
                continue
            stored = (float(d_stored), float(lo_stored) if lo_stored is not None else 0.0,
                      float(hi_stored) if hi_stored is not None else 0.0,
                      int(m["n"]), bool(m["significant"]))
            if family is None:
                verdict = "UNINFERABLE_SOURCE"
                drift = None
            elif (sfm := check_source_field_mismatch(m, family)) is not None:
                # Field mismatch: try the alternative source file to confirm
                # whether the value itself is correct (just misattributed).
                alt_metric = m.get("metric", "")
                alt_family = None
                if family == "zvf130_method_risk":
                    alt_family = "zvf130_risk_index"
                elif family == "zvf130_risk_index":
                    alt_family = "zvf130_method_risk"
                alt_match = False
                alt_verdict = ""
                if alt_family is not None:
                    alt_rec = recompute(alt_family, {"metric": alt_metric},
                                        method_variant)
                    if alt_rec is not None:
                        r_d = alt_rec[0]
                        tol = max(0.01, 0.05 * max(abs(stored[0]), 1e-9))
                        if abs(r_d - stored[0]) <= tol:
                            alt_match = True
                            alt_verdict = "POINT_MATCH_WRONG_SOURCE"
                        else:
                            alt_verdict = "POINT_DRIFT_WRONG_SOURCE"
                    else:
                        alt_verdict = "NO_ALT_RECOMPUTE"
                else:
                    alt_verdict = "NO_ALT_FAMILY"
                verdict = alt_verdict if alt_verdict else sfm
                drift = None
                per_entry[eid]["drift_kinds"][verdict] += 1
                per_entry[eid]["drift"] += 1
            else:
                rec = recompute(family, args, method_variant)
                verdict, drift = classify(stored, rec)
                if verdict == "MATCH" or verdict == "MATCH_POINT":
                    per_entry[eid]["match"] += 1
                elif verdict == "MISSING_SOURCE":
                    per_entry[eid]["missing"] += 1
                else:
                    per_entry[eid]["drift"] += 1
                    per_entry[eid]["drift_kinds"][verdict] += 1
            rows.append({
                "entry_id": eid,
                "metric": m.get("metric", ""),
                "panel": m.get("panel", ""),
                "stored_delta": float(m["delta"]),
                "stored_ci_low": "" if m.get("ci_low") is None else float(m["ci_low"]),
                "stored_ci_high": "" if m.get("ci_high") is None else float(m["ci_high"]),
                "stored_n": int(m["n"]),
                "stored_sig": int(bool(m["significant"])),
                "verdict": verdict,
                "abs_drift": "" if drift is None else f"{drift:.6f}",
                "source_family": family or "",
            })

    # write per-row audit
    out_tsv = os.path.join(OUT_DIR, "p6_iter146_audit.tsv")
    with open(out_tsv, "w") as f:
        cols = ["entry_id", "metric", "panel", "stored_delta", "stored_ci_low",
                "stored_ci_high", "stored_n", "stored_sig", "verdict",
                "abs_drift", "source_family"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # write auto-fix plan (rows where the value is correct but source is misattributed)
    fix_rows = []
    src_map = {
        "zvf130_method_risk": "experiments/results/zvf_iter130_risk_index.tsv",
        "zvf130_risk_index": "experiments/results/zvf_iter130_method_risk.tsv",
    }
    for r in rows:
        v = r["verdict"]
        if v == "POINT_MATCH_WRONG_SOURCE":
            cur_src = None
            # the source_family tells us which file was cited; we want the OTHER one
            if r["source_family"] in src_map:
                new_src = src_map[r["source_family"]]
                fix_rows.append({
                    "entry_id": r["entry_id"],
                    "metric": r["metric"],
                    "panel": r["panel"],
                    "old_source": "(see registry)",
                    "new_source": new_src,
                    "stored_delta": r["stored_delta"],
                    "verified_match": "yes",
                })
    fix_tsv = os.path.join(OUT_DIR, "p6_iter146_fix_plan.tsv")
    with open(fix_tsv, "w") as f:
        f.write("entry_id\tmetric\tpanel\told_source\tnew_source\t"
                "stored_delta\tverified_match\n")
        for fr in fix_rows:
            f.write("\t".join(str(fr[c]) for c in
                              ["entry_id", "metric", "panel", "old_source",
                               "new_source", "stored_delta", "verified_match"]) + "\n")

    # write per-entry summary
    out_per = os.path.join(OUT_DIR, "p6_iter146_per_entry.tsv")
    with open(out_per, "w") as f:
        f.write("entry_id\ttotal\tmatch\tdrift\tmissing\tpct_match\n")
        for eid in sorted(per_entry):
            e = per_entry[eid]
            pct = (e["match"] / e["total"] * 100.0) if e["total"] else 0.0
            f.write(f"{eid}\t{e['total']}\t{e['match']}\t{e['drift']}\t"
                    f"{e['missing']}\t{pct:.1f}\n")

    # headline counts
    total = sum(e["total"] for e in per_entry.values())
    n_match = sum(e["match"] for e in per_entry.values())
    n_drift = sum(e["drift"] for e in per_entry.values())
    n_missing = sum(e["missing"] for e in per_entry.values())
    drift_kinds = defaultdict(int)
    match_kinds = defaultdict(int)
    for r in rows:
        v = r["verdict"]
        if v.startswith("DRIFT_") or v.startswith("POINT_") or v.startswith("SOURCE_") \
           or v == "UNINFERABLE_SOURCE" or v == "MISSING_SOURCE":
            drift_kinds[v.split(" ")[0]] += 1
        else:
            match_kinds[v] += 1
    families = defaultdict(int)
    for r in rows:
        families[r["source_family"]] += 1

    summary = {
        "iter": 146,
        "pillar": "P6",
        "rows_audited": total,
        "n_match": n_match,
        "n_drift": n_drift,
        "n_missing_source": n_missing,
        "pct_match": (n_match / total * 100.0) if total else 0.0,
        "drift_kinds": dict(drift_kinds),
        "match_kinds": dict(match_kinds),
        "source_family_counts": dict(families),
        "entries_with_drift": [eid for eid, e in per_entry.items()
                               if e["drift"] > 0],
        "boot_recipe": {"seed": BOOT_SEED, "B": B_BOOT,
                        "tolerance_abs": 0.01, "tolerance_rel": 0.05},
    }
    out_json = os.path.join(OUT_DIR, "p6_iter146_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
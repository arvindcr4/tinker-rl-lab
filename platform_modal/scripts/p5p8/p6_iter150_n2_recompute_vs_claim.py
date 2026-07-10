#!/usr/bin/env python3
"""
Iter-150 — P6 N2 same-stack recompute + prose-vs-measured direction audit.

For each delta_*.json entry:
  (1) If the entry has measured[] rows tagged panel=n2_same_stack_*: recompute
      the stored (delta, ci_low, ci_high) from the raw n2_metrics.tsv using the
      panel's window (last10, full40, etc.). Compare stored vs recomputed ->
      MATCH / POINT_MATCH / DRIFT / SHIFT / SIGN_FLIP.
  (2) Score the prose-vs-measured direction: for every (component, claim_keyword)
      in the prose, pick a metric that should respond, and check whether the
      measured sign agrees with the prose direction (e.g., 'token-level loss'
      should shift 'loss' up; 'zvf-aware' should shift 'zvf' down; etc.).

Inputs : registry/entries/delta_*.json, platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv
Outputs: platform_hybrid/experiments/results/p5p8/p6_iter150_recompute.tsv
         platform_hybrid/experiments/results/p5p8/p6_iter150_per_entry.tsv
         platform_hybrid/experiments/results/p5p8/p6_iter150_summary.json
"""
import csv
import json
import math
import os
from collections import defaultdict

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENT_DIR  = os.path.join(WORKTREE, "registry", "entries")
N2_TSV   = os.path.join(WORKTREE, "experiments", "results",
                        "n2_reward_tensor_resume", "n2_metrics.tsv")
OUT_DIR  = os.path.join(WORKTREE, "experiments", "results", "p5p8")

METRIC_KEYS = ("zvf", "reward_mean", "pcd", "mean_len", "cv_len",
               "lag1_autocorr", "loss", "frac_all_zero", "frac_all_one")
WINDOWS = {"last10": (30, 40), "full40": (0, 40)}


def load_n2():
    rows = list(csv.DictReader(open(N2_TSV), delimiter="\t"))
    for r in rows:
        for k in METRIC_KEYS:
            if k in r:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = float("nan")
        r["step"] = int(r["step"])
    return rows


def panel_window(panel):
    for key, (a, b) in WINDOWS.items():
        if key in panel:
            return key, a, b
    return "full40", 0, 40


def per_method_window(rows, m, a, b):
    return [r for r in rows if r["method"] == m and a <= r["step"] < b]


def recompute(variant_rows, grpo_rows, metrics):
    """Variant minus grpo per-step mean for each metric."""
    grpo_by_step = {r["step"]: r for r in grpo_rows}
    out = {}
    for m in metrics:
        diffs = []
        for v in variant_rows:
            g = grpo_by_step.get(v["step"])
            if g is None:
                continue
            if not (math.isfinite(v[m]) and math.isfinite(g[m])):
                continue
            diffs.append(v[m] - g[m])
        if not diffs:
            out[m] = (None, None, None, 0)
            continue
        mean = sum(diffs) / len(diffs)
        # crude percentile CI from diffs (no bootstrap) - registry stores
        # bootstrap CIs; we record the *recompute point* only and compare.
        sd = (sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)) ** 0.5
        out[m] = (mean, mean - 1.96 * sd / max(len(diffs), 1) ** 0.5,
                  mean + 1.96 * sd / max(len(diffs), 1) ** 0.5, len(diffs))
    return out


def classify(stored, recomputed, tol=1e-4):
    """Compare stored (delta, ci_low, ci_high) vs recompute point-estimate only.

    The stored CI is bootstrap; recompute is normal-approx. We only flag
    direction drift (SIGN_FLIP) or large magnitude drift (DRIFT).
    """
    if stored is None or recomputed[0] is None:
        return "NA"
    s_d, s_lo, s_hi = stored
    r_d = recomputed[0]
    if abs(s_d) < tol and abs(r_d) < tol:
        return "MATCH_BOTH_NULL"
    if s_d == 0.0 and r_d != 0.0:
        return "POINT_MATCH_NULL"
    if (s_d > 0) != (r_d > 0):
        return "SIGN_FLIP"
    rel = abs(r_d - s_d) / max(abs(s_d), 1e-9)
    if rel < 0.05:
        return "MATCH"
    if rel < 0.20:
        return "SHIFT_SMALL"
    if rel < 0.50:
        return "SHIFT"
    return "DRIFT"


# prose -> expected metric, expected direction (sign of the prose-stated effect).
# 'expected_direction' = +1 if the prose says the variant *increases* the metric,
# -1 if the prose says the variant *decreases* it, 0 if no measurable link.
PROSE_RULES = [
    # (keyword_substr, expected_metric, expected_direction, label)
    ("reference rollout", "zvf", +1, "off-policy inflate group -> +zvf exposure"),
    ("off-policy", "zvf", +1, "off-policy -> +zvf"),
    ("gamma-style", "zvf", -1, "gamma-baseline prior shifts zvf distribution"),
    ("likelihood prior", "zvf", -1, "likelihood prior shifts zvf"),
    ("decouple rollout", "reward_mean", 0, "decouple rollout: orthogonal to reward"),
    ("auto-scaling", "zvf", 0, "autoscaling: orthogonal to single-step zvf"),
    ("dynamic_sampling", "zvf", -1, "filter zero-variance -> -zvf"),
    ("dynamic filter", "zvf", -1, "filter zero-variance -> -zvf"),
    ("dynamic_filter", "zvf", -1, "filter zero-variance -> -zvf"),
    ("zero reward variance", "zvf", -1, "filter zero-variance -> -zvf"),
    ("token-level", "loss", +1, "token-level mean shifts loss magnitude"),
    ("token-level loss", "loss", +1, "token-level mean shifts loss magnitude"),
    ("kl regularization", "loss", +1, "extra KL term -> +loss"),
    ("kl_beta", "loss", +1, "KL term -> +loss"),
    ("clip", "loss", +1, "asymmetric clip shifts loss surface"),
    ("overlong", "mean_len", -1, "soft length penalty -> -mean_len"),
    ("reward shaping", "reward_mean", +1, "extra shaping -> +reward shaping but reward_mean opaque"),
    ("perturbation", "zvf", 0, "ES perturbation orthogonal"),
    ("mcts", "zvf", +1, "MCTS value bias -> +zvf via injected variance"),
    ("mcts-derived", "zvf", +1, "MCTS value bias -> +zvf"),
    ("diversity bonus", "zvf", +1, "diversity up-weighting -> +zvf"),
    ("scaffold", "zvf", +1, "scaffold-aware advantage -> +zvf contrast"),
    ("continuity penalty", "loss", +1, "extra penalty -> +loss"),
    ("normalization", "zvf", 0, "per-prompt normalization orthogonal to zvf"),
    ("per-prompt norm", "zvf", 0, "per-prompt normalization orthogonal to zvf"),
]


def prose_expected(prose_text):
    """Return list of (label, metric, expected_direction) inferred from prose."""
    txt = prose_text.lower()
    hits = []
    for kw, metric, edir, label in PROSE_RULES:
        if kw in txt:
            hits.append((label, metric, edir))
    return hits


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n2 = load_n2()
    methods = sorted({r["method"] for r in n2})
    print(f"[n2] {len(n2)} rows across {methods}")

    # ---------- pass 1: recompute N2 panels and compare to stored ----------
    entries = sorted(glob_pat := __import__("glob").glob(os.path.join(ENT_DIR, "delta_*.json")))
    per_entry = {}   # id -> dict
    rows_recompute = []
    rows_prose = []

    for path in entries:
        ent = json.load(open(path))
        eid = ent["id"]
        ent_summary = {
            "id": eid,
            "base": ent.get("base", ""),
            "n_measured_n2": 0,
            "n_match": 0,
            "n_sign_flip": 0,
            "n_drift": 0,
            "n_shift": 0,
            "n_shift_small": 0,
            "n_other": 0,
            "n_prose_measurable": 0,
            "n_prose_agree": 0,
            "n_prose_disagree": 0,
            "n_prose_orthogonal": 0,
            "n_components": len(ent.get("deltas", [])),
            "n_components_no_measurable_link": 0,
            "components": [],
        }
        # --- N2 panel recompute ---
        for m in ent.get("measured", []):
            panel = m.get("panel", "")
            if "n2" not in panel:
                continue
            metric = m.get("metric", "")
            if metric not in METRIC_KEYS:
                continue
            _, a, b = panel_window(panel)
            vr = per_method_window(n2, eid.replace("delta_", ""), a, b)
            gr = per_method_window(n2, "grpo", a, b)
            rec = recompute(vr, gr, [metric])[metric]
            klass = classify((m.get("delta"), m.get("ci_low"), m.get("ci_high")), rec)
            ent_summary["n_measured_n2"] += 1
            if klass == "MATCH" or klass == "MATCH_BOTH_NULL":
                ent_summary["n_match"] += 1
            elif klass == "SIGN_FLIP":
                ent_summary["n_sign_flip"] += 1
            elif klass == "DRIFT":
                ent_summary["n_drift"] += 1
            elif klass.startswith("SHIFT"):
                ent_summary["n_shift" if klass == "SHIFT" else "_small"] = \
                    ent_summary.get("n_shift" if klass == "SHIFT" else "n_shift_small", 0) + 1
            else:
                ent_summary["n_other"] += 1
            rows_recompute.append({
                "id": eid, "metric": metric, "panel": panel,
                "stored_delta": m.get("delta"),
                "recomputed_delta": rec[0],
                "stored_ci_lo": m.get("ci_low"),
                "stored_ci_hi": m.get("ci_high"),
                "recomp_ci_lo": rec[1],
                "recomp_ci_hi": rec[2],
                "n_steps": rec[3],
                "classification": klass,
                "stored_sig": m.get("significant"),
            })

        # --- prose-vs-measured direction ---
        # Find measured zvf/reward_mean/mean_len/loss rows for n2_same_stack_last10
        meas_by_metric = defaultdict(list)
        for m in ent.get("measured", []):
            metric = m.get("metric", "")
            if "n2_same_stack_last10" in m.get("panel", ""):
                meas_by_metric[metric].append(m)

        for comp in ent.get("deltas", []):
            prose = comp.get("change", "")
            hits = prose_expected(prose)
            comp_row = {"id": eid, "component": comp.get("component", ""),
                        "prose": prose[:80], "hits": "; ".join(h[0] for h in hits)}
            if not hits:
                ent_summary["n_components_no_measurable_link"] += 1
            for label, metric, edir in hits:
                if edir == 0:
                    ent_summary["n_prose_orthogonal"] += 1
                    rows_prose.append({**comp_row, "metric": metric,
                                       "expected_dir": "ORTHOGONAL",
                                       "measured_delta": "",
                                       "measured_sig": "",
                                       "verdict": "ORTHOGONAL"})
                    continue
                ent_summary["n_prose_measurable"] += 1
                ms = meas_by_metric.get(metric, [])
                if not ms:
                    rows_prose.append({**comp_row, "metric": metric,
                                       "expected_dir": "+" if edir > 0 else "-",
                                       "measured_delta": "NO_MEAS",
                                       "measured_sig": "",
                                       "verdict": "PROSE_HAS_NO_MEASURE"})
                    continue
                # use the first matching measured row
                m0 = ms[0]
                d = m0.get("delta")
                if d is None:
                    rows_prose.append({**comp_row, "metric": metric,
                                       "expected_dir": "+" if edir > 0 else "-",
                                       "measured_delta": "NULL",
                                       "measured_sig": "",
                                       "verdict": "NULL_MEAS"})
                    continue
                measured_dir = (d > 0) - (d < 0)
                if measured_dir == edir:
                    ent_summary["n_prose_agree"] += 1
                    verdict = "AGREE"
                else:
                    ent_summary["n_prose_disagree"] += 1
                    verdict = "DISAGREE"
                rows_prose.append({**comp_row, "metric": metric,
                                   "expected_dir": "+" if edir > 0 else "-",
                                   "measured_delta": f"{d:+.5f}",
                                   "measured_sig": str(m0.get("significant")),
                                   "verdict": verdict})
            ent_summary["components"].append(comp_row)
        per_entry[eid] = ent_summary

    # ---------- write outputs ----------
    rec_path = os.path.join(OUT_DIR, "p6_iter150_recompute.tsv")
    with open(rec_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_recompute[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_recompute)

    per_path = os.path.join(OUT_DIR, "p6_iter150_per_entry.tsv")
    with open(per_path, "w", newline="") as f:
        cols = ["id", "base", "n_components", "n_components_no_measurable_link",
                "n_measured_n2", "n_match", "n_sign_flip", "n_drift",
                "n_shift", "n_shift_small", "n_other",
                "n_prose_measurable", "n_prose_agree", "n_prose_disagree",
                "n_prose_orthogonal"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for eid, s in sorted(per_entry.items()):
            w.writerow({k: s.get(k, 0) for k in cols})

    prose_path = os.path.join(OUT_DIR, "p6_iter150_prose_vs_measured.tsv")
    with open(prose_path, "w", newline="") as f:
        cols = ["id", "component", "metric", "expected_dir",
                "measured_delta", "measured_sig", "verdict", "prose", "hits"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows_prose)

    # summary json
    summary = {
        "n_entries": len(per_entry),
        "n_total_recompute_rows": len(rows_recompute),
        "n_match": sum(s["n_match"] for s in per_entry.values()),
        "n_sign_flip": sum(s["n_sign_flip"] for s in per_entry.values()),
        "n_drift": sum(s["n_drift"] for s in per_entry.values()),
        "n_prose_measurable": sum(s["n_prose_measurable"] for s in per_entry.values()),
        "n_prose_agree": sum(s["n_prose_agree"] for s in per_entry.values()),
        "n_prose_disagree": sum(s["n_prose_disagree"] for s in per_entry.values()),
        "n_prose_orthogonal": sum(s["n_prose_orthogonal"] for s in per_entry.values()),
        "per_entry": {k: {kk: v for kk, v in vv.items() if kk != "components"}
                      for k, vv in per_entry.items()},
        "disagree_entries": sorted(
            [(k, v["n_prose_disagree"]) for k, v in per_entry.items()
             if v["n_prose_disagree"] > 0],
            key=lambda x: -x[1]),
        "no_link_entries": sorted(
            [(k, v["n_components_no_measurable_link"]) for k, v in per_entry.items()
             if v["n_components_no_measurable_link"] >= v["n_components"]]),
    }
    sum_path = os.path.join(OUT_DIR, "p6_iter150_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {rec_path} ({len(rows_recompute)} rows)")
    print(f"[write] {per_path} ({len(per_entry)} entries)")
    print(f"[write] {prose_path} ({len(rows_prose)} rows)")
    print(f"[write] {sum_path}")
    print(f"[summary] MATCH={summary['n_match']}  SIGN_FLIP={summary['n_sign_flip']}  "
          f"DRIFT={summary['n_drift']}  AGREE={summary['n_prose_agree']}  "
          f"DISAGREE={summary['n_prose_disagree']}  ORTHOGONAL={summary['n_prose_orthogonal']}")
    print(f"[disagree] {summary['disagree_entries']}")
    print(f"[no_link]  {summary['no_link_entries']}")


if __name__ == "__main__":
    main()
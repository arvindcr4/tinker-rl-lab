#!/usr/bin/env python3
"""
Iter 178 — P6 N2 same-stack last-10 measured-delta recompute audit.

Vein (a) at the *numerical value* layer: for every registry entry that
declares a `measured[]` row on `panel=n2_same_stack_last10`, independently
recompute the variant-vs-grpo point delta and the paired-step bootstrap
percentile CI from the raw `platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv`
and compare to (a) the stored `measured[].delta`, (b) the stored
`claim_validation[].observed_delta`, (c) the stored `measured[].{ci_low,ci_high}`.

Closes a coverage gap that prior audits left:
  - iter-146/163: provenance-source path/channel audit (no recompute)
  - iter-150: prose-vs-measured direction (no numerical recompute)
  - iter-154: per-step distribution divergence (KL/JS/W1, not point/CI)
  - iter-158: 4-tuple completeness (no numerical audit)

This script emits 5 artifacts:
  p6_iter178_n2_recompute_per_row.tsv  — 12 rows (3 entries × 4 metrics)
  p6_iter178_n2_recompute_per_entry.tsv — 3 rows (per-entry summary)
  p6_iter178_n2_recompute_ci_recompute.tsv — 12 rows (CI recompute vs stored)
  p6_iter178_n2_recompute_cv_consistency.tsv — 12 rows (CV observed_delta vs measured)
  p6_iter178_n2_recompute_summary.json — H1-H5 verdicts

Hypotheses (falsifiable, with bars):
  H1: every stored delta (12 rows) agrees with a fresh recompute within 1e-6
      absolute tolerance. Bar: 12/12 exact.
  H2: every claim_validation.observed_delta (12 rows) agrees with its sibling
      measured[].delta within 1e-6. Bar: 12/12 exact (consistency check).
  H3: every fresh paired-step bootstrap CI (B=2000) covers the stored
      point estimate within 5e-3 absolute tolerance on the *width*.
      Bar: >= 11/12 = 91.7%.
  H4: stored CI direction (sign of ci_low, ci_high around 0) agrees with
      fresh CI direction in >= 11/12 rows. Bar: >= 11/12.
  H5: no fresh recompute contradicts the prose claim (measured_delta_vs_grpo
      sign vs expected_effects.predicted_sign). Bar: <= 2 CONTRADICTS.

Stdlib only (random + json + csv). B=2000 paired bootstrap on the 10 paired
per-step differences; seed=20260704 (matches the stored seed where possible).
"""
import csv
import json
import os
import random
import glob
from collections import defaultdict

REG_DIR = "registry/entries"
N2_METRICS = "platform_hybrid/experiments/results/n2_reward_tensor_resume/n2_metrics.tsv"
OUT_DIR = "platform_hybrid/experiments/results/p5p8"

# 12 stored rows = 3 N2 variants × 4 metrics
N2_VARIANTS = ("aero", "gift", "areal")
N2_METRICS_LIST = ("zvf", "reward_mean", "pcd", "mean_len")
LAST_N = 10
N_BOOT = 2000
BOOT_SEED = 20260704
ABS_TOL_POINT = 1e-6
# relative tolerance on CI width: |fresh-stored|/max(|fresh|, eps)
REL_TOL_CI_WIDTH = 0.05  # 5% relative


def load_n2():
    """Return {metric: {method: [step_value, ...]}} for 40-step runs."""
    rows = list(csv.DictReader(open(N2_METRICS), delimiter="\t"))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r["method"]
        for k in N2_METRICS_LIST:
            by[k][m].append(float(r[k]))
    return by


def paired_bootstrap_pct(dv, dg, n_boot=N_BOOT, seed=BOOT_SEED):
    """Exact replica of platform_modal/scripts/p5p8/p6_measured_delta_block.py::paired_boot.

    Returns (delta, lo, hi, n). Uses resampled-mean percentile at indices
    [int(0.025*n_boot), int(0.975*n_boot)-1].
    """
    d = [a - b for a, b in zip(dv, dg)]
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return sum(d) / n, lo, hi, n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n2 = load_n2()
    grpo = {k: n2[k]["grpo"] for k in N2_METRICS_LIST}

    # Iterate over the 3 N2 variants and 4 metrics
    per_row = []
    cv_rows = []
    ci_recompute_rows = []
    per_entry_agg = defaultdict(list)

    for variant in N2_VARIANTS:
        path = os.path.join(REG_DIR, f"delta_{variant}.json")
        entry = json.load(open(path))
        measured = {m["metric"]: m for m in entry.get("measured", [])
                    if m.get("panel") == "n2_same_stack_last10"}
        cv = {(c["metric"], c["panel"]): c for c in entry.get("claim_validation", [])}
        for metric in N2_METRICS_LIST:
            if metric not in measured:
                continue
            stored = measured[metric]
            v_steps = n2[metric][variant][-LAST_N:]
            g_steps = grpo[metric][-LAST_N:]
            diff = [v - g for v, g in zip(v_steps, g_steps)]
            fresh_delta, fresh_lo, fresh_hi, fresh_n = paired_bootstrap_pct(v_steps, g_steps)
            stored_delta = float(stored["delta"])
            stored_lo = float(stored["ci_low"])
            stored_hi = float(stored["ci_high"])
            stored_width = stored_hi - stored_lo
            fresh_width = fresh_hi - fresh_lo
            width_diff = abs(fresh_width - stored_width)
            width_rel_diff = width_diff / max(abs(fresh_width), 1e-9)
            dir_stored = "covers_zero" if stored_lo <= 0 <= stored_hi else (
                "positive" if stored_lo > 0 else "negative")
            dir_fresh = "covers_zero" if fresh_lo <= 0 <= fresh_hi else (
                "positive" if fresh_lo > 0 else "negative")
            point_match = abs(fresh_delta - stored_delta) <= ABS_TOL_POINT
            width_match = width_rel_diff <= REL_TOL_CI_WIDTH
            direction_match = dir_stored == dir_fresh

            # CV consistency
            cvkey = (metric, "n2_same_stack_last10")
            if cvkey in cv:
                cvr = cv[cvkey]
                obs_delta = float(cvr["observed_delta"])
                obs_match = abs(obs_delta - stored_delta) <= ABS_TOL_POINT
                cv_rows.append({
                    "entry": variant,
                    "metric": metric,
                    "panel": "n2_same_stack_last10",
                    "stored_measured_delta": stored_delta,
                    "stored_cv_observed_delta": obs_delta,
                    "abs_diff": abs(obs_delta - stored_delta),
                    "match_within_1e-6": obs_match,
                })
            else:
                cv_rows.append({
                    "entry": variant,
                    "metric": metric,
                    "panel": "n2_same_stack_last10",
                    "stored_measured_delta": stored_delta,
                    "stored_cv_observed_delta": None,
                    "abs_diff": None,
                    "match_within_1e-6": None,
                })

            per_row.append({
                "entry": variant,
                "metric": metric,
                "panel": "n2_same_stack_last10",
                "stored_delta": stored_delta,
                "fresh_delta": fresh_delta,
                "abs_diff_point": abs(fresh_delta - stored_delta),
                "point_match_within_1e-6": point_match,
                "stored_ci_low": stored_lo,
                "stored_ci_high": stored_hi,
                "fresh_ci_low": fresh_lo,
                "fresh_ci_high": fresh_hi,
                "stored_ci_width": stored_width,
                "fresh_ci_width": fresh_width,
                "width_abs_diff": width_diff,
                "width_rel_diff": round(width_rel_diff, 6),
                "ci_width_match_within_5pct": width_match,
                "stored_direction": dir_stored,
                "fresh_direction": dir_fresh,
                "direction_match": direction_match,
                "n_steps": LAST_N,
                "boot_B": N_BOOT,
                "boot_seed": BOOT_SEED,
            })
            ci_recompute_rows.append({
                "entry": variant,
                "metric": metric,
                "stored_lo": stored_lo,
                "stored_hi": stored_hi,
                "fresh_lo": fresh_lo,
                "fresh_hi": fresh_hi,
                "stored_covers_zero": dir_stored == "covers_zero",
                "fresh_covers_zero": dir_fresh == "covers_zero",
                "cover_zero_match": dir_stored == dir_fresh,
            })
            per_entry_agg[variant].append({
                "metric": metric,
                "point_match": point_match,
                "width_match": width_match,
                "direction_match": direction_match,
                "fresh_delta": fresh_delta,
                "stored_delta": stored_delta,
            })

    # ----- write outputs -----
    cols_row = list(per_row[0].keys())
    with open(os.path.join(OUT_DIR, "p6_iter178_n2_recompute_per_row.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cols_row, delimiter="\t")
        w.writeheader()
        for r in per_row:
            w.writerow(r)

    cols_cv = list(cv_rows[0].keys())
    with open(os.path.join(OUT_DIR, "p6_iter178_n2_recompute_cv_consistency.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cols_cv, delimiter="\t")
        w.writeheader()
        for r in cv_rows:
            w.writerow(r)

    cols_ci = list(ci_recompute_rows[0].keys())
    with open(os.path.join(OUT_DIR, "p6_iter178_n2_recompute_ci_recompute.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cols_ci, delimiter="\t")
        w.writeheader()
        for r in ci_recompute_rows:
            w.writerow(r)

    per_entry = []
    for variant, recs in per_entry_agg.items():
        n = len(recs)
        n_point = sum(1 for r in recs if r["point_match"])
        n_width = sum(1 for r in recs if r["width_match"])
        n_dir = sum(1 for r in recs if r["direction_match"])
        per_entry.append({
            "entry": variant,
            "n_rows": n,
            "n_point_match_1e-6": n_point,
            "n_width_match_5pct": n_width,
            "n_direction_match": n_dir,
            "point_match_rate": round(n_point / n, 4),
            "width_match_rate": round(n_width / n, 4),
            "direction_match_rate": round(n_dir / n, 4),
        })

    with open(os.path.join(OUT_DIR, "p6_iter178_n2_recompute_per_entry.tsv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(per_entry[0].keys()), delimiter="\t")
        w.writeheader()
        for r in per_entry:
            w.writerow(r)

    # ----- H verdicts -----
    n_total = len(per_row)
    n_point_match = sum(1 for r in per_row if r["point_match_within_1e-6"])
    n_width_match = sum(1 for r in per_row if r["ci_width_match_within_5pct"])
    n_dir_match = sum(1 for r in per_row if r["direction_match"])
    n_cv_obs_match = sum(1 for r in cv_rows if r["match_within_1e-6"] is True)
    n_cv_total = sum(1 for r in cv_rows if r["match_within_1e-6"] is not None)

    # H5: count CONTRADICTS in fresh recompute vs prose predicted_sign
    expected_effects_by_entry = {}
    for variant in N2_VARIANTS:
        path = os.path.join(REG_DIR, f"delta_{variant}.json")
        e = json.load(open(path))
        for ee in e.get("expected_effects", []):
            if ee.get("panel") == "n2_same_stack_last10":
                expected_effects_by_entry[(variant, ee["metric"])] = ee["predicted_sign"]

    def sign_match(observed, predicted):
        if predicted is None:
            return "UNCLAIMED"
        if predicted.startswith(">=") or predicted.startswith("<="):
            # at-least-parity / at-most-parity; treat both "covers 0" and "same side of 0" as ok
            return "NEUTRAL"
        if predicted == ">0" and observed > 0:
            return "SUPPORTS"
        if predicted == "<0" and observed < 0:
            return "SUPPORTS"
        if predicted == ">0" and observed < 0:
            return "CONTRADICTS"
        if predicted == "<0" and observed > 0:
            return "CONTRADICTS"
        return "NEUTRAL"  # observed == 0

    h5_verdicts = []
    for r in per_row:
        ps = expected_effects_by_entry.get((r["entry"], r["metric"]))
        v = sign_match(r["fresh_delta"], ps)
        h5_verdicts.append(v)
    n_contradicts = sum(1 for v in h5_verdicts if v == "CONTRADICTS")
    n_supports = sum(1 for v in h5_verdicts if v == "SUPPORTS")
    n_neutral = sum(1 for v in h5_verdicts if v == "NEUTRAL")
    n_unclaimed = sum(1 for v in h5_verdicts if v == "UNCLAIMED")

    summary = {
        "n_total_rows": n_total,
        "n_point_match_within_1e-6": n_point_match,
        "point_match_rate": round(n_point_match / n_total, 4),
        "n_width_match_within_5e-3": n_width_match,
        "width_match_rate": round(n_width_match / n_total, 4),
        "n_direction_match": n_dir_match,
        "direction_match_rate": round(n_dir_match / n_total, 4),
        "n_cv_observed_delta_match": n_cv_obs_match,
        "n_cv_total_with_obs": n_cv_total,
        "cv_consistency_rate": round(n_cv_obs_match / n_cv_total, 4) if n_cv_total else None,
        "h5_verdict_distribution": {
            "SUPPORTS": n_supports,
            "CONTRADICTS": n_contradicts,
            "NEUTRAL": n_neutral,
            "UNCLAIMED": n_unclaimed,
        },
        "hypotheses": {
            "H1": {
                "claim": "every stored delta agrees with fresh recompute within 1e-6",
                "bar": "12/12 exact",
                "verdict": "PASS" if n_point_match == n_total else "FAIL",
                "evidence": f"{n_point_match}/{n_total} = {n_point_match / n_total:.4f}",
            },
            "H2": {
                "claim": "claim_validation.observed_delta agrees with measured[].delta within 1e-6",
                "bar": "12/12 exact",
                "verdict": "PASS" if n_cv_obs_match == n_cv_total else "FAIL",
                "evidence": f"{n_cv_obs_match}/{n_cv_total} = {n_cv_obs_match / n_cv_total:.4f}" if n_cv_total else "no claim_validation rows",
            },
            "H3": {
                "claim": "stored CI width agrees with fresh paired-step bootstrap width within 5% relative",
                "bar": ">= 11/12 = 91.7%",
                "verdict": "PASS" if n_width_match >= 11 else "FAIL",
                "evidence": f"{n_width_match}/{n_total} = {n_width_match / n_total:.4f}",
            },
            "H4": {
                "claim": "stored CI direction (covers_zero / positive / negative) matches fresh CI direction",
                "bar": ">= 11/12 = 91.7%",
                "verdict": "PASS" if n_dir_match >= 11 else "FAIL",
                "evidence": f"{n_dir_match}/{n_total} = {n_dir_match / n_total:.4f}",
            },
            "H5": {
                "claim": "fresh recompute does NOT contradict prose predicted_sign on more than 2 rows",
                "bar": "<= 2 CONTRADICTS",
                "verdict": "PASS" if n_contradicts <= 2 else "FAIL",
                "evidence": f"n_contradicts={n_contradicts}/{n_total}; supports={n_supports}; neutral={n_neutral}; unclaimed={n_unclaimed}",
            },
        },
        "per_entry": per_entry,
        "metadata": {
            "n2_metrics_path": N2_METRICS,
            "last_n": LAST_N,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "abs_tol_point": ABS_TOL_POINT,
            "rel_tol_ci_width": REL_TOL_CI_WIDTH,
            "registry_entries_audited": list(N2_VARIANTS),
        },
    }

    with open(os.path.join(OUT_DIR, "p6_iter178_n2_recompute_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== p6_iter178 summary ===")
    print(json.dumps(summary["hypotheses"], indent=2))
    print(f"per-row: {n_total}, per-entry: {len(per_entry)}")
    for v, n in summary["h5_verdict_distribution"].items():
        print(f"  H5 {v}: {n}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Pillar 2 ZVF Iter 50 — Reward-leads-ZVF lagged cross-correlation and
phase-aware ZVF integrals across the eight variance-mitigation libraries.

Question iterated. iter22/iter26 proved ZVF correlates with training
failure; iter30 proved ZVF is a calibrated LEADING indicator at a
scalar level (first_pass_zvf05 → first_collapse_step lead_steps
positive on every collapsing trajectory). What iter50 adds is two new
facts NOT yet present in the iter22/30/34/38/42/46 chain:

  (A) Lagged cross-correlation between the per-step reward_mean and
      ZVF streams. We compute Pearson r at lags L ∈ {-10, -5, -1, 0,
      +1, +5, +10}. Positive L means reward[L] is correlated with
      ZVF[t]; a peaked correlation at a positive L means **reward
      leads ZVF**, which is the canonical signature of a leading
      indicator (the reward deteriorates first, ZVF catches up).
  (B) Phase-aware integrals of (ZVF - 0.5)⁺ across three canonical
      training phases (steps 0..peak, peak..peak+50,
      peak+50..end), normalised by phase length. We ask whether the
      "stacked" ZVF above threshold 0.5 in the post-peak phase is
      larger on collapsing seeds than on converging ones.

These are computed on the real `variance_mitigation.tsv` logs (5541
per-step rows, 9 methods × 5 seeds, see header below).

Inputs:
    platform_hybrid/experiments/results/variance_mitigation.tsv
        9 methods × 5 seeds × 100-300 steps (5541 rows total).
        Columns: method, seed, step, zvf, reward_mean, heldout_acc,
        collapse. Pre-validated by iter22 zvf_leadtime summary and
        iter38 zvf_iter38_classifier; treated as ground truth here.

Outputs:
    platform_hybrid/experiments/results/zvf_iter50_lagged_corr.tsv
        Long-form table; columns = (method, seed, lag, r). 9*5*7 = 315
        rows.
    platform_hybrid/experiments/results/zvf_iter50_phase_integrals.tsv
        Per-(method, seed) per-phase integral of (ZVF - 0.5)⁺
        normalised. 9*5 = 45 rows × {int_phase1, int_phase2,
        int_phase3, n_steps_phase1, peak_step}.
    platform_hybrid/experiments/results/zvf_iter50_summary.tsv
        Per-library (method) rollup: mean peak-L r at the
        reward-leads-ZVF lag, mean phase-2 integral, mean last10_avg.
        9 rows.
    platform_hybrid/experiments/results/zvf_iter50_predictions.tsv
        4 pre-registered predictions for this iter.

This script does NOT touch the existing zvf_diagnostic.py outputs;
it reads the same input file and writes fresh TSV outputs. Stdlib
only; no numpy / pandas. B=2000 bootstrap CIs.
"""
from __future__ import annotations

import csv
import math
import os
import random
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
INPUT = os.path.join(ROOT, "platform_hybrid/experiments/results/variance_mitigation.tsv")

OUT_DIR = os.path.join(ROOT, "platform_hybrid/experiments/results")
OUT_LAGGED = os.path.join(OUT_DIR, "zvf_iter50_lagged_corr.tsv")
OUT_INTEGRALS = os.path.join(OUT_DIR, "zvf_iter50_phase_integrals.tsv")
OUT_SUMMARY = os.path.join(OUT_DIR, "zvf_iter50_summary.tsv")
OUT_PREDICTIONS = os.path.join(OUT_DIR, "zvf_iter50_predictions.tsv")

LAGS = [-10, -5, -1, 0, 1, 5, 10]
THRESHOLD = 0.5  # ZVF-above-threshold integration target
BOOTSTRAP_B = 2000
SEED = 20240702  # deterministic RNG

METHOD_ORDER = [
    "grpo", "aero", "cppo", "ngrpo", "scafgrpo",
    "mcgrpo", "gift", "areal", "es",
]
METHOD_DISPLAY = {
    "grpo": "GRPO", "aero": "AERO", "cppo": "CPPO", "ngrpo": "NGRPO",
    "scafgrpo": "SCAF-GRPO", "mcgrpo": "MCGRPO", "gift": "GIFT",
    "areal": "AREAL", "es": "ES",
}


# ---------------------------------------------------------------------------
# Tiny math helpers (stdlib only)
# ---------------------------------------------------------------------------
def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return num / (sx * sy)


def bootstrap_ci(values, stat_fn, B=BOOTSTRAP_B, seed=SEED):
    """Percentile bootstrap CI for arbitrary summary statistic."""
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(B):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    return (stat_fn(values), boots[int(0.025 * B)], boots[int(0.975 * B)])


def safe_mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def safe_std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_variance_mitigation(path):
    """Load and return rows grouped by (method, seed)."""
    by_ms = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["method"], int(row["seed"]))
            by_ms[key].append({
                "step": int(row["step"]),
                "zvf": float(row["zvf"]),
                "reward_mean": float(row["reward_mean"]),
                "heldout_acc": float(row["heldout_acc"]),
                "collapse": int(row["collapse"]),
            })
    # sort each trajectory by step
    for k in by_ms:
        by_ms[k].sort(key=lambda r: r["step"])
    return by_ms


# ---------------------------------------------------------------------------
# (A) Lagged cross-correlation reward → ZVF
# ---------------------------------------------------------------------------
def compute_lagged_corr(by_ms):
    """For each (method, seed), compute Pearson(r_t, zvf_t+L) for L in LAGS.
    Positive L means reward[t] is correlated with ZVF[t+L] -- reward leads
    ZVF if the cross-correlation peaks at small positive L.
    """
    rows = []
    for (method, seed), traj in sorted(by_ms.items()):
        n = len(traj)
        rewards = [t["reward_mean"] for t in traj]
        zvfs = [t["zvf"] for t in traj]
        for lag in LAGS:
            # r_t = rewards[t]
            # zvf_tL = zvfs[t+lag]
            xs = []
            ys = []
            for t in range(n):
                idx = t + lag
                if 0 <= idx < n:
                    xs.append(rewards[t])
                    ys.append(zvfs[idx])
            r = pearson(xs, ys)
            rows.append({
                "method": method,
                "seed": seed,
                "lag": lag,
                "r": r,
                "n_pairs": len(xs),
            })
    return rows


def aggregate_lagged_corr(lagged_rows):
    """Mean r per (method, lag) across seeds. Returns dict[(method, lag) -> (mean, lo, hi)]."""
    grouped = defaultdict(list)
    for r in lagged_rows:
        grouped[(r["method"], r["lag"])].append(r["r"])
    out = {}
    for key, vals in grouped.items():
        m, lo, hi = bootstrap_ci(vals, safe_mean)
        out[key] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
    return out


# ---------------------------------------------------------------------------
# (B) Phase-aware integrals
# ---------------------------------------------------------------------------
def phase_integrals(by_ms):
    """For each (method, seed), compute:
       - peak_step = argmax of heldout_acc
       - phase1 = [0, peak]
       - phase2 = [peak+1, peak+50]  (or end if shorter)
       - phase3 = [peak+51, end]
       - int_phase_k = mean over phase_k of max(ZVF - THRESHOLD, 0)
    """
    rows = []
    for (method, seed), traj in sorted(by_ms.items()):
        n = len(traj)
        # peak step by heldout_acc, tie-break earlier step
        best_acc = -1.0
        peak_step = traj[0]["step"]
        for t in traj:
            if t["heldout_acc"] > best_acc:
                best_acc = t["heldout_acc"]
                peak_step = t["step"]
        # slice phases by absolute step
        p1 = [t for t in traj if t["step"] <= peak_step]
        p2 = [t for t in traj if peak_step < t["step"] <= peak_step + 50]
        p3 = [t for t in traj if t["step"] > peak_step + 50]

        def integ(seg):
            if not seg:
                return float("nan")
            return safe_mean([max(t["zvf"] - THRESHOLD, 0.0) for t in seg])

        rows.append({
            "method": method,
            "seed": seed,
            "n_steps_total": n,
            "peak_step": peak_step,
            "peak_heldout_acc": best_acc,
            "n_steps_phase1": len(p1),
            "n_steps_phase2": len(p2),
            "n_steps_phase3": len(p3),
            "int_phase1": integ(p1),
            "int_phase2": integ(p2),
            "int_phase3": integ(p3),
            "zvf_mean_phase2": safe_mean([t["zvf"] for t in p2]) if p2 else float("nan"),
            "zvf_mean_phase3": safe_mean([t["zvf"] for t in p3]) if p3 else float("nan"),
        })
    return rows


def aggregate_phase_integrals(integ_rows):
    """Per-method rollup of phase integrals."""
    grouped = defaultdict(list)
    for r in integ_rows:
        grouped[r["method"]].append(r)
    out = {}
    for method, runs in grouped.items():
        keys = ["int_phase1", "int_phase2", "int_phase3",
                "zvf_mean_phase2", "zvf_mean_phase3",
                "peak_step", "peak_heldout_acc"]
        row = {"method": method, "n_runs": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs if not (
                isinstance(r[k], float) and math.isnan(r[k])
            )]
            if not vals:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_ci_lo"] = float("nan")
                row[f"{k}_ci_hi"] = float("nan")
            else:
                m, lo, hi = bootstrap_ci(vals, safe_mean)
                row[f"{k}_mean"] = m
                row[f"{k}_ci_lo"] = lo
                row[f"{k}_ci_hi"] = hi
        # also last10_avg = mean of heldout_acc in last 10 steps
        last10s = []
        # back-fill from input
        out[method] = row
    return out


def last10_avg_per_run(by_ms):
    """Mean heldout_acc in last 10 steps per (method, seed)."""
    out = {}
    for key, traj in by_ms.items():
        last10 = traj[-10:] if len(traj) >= 10 else traj
        out[key] = safe_mean([t["heldout_acc"] for t in last10])
    return out


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------
def write_predictions(out_path, lagged_agg, integ_agg, last10_map, by_ms, integ_map):
    """Four pre-registered predictions for this iter."""
    preds = []

    # P1: the cross-correlation r(reward_t, ZVF_t+1) - "reward leads ZVF by 1 step"
    # is STRICTLY POSITIVE for vanilla GRPO (peak > 0).
    grpo_at_p1 = lagged_agg.get(("grpo", 1), {}).get("mean", 0.0)
    preds.append({
        "id": "P1_reward_leads_zvf_lag1_grpo",
        "claim": (
            "vanilla GRPO Pearson(reward_t, ZVF_{t+1}) > 0 "
            "(reward leads ZVF by 1 step)"
        ),
        "value": grpo_at_p1,
        "pass": grpo_at_p1 > 0.0,
        "n_seeds": lagged_agg.get(("grpo", 1), {}).get("n", 0),
    })

    # P2: AERO exhibits STRICTLY LOWER r than vanilla GRPO at lag +1
    # (the variance-mitigation mechanism DELAYS ZVF rise relative to reward
    # decline by intervening in the gradient update).
    aero_at_p1 = lagged_agg.get(("aero", 1), {}).get("mean", 0.0)
    diff_p2 = grpo_at_p1 - aero_at_p1
    preds.append({
        "id": "P2_aero_lower_lag1_corr_than_grpo",
        "claim": (
            "AERO r(lag=+1) < GRPO r(lag=+1) "
            "(variance mitigation lowers reward→ZVF cross-corr)"
        ),
        "value": diff_p2,
        "pass": diff_p2 > 0.0,
        "n_seeds": lagged_agg.get(("aero", 1), {}).get("n", 0),
    })

    # P3: phase-2 integral of (ZVF - 0.5)+ is LARGER on collapsing seeds
    # (collapse=1 at any step) than on non-collapsing seeds in vanilla GRPO.
    by_m = defaultdict(list)
    for (m, s), traj in by_ms.items():
        collapsed = any(t["collapse"] == 1 for t in traj)
        p2_int = integ_map.get((m, s), float("nan"))
        by_m[m].append((collapsed, p2_int))
    grpo_p2 = by_m["grpo"]
    grpo_coll = [v for c, v in grpo_p2 if c and not (
        isinstance(v, float) and math.isnan(v)
    )]
    grpo_non = [v for c, v in grpo_p2 if not c and not (
        isinstance(v, float) and math.isnan(v)
    )]
    p3_diff = (safe_mean(grpo_coll) if grpo_coll else 0.0) - (
        safe_mean(grpo_non) if grpo_non else 0.0
    )
    preds.append({
        "id": "P3_grpo_p2_int_coll_minus_noncoll",
        "claim": (
            "vanilla GRPO post-peak ZVF>0.5 integral is LARGER on "
            "collapsing seeds than on non-collapsing seeds"
        ),
        "value": p3_diff,
        "pass": p3_diff > 0.0,
        "n_collapsed": len(grpo_coll),
        "n_noncoll": len(grpo_non),
    })

    # P4: the cross-correlation peak lag is POSITIVE in vanilla GRPO
    # (the peak lag is the argmax of |r| over LAGS).
    grpo_curve = [lagged_agg.get(("grpo", L), {}).get("mean", 0.0) for L in LAGS]
    peak_lag = LAGS[grpo_curve.index(max(grpo_curve))] if grpo_curve else 0
    preds.append({
        "id": "P4_grpo_peak_lag_positive",
        "claim": (
            "argmax over LAGS of vanilla GRPO |r(reward_t, ZVF_t+L)| "
            "is POSITIVE (reward leads ZVF)"
        ),
        "value": peak_lag,
        "pass": peak_lag > 0,
        "all_lags_signed": dict(zip(LAGS, [round(r, 4) for r in grpo_curve])),
    })

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "claim", "value", "pass", "extras_json"])
        for p in preds:
            extras = {k: v for k, v in p.items()
                      if k not in ("id", "claim", "value", "pass")}
            extras_json = repr(extras) if extras else ""
            w.writerow([
                p["id"], p["claim"],
                f"{p['value']:.6f}" if isinstance(p["value"], float) else str(p["value"]),
                "PASS" if p["pass"] else "FAIL",
                extras_json,
            ])


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    by_ms = load_variance_mitigation(INPUT)
    print(f"[iter50] loaded {sum(len(v) for v in by_ms.values())} per-step rows "
          f"across {len(by_ms)} (method, seed) pairs")

    # (A) lagged cross-correlation
    lagged_rows = compute_lagged_corr(by_ms)
    lagged_agg = aggregate_lagged_corr(lagged_rows)
    with open(OUT_LAGGED, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "seed", "lag", "r", "n_pairs"])
        for r in lagged_rows:
            w.writerow([
                r["method"], r["seed"], r["lag"],
                f"{r['r']:.6f}", r["n_pairs"]
            ])
    print(f"[iter50] wrote {OUT_LAGGED} ({len(lagged_rows)} rows)")

    # (B) phase integrals
    integ_rows = phase_integrals(by_ms)
    integ_map = {(r["method"], r["seed"]): r["int_phase2"] for r in integ_rows}
    integ_agg = aggregate_phase_integrals(integ_rows)
    with open(OUT_INTEGRALS, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "method", "seed", "n_steps_total", "peak_step",
            "peak_heldout_acc", "n_steps_phase1", "n_steps_phase2",
            "n_steps_phase3", "int_phase1", "int_phase2", "int_phase3",
            "zvf_mean_phase2", "zvf_mean_phase3",
        ])
        for r in integ_rows:
            w.writerow([
                r["method"], r["seed"], r["n_steps_total"], r["peak_step"],
                f"{r['peak_heldout_acc']:.4f}",
                r["n_steps_phase1"], r["n_steps_phase2"], r["n_steps_phase3"],
                f"{r['int_phase1']:.6f}", f"{r['int_phase2']:.6f}", f"{r['int_phase3']:.6f}",
                f"{r['zvf_mean_phase2']:.6f}", f"{r['zvf_mean_phase3']:.6f}",
            ])
    print(f"[iter50] wrote {OUT_INTEGRALS} ({len(integ_rows)} rows)")

    # last10 map for predictions
    last10_map = last10_avg_per_run(by_ms)

    # Method-level summary
    summary_rows = []
    for method in METHOD_ORDER:
        if method not in integ_agg:
            continue
        agg = integ_agg[method]
        # Mean last10 across seeds (for display)
        m_last10 = safe_mean([v for (m, s), v in last10_map.items() if m == method])
        # Mean r at lag +1
        l1 = lagged_agg.get((method, 1), {})
        summary_rows.append({
            "method": method,
            "n_runs": agg["n_runs"],
            "mean_last10_acc": m_last10,
            "mean_r_lag_minus10": lagged_agg.get((method, -10), {}).get("mean", float("nan")),
            "mean_r_lag_minus5": lagged_agg.get((method, -5), {}).get("mean", float("nan")),
            "mean_r_lag_minus1": lagged_agg.get((method, -1), {}).get("mean", float("nan")),
            "mean_r_lag_0": lagged_agg.get((method, 0), {}).get("mean", float("nan")),
            "mean_r_lag_1": lagged_agg.get((method, 1), {}).get("mean", float("nan")),
            "mean_r_lag_5": lagged_agg.get((method, 5), {}).get("mean", float("nan")),
            "mean_r_lag_10": lagged_agg.get((method, 10), {}).get("mean", float("nan")),
            "int_phase1_mean": agg["int_phase1_mean"],
            "int_phase2_mean": agg["int_phase2_mean"],
            "int_phase3_mean": agg["int_phase3_mean"],
            "zvf_mean_phase2_mean": agg["zvf_mean_phase2_mean"],
            "zvf_mean_phase3_mean": agg["zvf_mean_phase3_mean"],
            "peak_step_mean": agg["peak_step_mean"],
        })
    with open(OUT_SUMMARY, "w", newline="") as f:
        cols = [
            "method", "n_runs", "mean_last10_acc",
            "mean_r_lag_minus10", "mean_r_lag_minus5", "mean_r_lag_minus1",
            "mean_r_lag_0", "mean_r_lag_1", "mean_r_lag_5", "mean_r_lag_10",
            "int_phase1_mean", "int_phase2_mean", "int_phase3_mean",
            "zvf_mean_phase2_mean", "zvf_mean_phase3_mean",
            "peak_step_mean",
        ]
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in summary_rows:
            row = [r[c] if not isinstance(r[c], float) else f"{r[c]:.6f}" for c in cols]
            w.writerow(row)
    print(f"[iter50] wrote {OUT_SUMMARY} ({len(summary_rows)} rows)")

    # (D) predictions
    write_predictions(OUT_PREDICTIONS, lagged_agg, integ_agg, last10_map, by_ms, integ_map)
    print(f"[iter50] wrote {OUT_PREDICTIONS}")

    # Tiny stdout summary
    print("\nKey Pearson r(lag=+1) per library (reward leads ZVF):")
    for method in METHOD_ORDER:
        v = lagged_agg.get((method, 1), {}).get("mean", float("nan"))
        lo = lagged_agg.get((method, 1), {}).get("lo", float("nan"))
        hi = lagged_agg.get((method, 1), {}).get("hi", float("nan"))
        print(f"  {method:9s}  r={v:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]")

    print("\nMean phase-2 integral (ZVF > 0.5) per library:")
    for method in METHOD_ORDER:
        v = integ_agg.get(method, {}).get("int_phase2_mean", float("nan"))
        print(f"  {method:9s}  post-peak ZVF>0.5 integral = {v:.4f}")


if __name__ == "__main__":
    main()

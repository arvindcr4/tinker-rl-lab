#!/usr/bin/env python3
"""Iter 108 -- Pillar 4 (Length Bias / Dr.GRPO): progress-window + L-quantile
Conditional ICCA.

After iter96 (global ICCA), iter100 (bivariate VAR), iter104 (R-quantile
C-ICCA-Q), iter108 closes the spatial-temporal decomposition:

  Q1 (WHEN).  Slice each run into n_w=4 equal-length training-progress
      windows; re-fit AR(1) to L,R within each window; backward |CCF|
      over k in {-3,-2,-1}.  Paired Dr.GR - GR delta per window.
      Spearman rho(delta, window_index): negative = "severship grows
      with progress".

  Q2 (WHERE in L).  Mirror iter104 but bin time-steps by RANK of L
      (not R).  Same AR(1)+CCF machinery.  log2(bwd_share_q4 / q0)
      per (task, algo).  A flip = "Dr.GRPO redirects the backward
      channel away from the high-L regime".

INPUTS : experiments/results/drgrpo_vs_grpo.json
         experiments/results/drgrpo_gsm8k_cot_full.json
OUTPUTS: 6 TSVs + meta under experiments/results/length_bias_iter108_*
USAGE  : python3 platform_modal/scripts/length_bias_iter108.py [--n_w 4 --n_q 5 --K 3 --B 10000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")

DRGRPO_VS_GRPO_PATH = os.path.join(RESULTS, "drgrpo_vs_grpo.json")
DRGRPO_GSM8K_PATH = os.path.join(RESULTS, "drgrpo_gsm8k_cot_full.json")


def load_step_log(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        d = json.load(fh)
    runs_raw = d["runs"]
    out = []
    for r in runs_raw:
        step_log = r.get("step_log") or []
        if len(step_log) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in step_log], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in step_log], dtype=np.float64)
        out.append({"algo": r["algo"], "seed": r["seed"], "n": int(len(step_log)),
                    "L": L, "R": R})
    return out


def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit x_t = phi*x_{t-1} + c + e_t  and return (phi, c, e_t) with len n-1."""
    x_lag = x[:-1]
    x_cur = x[1:]
    X = np.column_stack([x_lag, np.ones_like(x_lag)])
    coef, *_ = np.linalg.lstsq(X, x_cur, rcond=None)
    phi, c = float(coef[0]), float(coef[1])
    e = x_cur - (phi * x_lag + c)
    return phi, c, e


def ccf_at_lags(e_a: np.ndarray, e_b: np.ndarray, K: int) -> np.ndarray:
    """CCF for lags -K..+K; length 2K+1.  Falls back to zeros for tiny bins."""
    n = min(len(e_a), len(e_b))
    a = e_a[:n] - e_a[:n].mean()
    b = e_b[:n] - e_b[:n].mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum()) + 1e-300
    out = np.zeros(2 * K + 1, dtype=np.float64)
    for i, k in enumerate(range(-K, K + 1)):
        if k >= 0:
            x = a[:n - k]
            y = b[k:n]
        else:
            x = a[-k:n]
            y = b[:n + k]
        if len(x) < 3:
            out[i] = 0.0
            continue
        out[i] = float(np.dot(x, y) / denom)
    return out


def paired_bootstrap_delta(g, d, B, statistic=np.median):
    g = np.array(g, dtype=np.float64); d = np.array(d, dtype=np.float64)
    n = min(len(g), len(d))
    if n == 0:
        return {"delta": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"), "n": 0}
    g = g[:n]; d = d[:n]
    diffs = d - g
    rng = np.random.default_rng(0x10C8)
    idx = rng.integers(0, n, size=(B, n))
    boot = statistic(diffs[idx], axis=1)
    point = float(statistic(diffs))
    return {"delta": point,
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "p": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
            "n": int(n)}


def write_tsv(name: str, rows: list[dict], keys: list[str]) -> str:
    out = os.path.join(RESULTS, name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[iter108] wrote {out}  ({len(rows)} rows)")
    return out


def paired_seed_slices(by_algo: dict) -> tuple[list, dict, dict]:
    """Return (common_seeds, seeds_g, seeds_d) for paired per-seed stats."""
    seeds_g = {r["seed"]: r for r in by_algo["grpo"]}
    seeds_d = {r["seed"]: r for r in by_algo["dr_grpo"]}
    return sorted(set(seeds_g) & set(seeds_d)), seeds_g, seeds_d


# ===========================================================================
#   Progress-window ICCA
# ===========================================================================

def progress_window_icca(L: np.ndarray, R: np.ndarray, K: int,
                         n_w: int) -> list[dict[str, Any]]:
    """Slice into n_w equal-length windows, fit AR(1)+CCF within each."""
    n = len(L)
    edges = [int(np.floor(n * w / n_w)) for w in range(n_w + 1)]
    for w in range(1, n_w + 1):
        edges[w] = max(edges[w], edges[w - 1] + 4)
        edges[w] = min(edges[w], n)
    empty = lambda w: {"window": w, "n_in_window": 0, "phi_L": 0.0,
                       "phi_R": 0.0, "bwd": 0.0, "fwd": 0.0,
                       "bwd_signed": 0.0, "fwd_signed": 0.0,
                       "ccf": [0.0] * (2 * K + 1)}
    out = []
    for w in range(n_w):
        lo, hi = edges[w], edges[w + 1]
        if hi - lo < 5:
            out.append({**empty(w), "n_in_window": int(hi - lo)})
            continue
        L_w, R_w = L[lo:hi], R[lo:hi]
        _, _, e_L = fit_ar1(L_w); _, _, e_R = fit_ar1(R_w)
        cc = ccf_at_lags(e_L, e_R, K=K)
        out.append({
            "window": w, "n_in_window": int(hi - lo),
            "phi_L": float(fit_ar1(L_w)[0]),
            "phi_R": float(fit_ar1(R_w)[0]),
            "bwd": float(np.sum(np.abs(cc[:K]))),
            "fwd": float(np.sum(np.abs(cc[K + 1:]))),
            "bwd_signed": float(np.sum(cc[:K])),
            "fwd_signed": float(np.sum(cc[K + 1:])),
            "ccf": [float(cc[i]) for i in range(2 * K + 1)],
        })
    return out


# ===========================================================================
#   Length-quantile ICCA (mirror of iter104)
# ===========================================================================

def length_quantile_icca(L: np.ndarray, R: np.ndarray, K: int,
                         n_q: int) -> dict[str, Any]:
    """Mirror of iter104: bin time-steps by RANK of L (not R)."""
    _, _, e_L = fit_ar1(L); _, _, e_R = fit_ar1(R)
    L_for_bin = L[1:]
    n = len(L_for_bin)
    if np.std(L_for_bin) < 1e-12:
        bins = np.zeros(n, dtype=int)
    else:
        edges = np.quantile(L_for_bin, np.linspace(0, 1, n_q + 1))
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        bins = np.digitize(L_for_bin, edges[1:-1])
    out: dict[str, Any] = {"q": {}}
    for q_idx in range(n_q):
        mask = bins == q_idx
        n_in = int(mask.sum())
        if n_in < 4:
            out["q"][q_idx] = {"size": n_in, "bwd": 0.0, "fwd": 0.0,
                               "bwd_signed": 0.0, "fwd_signed": 0.0,
                               "ccf": [0.0] * (2 * K + 1)}
            continue
        cc = ccf_at_lags(e_L[mask], e_R[mask], K=K)
        out["q"][q_idx] = {
            "size": n_in,
            "bwd": float(np.sum(np.abs(cc[:K]))),
            "fwd": float(np.sum(np.abs(cc[K + 1:]))),
            "bwd_signed": float(np.sum(cc[:K])),
            "fwd_signed": float(np.sum(cc[K + 1:])),
            "ccf": [float(cc[i]) for i in range(2 * K + 1)],
        }
    return out


# ===========================================================================
#   Main
# ===========================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iter108 -- progress+length ICCA")
    parser.add_argument("--n_w", type=int, default=4)
    parser.add_argument("--n_q", type=int, default=5)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--seed_base", type=int, default=20260703)
    args = parser.parse_args(argv)

    n_w, n_q, K, B = int(args.n_w), int(args.n_q), int(args.K), int(args.B)
    seed_base = int(args.seed_base)

    runs = ([(r, "arithmetic_easy") for r in load_step_log(DRGRPO_VS_GRPO_PATH)]
            + [(r, "gsm8k_cot") for r in load_step_log(DRGRPO_GSM8K_PATH)])

    # ---------------------------------------------------------------------
    # Q1.  Progress-window ICCA
    # ---------------------------------------------------------------------
    progress_perrun: list[dict[str, Any]] = []
    progress_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r, task in runs:
        recs = progress_window_icca(r["L"], r["R"], K=K, n_w=n_w)
        for wrec in recs:
            progress_perrun.append({"task": task, "algo": r["algo"],
                                    "seed": int(r["seed"]),
"n_total": int(r["n"]), **wrec})
        progress_by_task.setdefault(task, {"grpo": [], "dr_grpo": []})[r["algo"]].append(
            {"seed": int(r["seed"]), "n": int(r["n"]), "windows": recs}
        )

    write_tsv("length_bias_iter108_perrun_progress.tsv", progress_perrun,
              ["task", "algo", "seed", "n_total", "window", "n_in_window",
               "phi_L", "phi_R", "bwd", "fwd", "bwd_signed", "fwd_signed"])

    # paired Dr.GR - GR delta per (task, window, side)
    paired_prog_rows = []
    sides = ["bwd", "fwd", "bwd_signed", "fwd_signed"]
    for task, by_algo in progress_by_task.items():
        common, seeds_g, seeds_d = paired_seed_slices(by_algo)
        for w_idx in range(n_w):
            for side in sides:
                g = [float(seeds_g[s]["windows"][w_idx][side]) for s in common]
                d = [float(seeds_d[s]["windows"][w_idx][side]) for s in common]
                paired_prog_rows.append({"task": task, "window": w_idx,
                                          "side": side,
                                          **paired_bootstrap_delta(g, d, B=B)})
    write_tsv("length_bias_iter108_paired_progress.tsv", paired_prog_rows,
              ["task", "window", "side", "delta", "ci_lo", "ci_hi", "p", "n"])

    # trend (Spearman) of the per-window |CCF| delta across window index
    trend_prog_rows = []
    win_idx = np.arange(1, n_w + 1)
    for task, by_algo in progress_by_task.items():
        common, seeds_g, seeds_d = paired_seed_slices(by_algo)
        for side in sides:
            gr_vals = np.array([np.mean([seeds_g[s]["windows"][w][side]
                                          for s in common]) for w in range(n_w)])
            dr_vals = np.array([np.mean([seeds_d[s]["windows"][w][side]
                                          for s in common]) for w in range(n_w)])
            sp_d, sp_g, sp_r = (stats.spearmanr(win_idx, dr_vals - gr_vals),
                                 stats.spearmanr(win_idx, gr_vals),
                                 stats.spearmanr(win_idx, dr_vals))
            trend_prog_rows.append({
                "task": task, "side": side,
                "vals_gr_q0": float(gr_vals[0]), "vals_gr_q_last": float(gr_vals[-1]),
                "vals_dr_q0": float(dr_vals[0]), "vals_dr_q_last": float(dr_vals[-1]),
                "delta_q0": float(dr_vals[0] - gr_vals[0]),
                "delta_q_last": float(dr_vals[-1] - gr_vals[-1]),
                "spearman_rho_delta_vs_window": float(sp_d.statistic),
                "spearman_p_delta_vs_window": float(sp_d.pvalue),
                "spearman_rho_grpo": float(sp_g.statistic),
                "spearman_p_grpo": float(sp_g.pvalue),
                "spearman_rho_drgrpo": float(sp_r.statistic),
                "spearman_p_drgrpo": float(sp_r.pvalue),
            })
    write_tsv("length_bias_iter108_trend_progress.tsv", trend_prog_rows,
              ["task", "side", "vals_gr_q0", "vals_gr_q_last",
               "vals_dr_q0", "vals_dr_q_last", "delta_q0", "delta_q_last",
               "spearman_rho_delta_vs_window", "spearman_p_delta_vs_window",
               "spearman_rho_grpo", "spearman_p_grpo",
               "spearman_rho_drgrpo", "spearman_p_drgrpo"])

    # ---------------------------------------------------------------------
    # Q2.  Length-quantile ICCA (mirror of iter104)
    # ---------------------------------------------------------------------
    length_perrun: list[dict[str, Any]] = []
    length_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r, task in runs:
        rec = length_quantile_icca(r["L"], r["R"], K=K, n_q=n_q)
        rec.update(task=task, algo=r["algo"], seed=int(r["seed"]), n=int(r["n"]))
        length_perrun.append(rec)
        length_by_task.setdefault(task, {"grpo": [], "dr_grpo": []})[r["algo"]].append(rec)

    len_keys = (["task", "algo", "seed", "n"]
                + [f"q{q}_{f}" for q in range(n_q)
                   for f in ("size", "bwd", "fwd", "bwd_signed", "fwd_signed")])
    len_rows = []
    for rec in length_perrun:
        row = {k: rec.get(k, "") for k in len_keys}
        for q in range(n_q):
            qq = rec["q"][q]
            for f in ("size", "bwd", "fwd", "bwd_signed", "fwd_signed"):
                row[f"q{q}_{f}"] = qq[f]
        len_rows.append(row)
    write_tsv("length_bias_iter108_summary_length.tsv", len_rows, len_keys)

    # paired Dr.GR - GR delta per (task, q, side)
    paired_len_rows = []
    for task, by_algo in length_by_task.items():
        common, seeds_g, seeds_d = paired_seed_slices(by_algo, n_q, None)
        for q in range(n_q):
            for side in ["bwd", "fwd", "bwd_signed", "fwd_signed"]:
                g = [float(seeds_g[s]["q"][q][side]) for s in common]
                d = [float(seeds_d[s]["q"][q][side]) for s in common]
                paired_len_rows.append({"task": task, "q": q, "side": side,
                                         **paired_bootstrap_delta(g, d, B=B)})
    write_tsv("length_bias_iter108_paired_length.tsv", paired_len_rows,
              ["task", "q", "side", "delta", "ci_lo", "ci_hi", "p", "n"])

    # share ratios & log2(q=4/q=0)
    trend_len_rows = []
    eps = 1e-4
    for task, by_algo in length_by_task.items():
        for algo, recs in by_algo.items():
            bwd_q = np.zeros(n_q); fwd_q = np.zeros(n_q)
            for rec in recs:
                for q in range(n_q):
                    bwd_q[q] += rec["q"][q]["bwd"]
                    fwd_q[q] += rec["q"][q]["fwd"]
            tot_b, tot_f = bwd_q.sum() + 1e-300, fwd_q.sum() + 1e-300
            trend_len_rows.append({
                "task": task, "algo": algo,
                "share_bwd_q0": float(bwd_q[0] / tot_b),
                "share_bwd_q4": float(bwd_q[-1] / tot_b),
                "share_fwd_q0": float(fwd_q[0] / tot_f),
                "share_fwd_q4": float(fwd_q[-1] / tot_f),
                "log2_bwd_q4_q0": float(np.log2((bwd_q[-1] + eps) / (bwd_q[0] + eps))),
            })
    write_tsv("length_bias_iter108_trend_length.tsv", trend_len_rows,
              ["task", "algo", "share_bwd_q0", "share_bwd_q4",
               "share_fwd_q0", "share_fwd_q4", "log2_bwd_q4_q0"])

    meta = {"iter": 108, "pillar": "P4-LengthBias",
            "K": K, "n_w": n_w, "n_q": n_q, "B": B, "seed_base": seed_base,
            "n_runs": len(runs),
            "tasks": sorted(set(t for _, t in runs)),
            "algos": sorted(set(r["algo"] for r, _ in runs))}
    out_path = os.path.join(RESULTS, "length_bias_iter108_meta.json")
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter108] wrote {out_path}")

    # Headlines (concise)
    def _head_paired(rows, label):
        print(f"\n[iter108 headline] {label}")
        for r in rows:
            print(f"  {r['task']:14s}: delta={r['delta']:+.4f} "
                  f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                  f"p={r['p']:.3f} n={r['n']}")
    _head_paired([r for r in paired_prog_rows if r["side"] == "bwd"],
                 "Progress-window bwd |CCF| (Dr.GRPO - GRPO)")
    _head_paired([r for r in paired_len_rows if r["side"] == "bwd"],
                 "Length-quantile bwd |CCF| Dr.GRPO - GRPO")
    print("\n[iter108 headline] log2(bwd share q=4 / q=0) per task/algo (L-quantile)")
    for r in trend_len_rows:
        print(f"  {r['task']:14s} {r['algo']:7s}: log2(q4/q0)={r['log2_bwd_q4_q0']:+.3f}")
    print("\n[iter108 headline] Spearman rho(delta, window_index) for bwd")
    for r in trend_prog_rows:
        if r["side"] == "bwd":
            print(f"  {r['task']:14s}: rho={r['spearman_rho_delta_vs_window']:+.3f} "
                  f"p={r['spearman_p_delta_vs_window']:.3f} "
                  f"delta[0]={r['delta_q0']:+.4f} delta[-1]={r['delta_q_last']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
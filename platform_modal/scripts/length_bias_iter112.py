#!/usr/bin/env python3
"""Iter 112 -- Pillar 4 (Length Bias / Dr.GRPO): Severship-reward coupling test.

Closes the gap between the iter108 *mechanism* finding (progressive
severship of the backward innovation coupling) and the *outcome* (Dr.GRPO
vs GRPO heldout accuracy delta).  Loads iter108's per-run progress table
(per-seed per-window Dr.GR vs GR backward |CCF|), pairs each per-seed
severship Δ_bwd with the per-seed windowed mean-reward delta ΔR_w from
the same step logs, and asks three falsifiable questions:

  Q1 (pooled Spearman).  Pool (seed, window) points per task, correlate
      Spearman( -Δ_bwd , Δ_R )  --  i.e. "more severship gives more
      reward improvement".  Sign convention: NEGATIVE Δ_bwd means Dr.GR
      is *seversing* the backward channel relative to GRPO, so we use
      -Δ_bwd as the "severship intensity" axis.

  Q2 (cumulative window).  For each task, regress cumulative-window-end
      reward delta on cumulative severship  --  Δ_R(w<=k) ~ |severship(w<=k)|.
      Tests whether severship compounds into reward, not just correlates.

  Q3 (permutation null).  Permute the (algo, seed) coupling within
      (task, window), recompute Spearman, get a null distribution of
      P_obs = (#(|rho_null| >= |rho_obs|) + 1) / (B+1).  Tests whether
      Q1's Spearman survives chance.

INPUTS :
  platform_hybrid/experiments/results/length_bias_iter108_perrun_progress.tsv
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json
OUTPUTS: 6 TSVs + meta under platform_hybrid/experiments/results/length_bias_iter112_*
USAGE  : python3 platform_modal/scripts/length_bias_iter112.py [--n_w 4 --B_perm 50000]
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
RES = os.path.join(ROOT, "experiments", "results")

ITER108_PERRUN = os.path.join(RES, "length_bias_iter108_perrun_progress.tsv")
DRGR_VS_GRPO = os.path.join(RES, "drgrpo_vs_grpo.json")
DRGR_GSM8K = os.path.join(RES, "drgrpo_gsm8k_cot_full.json")


def load_step_log(path: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        sl = r.get("step_log") or []
        if len(sl) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in sl], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in sl], dtype=np.float64)
        out.append({"algo": r["algo"], "seed": int(r["seed"]),
                    "n": int(len(sl)), "L": L, "R": R,
                    "heldout_acc": r.get("heldout_acc", float("nan"))})
    return out


def load_step_log_inverse(path: str, task_label: str) -> list[tuple]:
    """Return list of (run_dict, task_label) tuples (run_dict first)."""
    return [(r, task_label) for r in load_step_log(path)]


def window_rewards(R: np.ndarray, n_w: int) -> np.ndarray:
    """Return mean-R over n_w equal-length windows."""


def load_iter108_perrun() -> list[dict[str, Any]]:
    out = []
    with open(ITER108_PERRUN) as fh:
        hdr = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            row = dict(zip(hdr, f))
            row["window"] = int(row["window"])
            row["seed"] = int(row["seed"])
            row["n_total"] = int(row["n_total"])
            row["n_in_window"] = int(row["n_in_window"])
            for k in ("phi_L", "phi_R", "bwd", "fwd", "bwd_signed", "fwd_signed"):
                row[k] = float(row[k])
            out.append(row)
    return out


def window_rewards(R: np.ndarray, n_w: int) -> np.ndarray:
    """Return mean-R over n_w equal-length windows."""
    n = len(R)
    edges = [int(np.floor(n * w / n_w)) for w in range(n_w + 1)]
    for w in range(1, n_w + 1):
        edges[w] = max(edges[w], edges[w - 1] + 4)
        edges[w] = min(edges[w], n)
    out = np.zeros(n_w, dtype=np.float64)
    for w in range(n_w):
        out[w] = float(R[edges[w]:edges[w + 1]].mean())
    return out


def paired_per_window(perrun_iter108: list[dict], step_runs: list[tuple],
                      n_w: int) -> list[dict[str, Any]]:
    """Build per-(task, window, seed) Dr.GR - GR deltas in (bwd, R).

    step_runs: list of (task_label, run_dict).
    """
    # group iter108 perrun by (task, algo, seed) -> {window: {bwd, ...}}
    by_ts: dict[tuple, dict[int, dict[str, float]]] = {}
    for r in perrun_iter108:
        key = (r["task"], r["algo"], r["seed"])
        by_ts.setdefault(key, {})[r["window"]] = {
            "bwd": r["bwd"], "fwd": r["fwd"],
            "bwd_signed": r["bwd_signed"], "fwd_signed": r["fwd_signed"],
        }

    # group step_runs by (task, algo) -> {seed: {R, L, ...}}
    sr_by: dict[tuple, dict[int, dict]] = {}
    for run, task in step_runs:
        sr_by.setdefault((task, run["algo"]), {})[run["seed"]] = run

    out: list[dict[str, Any]] = []
    for (task, algo), seeds_map in sr_by.items():
        for seed, run in seeds_map.items():
            R_w = window_rewards(run["R"], n_w=n_w)
            for w in range(n_w):
                row_b = by_ts.get((task, algo, seed), {}).get(w)
                if row_b is None:
                    continue
                out.append({"task": task, "algo": algo, "seed": seed,
                            "window": w,
                            "bwd": float(row_b["bwd"]),
                            "fwd": float(row_b["fwd"]),
                            "R_w": float(R_w[w]),
                            "n_total": int(run["n"])})
    return out


def paired_delta_table(long_table: list[dict], tasks: list[str],
                       algos: tuple, n_w: int) -> dict:
    """Return {(task, window, seed): {'delta_bwd':..., 'delta_R':...}}."""
    by_seed: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in long_table:
        by_seed.setdefault((r["task"], r["algo"]), {}).setdefault(r["seed"], {})[
            r["window"]] = r
    common_seeds: dict[tuple, list[int]] = {}
    for task in tasks:
        s_g = set(by_seed.get((task, algos[0]), {}).keys())
        s_d = set(by_seed.get((task, algos[1]), {}).keys())
        common_seeds[(task, "pair")] = sorted(s_g & s_d)
    out = {}
    for task in tasks:
        for s in common_seeds[(task, "pair")]:
            for w in range(n_w):
                g = by_seed[(task, algos[0])][s].get(w)
                d = by_seed[(task, algos[1])][s].get(w)
                if g is None or d is None:
                    continue
                out[(task, w, s)] = {
                    "delta_bwd": float(d["bwd"] - g["bwd"]),
                    "delta_R": float(d["R_w"] - g["R_w"]),
                    "g_bwd": float(g["bwd"]),
                    "d_bwd": float(d["bwd"]),
                    "g_R": float(g["R_w"]),
                    "d_R": float(d["R_w"]),
                }
    return out, common_seeds


def spearman_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    sp = stats.spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue)


def permutation_null_spearman(x: np.ndarray, y: np.ndarray, B: int,
                              seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs, _ = spearman_pvalue(x, y)
    n = len(x)
    abs_obs = abs(obs)
    y_arr = np.array(y, dtype=np.float64)
    count = 0
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.permutation(n)
        r_b, _ = spearman_pvalue(x, y_arr[idx])
        boot[b] = r_b
        if abs(r_b) >= abs_obs:
            count += 1
    p_perm = (count + 1) / (B + 1)
    return {"obs_rho": float(obs), "abs_obs": float(abs_obs),
            "p_perm": float(p_perm), "null_mean": float(boot.mean()),
            "null_std": float(boot.std()),
"null_q025": float(np.quantile(boot, 0.025)),
            "null_q500": float(np.quantile(boot, 0.5)),
            "null_q975": float(np.quantile(boot, 0.975)),
            "n": int(n), "B": int(B)}


def write_tsv(name: str, rows: list[dict], keys: list[str]) -> str:
    out = os.path.join(RES, name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[iter112] wrote {out}  ({len(rows)} rows)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iter112 severship-reward coupling test")
    parser.add_argument("--n_w", type=int, default=4)
    parser.add_argument("--B_perm", type=int, default=50000)
    parser.add_argument("--seed_base", type=int, default=20260703)
    args = parser.parse_args(argv)

    n_w = int(args.n_w)
    B_perm = int(args.B_perm)
    seed_base = int(args.seed_base)
    ALGOS = ("grpo", "dr_grpo")
    TASKS = ["arithmetic_easy", "gsm8k_cot"]

    perrun108 = load_iter108_perrun()
    step_runs = load_step_log_inverse(DRGR_VS_GRPO, "arithmetic_easy") \
                + load_step_log_inverse(DRGR_GSM8K, "gsm8k_cot")
    long_tab = paired_per_window(perrun108, step_runs, n_w=n_w)

    delta_tab, common = paired_delta_table(long_tab, TASKS, ALGOS, n_w=n_w)

    # ----------- Q1: per-task pooled Spearman ------------
    pooled_rows = []
    perm_rows = []
    for task in TASKS:
        pts = [(delta_tab[(task, w, s)]["delta_bwd"],
                delta_tab[(task, w, s)]["delta_R"])
               for w in range(n_w)
               for s in common[(task, "pair")]]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        # severship = -Δ_bwd; reward = Δ_R
        severship = -x
        rho, p_p = spearman_pvalue(severship, y)
        null = permutation_null_spearman(
            severship, y, B=B_perm, seed=seed_base + hash(task) % 100000)
        pooled_rows.append({"task": task,
                            "spearman_rho_sever_vs_reward": rho,
                            "spearman_p_param": p_p,
                            "p_perm_two_sided": null["p_perm"],
                            "null_mean": null["null_mean"],
                            "null_std": null["null_std"],
                            "null_q025": null["null_q025"],
                            "null_q500": null["null_q500"],
                            "null_q975": null["null_q975"],
                            "n_points": null["n"],
                            "B_perm": B_perm,
                            "x_mean_severship": float(severship.mean()),
                            "y_mean_dR": float(y.mean())})
        perm_rows.append({"task": task, **null,
                          "spearman_p_param": p_p,
                          "point_rho_rvsreward": rho})

    # ----------- Q2: cumulative-window regression ---------
    cum_rows = []
    # per-task: build per-(seed, w_end) tuples
    for task in TASKS:
        seeds = common[(task, "pair")]
        for k in range(1, n_w + 1):  # accumulate up to and including k-1
            xs, ys = [], []
            for s in seeds:
                db = [delta_tab[(task, w, s)]["delta_bwd"]
                      for w in range(k) if (task, w, s) in delta_tab]
                dr = [delta_tab[(task, w, s)]["delta_R"]
                      for w in range(k) if (task, w, s) in delta_tab]
                if len(db) < 3:
                    continue
                cum_sev = -float(np.sum(db)) / len(db)  # mean severship across k windows
                cum_dR = float(np.sum(dr))             # accumulated reward delta
                xs.append(cum_sev); ys.append(cum_dR)
            xs = np.array(xs); ys = np.array(ys)
            if len(xs) >= 3:
                rho, p_p = spearman_pvalue(xs, ys)
                cum_rows.append({"task": task, "k_end": k, "n_seeds": len(xs),
                                  "rho_cum_sever_vs_cum_R": float(rho),
                                  "spearman_p_param": float(p_p),
                                  "mean_sever": float(xs.mean()),
                                  "mean_cum_dR": float(ys.mean())})

    # ----------- Q3: per-window per-task summary ----------
    per_window_rows = []
    for task in TASKS:
        for w in range(n_w):
            pairs = [(delta_tab[(task, w, s)]["delta_bwd"],
                      delta_tab[(task, w, s)]["delta_R"])
                     for s in common[(task, "pair")]
                     if (task, w, s) in delta_tab]
            if len(pairs) < 3:
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho, p_p = spearman_pvalue(-x, y)
            per_window_rows.append({"task": task, "window": w,
                                    "n_seeds": len(pairs),
                                    "mean_sever": float((-x).mean()),
                                    "mean_dR": float(y.mean()),
                                    "rho_window": float(rho),
                                    "spearman_p_param": float(p_p)})

    # ----------- bootstrap CI on pooled Spearman ----------
    rng = np.random.default_rng(seed_base)
    ci_rows = []
    for task in TASKS:
        pts = [(delta_tab[(task, w, s)]["delta_bwd"],
                delta_tab[(task, w, s)]["delta_R"])
               for w in range(n_w) for s in common[(task, "pair")]]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        sev = -x
        n = len(sev)
        obs_rho, _ = spearman_pvalue(sev, y)
        idx = rng.integers(0, n, size=(B_perm, n))
        boot = np.empty(B_perm)
        for b in range(B_perm):
            r_b, _ = spearman_pvalue(sev[idx[b]], y[idx[b]])
            boot[b] = r_b
        ci_rows.append({"task": task, "obs_rho": float(obs_rho),
                         "ci_lo": float(np.quantile(boot, 0.025)),
                         "ci_hi": float(np.quantile(boot, 0.975)),
                         "n": int(n), "B_boot": B_perm})

    # ----------- write all TSVs ----------
    write_tsv("length_bias_iter112_per_window.tsv",
              per_window_rows,
              ["task", "window", "n_seeds", "mean_sever", "mean_dR",
               "rho_window", "spearman_p_param"])
    write_tsv("length_bias_iter112_pooled.tsv",
              pooled_rows,
              ["task", "spearman_rho_sever_vs_reward", "spearman_p_param",
               "p_perm_two_sided",
               "null_mean", "null_std", "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "x_mean_severship", "y_mean_dR"])
    write_tsv("length_bias_iter112_permutation_null.tsv",
              perm_rows,
              ["task", "obs_rho", "abs_obs", "p_perm",
               "null_mean", "null_std", "null_q025", "null_q500",
               "null_q975", "n", "B",
               "spearman_p_param", "point_rho_rvsreward"])
    write_tsv("length_bias_iter112_cumulative.tsv",
              cum_rows,
              ["task", "k_end", "n_seeds",
               "rho_cum_sever_vs_cum_R", "spearman_p_param",
               "mean_sever", "mean_cum_dR"])
    write_tsv("length_bias_iter112_rho_bootstrap.tsv",
              ci_rows,
              ["task", "obs_rho", "ci_lo", "ci_hi", "n", "B_boot"])

    # consensus correlation between tasks
    all_x = []; all_y = []
    for task in TASKS:
        for w in range(n_w):
            for s in common[(task, "pair")]:
                if (task, w, s) in delta_tab:
                    all_x.append(-delta_tab[(task, w, s)]["delta_bwd"])
                    all_y.append(delta_tab[(task, w, s)]["delta_R"])
    consensus_rho, consensus_p = spearman_pvalue(np.array(all_x),
                                                  np.array(all_y))

    meta = {"iter": 112, "pillar": "P4-LengthBias",
            "n_w": n_w, "B_perm": B_perm, "seed_base": seed_base,
            "tasks": TASKS, "algos": list(ALGOS),
            "consensus_spearman": float(consensus_rho),
            "consensus_p_param": float(consensus_p),
            "n_consensus_points": int(len(all_x))}
    out = os.path.join(RES, "length_bias_iter112_meta.json")
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter112] wrote {out}")

    # headlines
    print("\n[iter112 headline] Pooled Spearman(-Δbwd, ΔR) per task")
    for r in pooled_rows:
        print(f"  {r['task']:14s}: ρ={r['spearman_rho_sever_vs_reward']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']}")
    print(f"\n[iter112 headline] Consensus across both tasks (n={len(all_x)}): "
          f"ρ={consensus_rho:+.3f} p={consensus_p:.3f}")
    print("\n[iter112 headline] Per-window Spearman(-Δbwd, ΔR)")
    for r in per_window_rows:
        print(f"  {r['task']:14s} w={r['window']}: ρ={r['rho_window']:+.3f} "
              f"p={r['spearman_p_param']:.3f} n={r['n_seeds']}")
    print("\n[iter112 headline] Cumulative-window regression (rho, p)")
    for r in cum_rows:
        print(f"  {r['task']:14s} k={r['k_end']}: ρ={r['rho_cum_sever_vs_cum_R']:+.3f} "
              f"p={r['spearman_p_param']:.3f} n={r['n_seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

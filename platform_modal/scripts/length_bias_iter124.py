#!/usr/bin/env python3
"""Iter 124 -- Pillar 4 (Length Bias / Dr.GRPO):
Severship x BASELINE length-inflation velocity (dL/dt) and
the corresponding reward-inflation velocity (dR/dt).

Iter 108 established per-window CCF magnitudes (|CCF_bwd|, |CCF_fwd|).
Iter 116 found an INVERTED dose response between severship and the
BASELINE ZVF (Dr.GR severs most where ZVF is LOW: the signal-starved
frontier).  Iter 120 found severship tracks the BASELINE GR backward-
CCF magnitude (the L -> R coupling strength).

Iter 124 turns to the DYNAMIC dimension:

  H1 -- Severship ~ baseline length-inflation velocity (dL/dt)
        H1: Spearman(-delta_bwd, dL_GR) > 0
        where dL_GR = L_w - L_{w-1}, the BASELINE GR
        per-window length growth rate.
        Positive rho => Dr.GR severs most on the WINDOWS
        where the GR baseline is ACTIVELY GROWING longer
        (the active length-inflation frontier).

  H2 -- Severship ~ baseline reward-inflation velocity (dR/dt)
        H2: Spearman(-delta_bwd, dR_GR) > 0
        Companion of H1 on the reward axis.  Tests whether
        Dr.GR severs most where BOTH L and R are inflating,
        i.e. the active-learning regime, or whether severship
        is selectively anti-L (length-specific).

  H3 -- Conditional independence test
        H3: After partialing out |CCF_bwd^GR| (iter120's predictor),
        the residual severship should retain an INDEPENDENT
        positive correlation with dL_GR.  Confirms dL/dt is
        NOT just a proxy for the static CCF magnitude.

  H4 -- Permutation null
        H1 rho survives a 10^5-shuffle chance distribution on
        the (window, seed) coupling within task.

  H5 -- Cross-task envelope of mean dL/dt vs mean severship.

INPUTS :
  experiments/results/length_bias_iter108_perrun_progress.tsv
  experiments/results/drgrpo_vs_grpo.json
  experiments/results/drgrpo_gsm8k_cot_full.json
OUTPUTS: 5 TSVs + meta under experiments/results/length_bias_iter124_*
USAGE  : python3 platform_modal/scripts/length_bias_iter124.py [--n_w 4 --B_perm 50000]
"""
from __future__ import annotations

import argparse
import csv
import json
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


# ---------------------------------------------------------------------------
# Loaders (mostly identical to iter120)
# ---------------------------------------------------------------------------
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
            for k in ("phi_L", "phi_R", "bwd", "fwd",
                      "bwd_signed", "fwd_signed"):
                row[k] = float(row[k])
            out.append(row)
    return out


def load_step_log(path: str, task_label: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        sl = r.get("step_log") or []
        if len(sl) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in sl],
                     dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in sl],
                     dtype=np.float64)
        out.append({"task": task_label, "algo": r["algo"],
                    "seed": int(r["seed"]), "n": int(len(sl)),
                    "L": L, "R": R})
    return out


def window_mean(x: np.ndarray, n_w: int) -> np.ndarray:
    """Return window-mean of x over n_w equal-length windows."""
    n = len(x)
    edges = [int(np.floor(n * w / n_w)) for w in range(n_w + 1)]
    for w in range(1, n_w + 1):
        edges[w] = max(edges[w], edges[w - 1] + 4)
        edges[w] = min(edges[w], n)
    out = np.zeros(n_w, dtype=np.float64)
    for w in range(n_w):
        out[w] = float(x[edges[w]:edges[w + 1]].mean())
    return out


def velocity(w_windows: np.ndarray) -> np.ndarray:
    """First-difference length-n_w array.  v[0] := w_windows[0]
    (no prior), v[w] = w_windows[w] - w_windows[w-1] for w>=1."""
    v = np.zeros_like(w_windows)
    v[0] = w_windows[0]
    if len(w_windows) > 1:
        v[1:] = w_windows[1:] - w_windows[:-1]
    return v


# ---------------------------------------------------------------------------
# Build long table indexed by (task, algo, seed, window) with iter108
# per-run CCF quantities and window-mean L,R.  Then make paired delta.
# ---------------------------------------------------------------------------
def build_long(perrun108: list[dict], step_runs: list[dict],
               n_w: int) -> list[dict[str, Any]]:
    by_step: dict[tuple, dict[int, dict]] = {}
    for r in step_runs:
        by_step.setdefault((r["task"], r["algo"]), {})[r["seed"]] = r

    out: list[dict[str, Any]] = []
    for r108 in perrun108:
        task = r108["task"]
        algo = r108["algo"]
        seed = r108["seed"]
        w = r108["window"]
        run = by_step.get((task, algo), {}).get(seed)
        if run is None:
            continue
        L_w = window_mean(run["L"], n_w=n_w)
        R_w = window_mean(run["R"], n_w=n_w)
        out.append({"task": task, "algo": algo, "seed": seed,
                    "window": w,
                    "bwd": float(r108["bwd"]),
                    "fwd": float(r108["fwd"]),
                    "bwd_signed": float(r108["bwd_signed"]),
                    "phi_L": float(r108["phi_L"]),
                    "phi_R": float(r108["phi_R"]),
                    "L_w": float(L_w[w]),
                    "R_w": float(R_w[w]),
                    "n_in_window": int(r108["n_in_window"]),
                    "n_total": int(r108["n_total"])})
    return out


def pair_long(long_tab: list[dict], tasks: list[str],
              algos: tuple, n_w: int) -> dict:
    by_tas: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in long_tab:
        by_tas.setdefault((r["task"], r["algo"]), {}).setdefault(
            r["seed"], {})[r["window"]] = r

    common: dict[str, list[int]] = {}
    for task in tasks:
        s_g = set(by_tas.get((task, algos[0]), {}).keys())
        s_d = set(by_tas.get((task, algos[1]), {}).keys())
        common[task] = sorted(s_g & s_d)

    out: dict[tuple, dict[str, float]] = {}
    for task in tasks:
        for s in common[task]:
            for w in range(n_w):
                g = by_tas[(task, algos[0])][s].get(w)
                d = by_tas[(task, algos[1])][s].get(w)
                if g is None or d is None:
                    continue
                out[(task, w, s)] = {
                    "delta_bwd": float(d["bwd"] - g["bwd"]),
                    "delta_fwd": float(d["fwd"] - g["fwd"]),
                    "delta_bwd_signed": float(d["bwd_signed"]
                                              - g["bwd_signed"]),
                    "delta_phi_L": float(d["phi_L"] - g["phi_L"]),
                    "delta_phi_R": float(d["phi_R"] - g["phi_R"]),
                    "delta_L": float(d["L_w"] - g["L_w"]),
                    "delta_R": float(d["R_w"] - g["R_w"]),
                    "g_bwd": float(g["bwd"]),
                    "d_bwd": float(d["bwd"]),
                    "g_fwd": float(g["fwd"]),
                    "d_fwd": float(d["fwd"]),
                    "g_phi_L": float(g["phi_L"]),
                    "g_phi_R": float(g["phi_R"]),
                    "g_L": float(g["L_w"]),
                    "g_R": float(g["R_w"]),
                    "d_L": float(d["L_w"]),
                    "d_R": float(d["R_w"]),
                }
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def spearman(x, y) -> tuple[float, float]:
    sp = stats.spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue)


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Rank-based partial Spearman: rho(x,y | z)."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    X = np.column_stack([rx, rz])
    X1 = np.column_stack([np.ones_like(rx), rz])
    beta_x, *_ = np.linalg.lstsq(X1, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(X1, ry, rcond=None)
    res_x = rx - X1 @ beta_x
    res_y = ry - X1 @ beta_y
    return spearman(res_x, res_y)


def permutation_null(x, y, B: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs, _ = spearman(x, y)
    abs_obs = abs(obs)
    n = len(x)
    y_arr = np.array(y, dtype=np.float64)
    count = 0
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.permutation(n)
        r_b, _ = spearman(x, y_arr[idx])
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
    print(f"[iter124] wrote {out}({len(rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Iter124: severship x baseline length-inflation "
                    "and reward-inflation velocity (dL/dt, dR/dt).")
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
    step_runs = (load_step_log(DRGR_VS_GRPO, "arithmetic_easy")
                 + load_step_log(DRGR_GSM8K, "gsm8k_cot"))

    # per-window L,R (already in long_tab), but we ALSO need the full
    # per-window array per (task, algo, seed) for velocity computation.
    long_tab = build_long(perrun108, step_runs, n_w=n_w)
    delta_tab = pair_long(long_tab, TASKS, ALGOS, n_w=n_w)

    # Build per-seed window arrays to compute dL/dt and dR/dt.
    per_seed_windows: dict[tuple, dict[str, np.ndarray]] = {}
    by_step: dict[tuple, dict[int, dict]] = {}
    for r in step_runs:
        by_step.setdefault((r["task"], r["algo"]), {})[r["seed"]] = r
    for task in TASKS:
        for algo in ALGOS:
            for seed, run in by_step.get((task, algo), {}).items():
                Lw = window_mean(run["L"], n_w=n_w)
                Rw = window_mean(run["R"], n_w=n_w)
                dL = velocity(Lw)
                dR = velocity(Rw)
                per_seed_windows[(task, algo, seed)] = {
                    "L": Lw, "R": Rw, "dL": dL, "dR": dR
                }

    common_seeds = {task: [] for task in TASKS}
    for (task, w, s) in delta_tab.keys():
        if w == 0 and s not in common_seeds[task]:
            common_seeds[task].append(s)
    for task in TASKS:
        common_seeds[task] = sorted(common_seeds[task])

    # Per-window rows for diagnostic visualization and per-window stats
    per_window_rows = []
    for task in TASKS:
        for w in range(n_w):
            sev_list, dL_list, dR_list, bwd_list = [], [], [], []
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                sev_list.append(-row["delta_bwd"])
                dL_list.append(g["dL"][w])
                dR_list.append(g["dR"][w])
                bwd_list.append(row["g_bwd"])
            if len(sev_list) < 3:
                continue
            x = np.array(sev_list)
            y_dL = np.array(dL_list)
            y_dR = np.array(dR_list)
            y_bwd = np.array(bwd_list)
            rho_dL, p_dL = spearman(x, y_dL)
            rho_dR, p_dR = spearman(x, y_dR)
            rho_bwd, p_bwd = spearman(x, y_bwd)
            per_window_rows.append({
                "task": task, "window": w, "n_seeds": len(sev_list),
                "mean_sever": float(x.mean()),
                "mean_dL_GR": float(y_dL.mean()),
                "mean_dR_GR": float(y_dR.mean()),
                "mean_bwd_GR": float(y_bwd.mean()),
                "rho_sever_vs_dL_GR": float(rho_dL),
                "p_sever_vs_dL_GR": float(p_dL),
                "rho_sever_vs_dR_GR": float(rho_dR),
                "p_sever_vs_dR_GR": float(p_dR),
                "rho_sever_vs_bwd_GR": float(rho_bwd),
                "p_sever_vs_bwd_GR": float(p_bwd),
            })

    # ---------- H1: sever ~ dL_GR (length-inflation velocity) ----------
    pooled_h1 = []
    perm_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(g["dL"][w])
        x = np.array(xs); y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h1", task)) % 100000)
        pooled_h1.append({"task": task,
                          "spearman_sever_vs_dL_GR": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_dL_GR": float(y.mean())})
        perm_h1.append({"task": task, "hypothesis": "H1_sever_vs_dL_GR",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H2: sever ~ dR_GR (reward-inflation velocity) ----------
    pooled_h2 = []
    perm_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(g["dR"][w])
        x = np.array(xs); y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h2", task)) % 100000)
        pooled_h2.append({"task": task,
                          "spearman_sever_vs_dR_GR": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_dR_GR": float(y.mean())})
        perm_h2.append({"task": task, "hypothesis": "H2_sever_vs_dR_GR",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H3: partial Spearman controlling for |CCF_bwd^GR| ----------
    partial_rows = []
    for task in TASKS:
        xs, ys, zs = [], [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(g["dL"][w])
                zs.append(row["g_bwd"])
        if len(xs) < 4:
            continue
        x = np.array(xs); y = np.array(ys); z = np.array(zs)
        rho_raw, p_raw = spearman(x, y)
        rho_part, p_part = partial_spearman(x, y, z)
        partial_rows.append({"task": task, "n": len(xs),
                             "rho_raw_sever_vs_dL": float(rho_raw),
                             "p_raw": float(p_raw),
                             "rho_partial_sever_vs_dL_given_bwd": float(rho_part),
                             "p_partial": float(p_part),
                             "rho_bwd_vs_dL_ctrl": float(spearman(z, y)[0])})

    # ---------- H4 / H5: cross-task envelope ----------
    envelope_rows = []
    for task in TASKS:
        per_seed_sever = []
        per_seed_dL = []
        per_seed_dR = []
        for s in common_seeds[task]:
            g = per_seed_windows.get((task, "grpo", s))
            if g is None:
                continue
            sevs = [-delta_tab[(task, w, s)]["delta_bwd"]
                    for w in range(n_w)
                    if (task, w, s) in delta_tab]
            per_seed_sever.append(float(np.mean(sevs)))
            per_seed_dL.append(float(np.mean(g["dL"])))
            per_seed_dR.append(float(np.mean(g["dR"])))
        envelope_rows.append({"task": task,
                              "n_seeds": len(per_seed_sever),
                              "mean_sever": float(np.mean(per_seed_sever)),
                              "std_sever": float(np.std(per_seed_sever)),
                              "mean_dL_GR": float(np.mean(per_seed_dL)),
                              "std_dL_GR": float(np.std(per_seed_dL)),
                              "mean_dR_GR": float(np.mean(per_seed_dR)),
                              "std_dR_GR": float(np.std(per_seed_dR))})
    se = np.array([r["mean_sever"] for r in envelope_rows])
    dle = np.array([r["mean_dL_GR"] for r in envelope_rows])
    dre = np.array([r["mean_dR_GR"] for r in envelope_rows])
    env_sev_dL_rho, env_sev_dL_p = float("nan"), float("nan")
    env_sev_dR_rho, env_sev_dR_p = float("nan"), float("nan")
    if len(se) >= 3:
        env_sev_dL_rho, env_sev_dL_p = spearman(se, dle)
        env_sev_dR_rho, env_sev_dR_p = spearman(se, dre)

    # ---------- consensus / bootstrap ----------
    all_sev_h1, all_dL = [], []
    all_sev_h2, all_dR = [], []
    for task in TASKS:
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                all_sev_h1.append(-row["delta_bwd"])
                all_dL.append(g["dL"][w])
                all_sev_h2.append(-row["delta_bwd"])
                all_dR.append(g["dR"][w])
    cons_h1_rho, cons_h1_p = spearman(np.array(all_sev_h1),
                                      np.array(all_dL))
    cons_h2_rho, cons_h2_p = spearman(np.array(all_sev_h2),
                                      np.array(all_dR))

    # bootstrap CIs
    rng = np.random.default_rng(seed_base + 8081)
    ci_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(g["dL"][w])
        x = np.array(xs); y = np.array(ys)
        obs, _ = spearman(x, y)
        n = len(x)
        idx = rng.integers(0, n, size=(B_perm, n))
        boot = np.empty(B_perm)
        for b in range(B_perm):
            r_b, _ = spearman(x[idx[b]], y[idx[b]])
            boot[b] = r_b
        ci_h1.append({"task": task, "obs_rho": float(obs),
                      "ci_lo": float(np.quantile(boot, 0.025)),
                      "ci_hi": float(np.quantile(boot, 0.975)),
                      "n": int(n), "B_boot": B_perm,
                      "hypothesis": "H1_sever_vs_dL_GR"})

    rng2 = np.random.default_rng(seed_base + 11743)
    ci_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                g = per_seed_windows.get((task, "grpo", s))
                if g is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(g["dR"][w])
        x = np.array(xs); y = np.array(ys)
        obs, _ = spearman(x, y)
        n = len(x)
        idx = rng2.integers(0, n, size=(B_perm, n))
        boot = np.empty(B_perm)
        for b in range(B_perm):
            r_b, _ = spearman(x[idx[b]], y[idx[b]])
            boot[b] = r_b
        ci_h2.append({"task": task, "obs_rho": float(obs),
                      "ci_lo": float(np.quantile(boot, 0.025)),
                      "ci_hi": float(np.quantile(boot, 0.975)),
                      "n": int(n), "B_boot": B_perm,
                      "hypothesis": "H2_sever_vs_dR_GR"})

    # ---------- write outputs ----------
    write_tsv("length_bias_iter124_pooled_h1_sever_vs_dL_GR.tsv",
              pooled_h1,
              ["task", "spearman_sever_vs_dL_GR", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_dL_GR"])
    write_tsv("length_bias_iter124_pooled_h2_sever_vs_dR_GR.tsv",
              pooled_h2,
              ["task", "spearman_sever_vs_dR_GR", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_dR_GR"])
    write_tsv("length_bias_iter124_permutation_null.tsv",
              perm_h1 + perm_h2,
              ["task", "hypothesis", "obs_rho", "abs_obs", "p_perm",
               "null_mean", "null_std", "null_q025", "null_q500",
               "null_q975", "n", "B",
               "spearman_p_param", "point_rho"])
    write_tsv("length_bias_iter124_envelope.tsv", envelope_rows,
              ["task", "n_seeds", "mean_sever", "std_sever",
               "mean_dL_GR", "std_dL_GR",
               "mean_dR_GR", "std_dR_GR"])
    write_tsv("length_bias_iter124_partial.tsv", partial_rows,
              ["task", "n",
               "rho_raw_sever_vs_dL", "p_raw",
               "rho_partial_sever_vs_dL_given_bwd", "p_partial",
               "rho_bwd_vs_dL_ctrl"])
    write_tsv("length_bias_iter124_per_window.tsv", per_window_rows,
              ["task", "window", "n_seeds",
               "mean_sever", "mean_dL_GR", "mean_dR_GR", "mean_bwd_GR",
               "rho_sever_vs_dL_GR", "p_sever_vs_dL_GR",
               "rho_sever_vs_dR_GR", "p_sever_vs_dR_GR",
               "rho_sever_vs_bwd_GR", "p_sever_vs_bwd_GR"])
    write_tsv("length_bias_iter124_rho_bootstrap.tsv", ci_h1 + ci_h2,
              ["task", "hypothesis", "obs_rho", "ci_lo", "ci_hi",
               "n", "B_boot"])

    meta = {"iter": 124, "pillar": "P4-LengthBias",
            "n_w": n_w, "B_perm": B_perm, "seed_base": seed_base,
            "tasks": TASKS, "algos": list(ALGOS),
            "consensus_H1_sever_vs_dL_GR": float(cons_h1_rho),
            "consensus_H1_p_param": float(cons_h1_p),
            "consensus_H2_sever_vs_dR_GR": float(cons_h2_rho),
            "consensus_H2_p_param": float(cons_h2_p),
            "n_consensus_points": int(len(all_sev_h1)),
            "envelope_sever_vs_dL_GR": float(env_sev_dL_rho),
            "envelope_sever_vs_dL_GR_p": float(env_sev_dL_p),
            "envelope_sever_vs_dR_GR": float(env_sev_dR_rho),
            "envelope_sever_vs_dR_GR_p": float(env_sev_dR_p),
            "n_tasks_envelope": int(len(envelope_rows))}
    out = os.path.join(RES, "length_bias_iter124_meta.json")
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter124] wrote {out}")

    # ---------- headlines ----------
    print("\n[iter124 headline] H1: Spearman(severship, GR baseline "
          "dL/dt) per task")
    for r in pooled_h1:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_dL_GR']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_dL_GR={r['mean_dL_GR']:+.3f}")
    print(f"\n[iter124 headline] H1 consensus (n={len(all_sev_h1)}): "
          f"rho={cons_h1_rho:+.3f} p={cons_h1_p:.3f}")
    print("\n[iter124 headline] H2: Spearman(severship, GR baseline "
          "dR/dt) per task")
    for r in pooled_h2:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_dR_GR']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_dR_GR={r['mean_dR_GR']:+.3f}")
    print(f"\n[iter124 headline] H2 consensus (n={len(all_sev_h2)}): "
          f"rho={cons_h2_rho:+.3f} p={cons_h2_p:.3f}")
    print("\n[iter124 headline] H3: partial Spearman controlling "
          "for |CCF_bwd^GR|")
    for r in partial_rows:
        print(f"  {r['task']:14s}: rho_raw={r['rho_raw_sever_vs_dL']:+.3f} "
              f"rho_partial={r['rho_partial_sever_vs_dL_given_bwd']:+.3f} "
              f"p_partial={r['p_partial']:.3f} n={r['n']}")
    print("\n[iter124 headline] H5: cross-task envelope")
    for r in envelope_rows:
        print(f"  {r['task']:14s}: mean_sever={r['mean_sever']:+.3f} "
              f"mean_dL_GR={r['mean_dL_GR']:+.3f} "
              f"mean_dR_GR={r['mean_dR_GR']:+.3f}")
    print(f"\n[iter124 headline] envelope rho(sever, dL_GR) "
          f"= {env_sev_dL_rho:+.3f} (p={env_sev_dL_p:.3f}) "
          f"rho(sever, dR_GR) = {env_sev_dR_rho:+.3f} "
          f"(p={env_sev_dR_p:.3f}) n_tasks={len(envelope_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
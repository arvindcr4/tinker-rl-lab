#!/usr/bin/env python3
"""Iter 120 -- Pillar 4 (Length Bias / Dr.GRPO):
Severship x baseline backward-CCF -- the "de-herding efficacy" frontier.

Extends iter108 (per-window AR(1)+CCF magnitude |CCF_bwd|) and
iter116 (sev x baseline ZVF, where iter116 found an INVERTED dose
response: Dr.GR severs most on the signal-STARVED task) by
testing the COMPLIMENTARY dimension: does severship intensity
track the BASELINE GR backward-channel CCF, i.e. the de-herding
LOAD the GRPO policy was carrying?

Sharp, falsifiable hypotheses:

  H1 (de-herding efficacy).  Per (task, window, seed):
      Spearman(|CCF_bwd^GR|, -delta_bwd) > 0.
      Meaning: Dr.GR severs most where the GR baseline
      backward-length-reward coupling was heaviest.

  H2 (per-task dose-response, H1 reframed).
      Spearman(|CCF_bwd^GR| - |CCF_bwd^Dr|, -delta_bwd)
      is positive -- Dr.GR reduces CCF by more where it
      was higher to begin with, AND severs the most
      on the same windows.

  H3 (cross-task envelope).  Per-task mean |CCF_bwd^GR|
      vs per-task mean -delta_bwd form a monotone envelope.

  H4 (permutation null).  H1 rho survives a 10^5-shuffle
      chance distribution on the (algo, seed) coupling
      within (task, window).

INPUTS :
  experiments/results/length_bias_iter108_perrun_progress.tsv
  experiments/results/drgrpo_vs_grpo.json
  experiments/results/drgrpo_gsm8k_cot_full.json
OUTPUTS: 5 TSVs + meta under experiments/results/length_bias_iter120_*
USAGE  : python3 scripts/length_bias_iter120.py [--n_w 4 --B_perm 50000]
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
# Loaders
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
                }
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def spearman(x, y) -> tuple[float, float]:
    sp = stats.spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue)


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
    print(f"[iter120] wrote {out}  ({len(rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Iter120: severship x baseline backward-CCF, "
                    "the de-herding efficacy frontier.")
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
    long_tab = build_long(perrun108, step_runs, n_w=n_w)
    delta_tab = pair_long(long_tab, TASKS, ALGOS, n_w=n_w)

    common_seeds = {task: [] for task in TASKS}
    for (task, w, s) in delta_tab.keys():
        if w == 0 and s not in common_seeds[task]:
            common_seeds[task].append(s)
    for task in TASKS:
        common_seeds[task] = sorted(common_seeds[task])

    # ---------- H1: sever ~ |CCF_bwd^GR| ----------
    pooled_h1 = []
    perm_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])   # sever
                ys.append(row["g_bwd"])        # |CCF_bwd_GR|
        x = np.array(xs); y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h1", task)) % 100000)
        pooled_h1.append({"task": task,
                          "spearman_sever_vs_bwd_GR": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_bwd_GR": float(y.mean())})
        perm_h1.append({"task": task, "hypothesis": "H1_sever_vs_bwd_GR",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H2: sever ~ |CCF_fwd^GR| (forward-CCF channel,
    #              INDEPENDENT of |CCF_bwd^GR|).  ----------
    # The forward channel R -> L is a different coupling path
    # (past reward predicting present length) and tests whether
    # Dr.GR's severship also tracks it, or only the backward path.
    pooled_h2 = []
    perm_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(row["g_fwd"])   # |CCF_fwd^GR|
        x = np.array(xs); y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h2", task)) % 100000)
        pooled_h2.append({"task": task,
                          "spearman_sever_vs_fwd_GR": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_fwd_GR": float(y.mean())})
        perm_h2.append({"task": task,
                        "hypothesis": "H2_sever_vs_fwd_GR",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H3: cross-task envelope ----------
    envelope_rows = []
    for task in TASKS:
        per_seed_sever = []
        per_seed_bwd_GR = []
        per_seed_fwd_GR = []
        for s in common_seeds[task]:
            sevs = [-delta_tab[(task, w, s)]["delta_bwd"]
                    for w in range(n_w)
                    if (task, w, s) in delta_tab]
            bzs = [delta_tab[(task, w, s)]["g_bwd"]
                   for w in range(n_w)
                   if (task, w, s) in delta_tab]
            fzs = [delta_tab[(task, w, s)]["g_fwd"]
                   for w in range(n_w)
                   if (task, w, s) in delta_tab]
            per_seed_sever.append(float(np.mean(sevs)))
            per_seed_bwd_GR.append(float(np.mean(bzs)))
            per_seed_fwd_GR.append(float(np.mean(fzs)))
        envelope_rows.append({"task": task,
                              "n_seeds": len(per_seed_sever),
                              "mean_sever": float(np.mean(per_seed_sever)),
                              "std_sever": float(np.std(per_seed_sever)),
                              "mean_bwd_GR":
                                  float(np.mean(per_seed_bwd_GR)),
                              "std_bwd_GR":
                                  float(np.std(per_seed_bwd_GR)),
                              "mean_fwd_GR":
                                  float(np.mean(per_seed_fwd_GR)),
                              "std_fwd_GR":
                                  float(np.std(per_seed_fwd_GR))})
    se = np.array([r["mean_sever"] for r in envelope_rows])
    bz = np.array([r["mean_bwd_GR"] for r in envelope_rows])
    fz = np.array([r["mean_fwd_GR"] for r in envelope_rows])
    env_sev_bz_rho, env_sev_bz_p = float("nan"), float("nan")
    env_sev_fz_rho, env_sev_fz_p = float("nan"), float("nan")
    if len(se) >= 3:
        env_sev_bz_rho, env_sev_bz_p = spearman(se, bz)
        env_sev_fz_rho, env_sev_fz_p = spearman(se, fz)

    # ---------- H4: per-window detail ----------
    per_window_rows = []
    for task in TASKS:
        for w in range(n_w):
            xs, ys_bwd, ys_fwd = [], [], []
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys_bwd.append(row["g_bwd"])
                ys_fwd.append(row["g_fwd"])
            if len(xs) < 3:
                continue
            x = np.array(xs)
            y_bwd = np.array(ys_bwd)
            y_fwd = np.array(ys_fwd)
            rho_bwd, p_bwd = spearman(x, y_bwd)
            rho_fwd, p_fwd = spearman(x, y_fwd)
            per_window_rows.append({"task": task, "window": w,
                                    "n_seeds": len(xs),
                                    "mean_sever": float(x.mean()),
                                    "mean_bwd_GR": float(y_bwd.mean()),
                                    "mean_fwd_GR": float(y_fwd.mean()),
                                    "rho_sever_vs_bwd_GR": float(rho_bwd),
                                    "p_sever_vs_bwd_GR": float(p_bwd),
                                    "rho_sever_vs_fwd_GR":
                                        float(rho_fwd),
                                    "p_sever_vs_fwd_GR": float(p_fwd)})

    # ---------- consensus / bootstrap ----------
    all_sev_h1, all_bwd = [], []
    all_sev_h2, all_fwd = [], []
    for task in TASKS:
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                all_sev_h1.append(-row["delta_bwd"])
                all_bwd.append(row["g_bwd"])
                all_sev_h2.append(-row["delta_bwd"])
                all_fwd.append(row["g_fwd"])
    cons_h1_rho, cons_h1_p = spearman(np.array(all_sev_h1),
                                      np.array(all_bwd))
    cons_h2_rho, cons_h2_p = spearman(np.array(all_sev_h2),
                                      np.array(all_fwd))

    # ---------- bootstrap CIs ----------
    rng = np.random.default_rng(seed_base + 7919)
    ci_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(row["g_bwd"])
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
                      "hypothesis": "H1_sever_vs_bwd_GR"})

    rng2 = np.random.default_rng(seed_base + 11587)
    ci_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common_seeds[task]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(row["g_fwd"])
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
                      "hypothesis": "H2_sever_vs_fwd_GR"})

    # ---------- write outputs ----------
    write_tsv("length_bias_iter120_pooled_h1_sever_vs_bwd_GR.tsv",
              pooled_h1,
              ["task", "spearman_sever_vs_bwd_GR", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_bwd_GR"])
    write_tsv("length_bias_iter120_pooled_h2_sever_vs_fwd_GR.tsv",
              pooled_h2,
              ["task", "spearman_sever_vs_fwd_GR", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_fwd_GR"])
    write_tsv("length_bias_iter120_permutation_null.tsv",
              perm_h1 + perm_h2,
              ["task", "hypothesis", "obs_rho", "abs_obs", "p_perm",
               "null_mean", "null_std", "null_q025", "null_q500",
               "null_q975", "n", "B",
               "spearman_p_param", "point_rho"])
    write_tsv("length_bias_iter120_envelope.tsv", envelope_rows,
              ["task", "n_seeds", "mean_sever", "std_sever",
               "mean_bwd_GR", "std_bwd_GR",
               "mean_fwd_GR", "std_fwd_GR"])
    write_tsv("length_bias_iter120_per_window.tsv", per_window_rows,
              ["task", "window", "n_seeds",
               "mean_sever", "mean_bwd_GR", "mean_fwd_GR",
               "rho_sever_vs_bwd_GR", "p_sever_vs_bwd_GR",
               "rho_sever_vs_fwd_GR", "p_sever_vs_fwd_GR"])
    write_tsv("length_bias_iter120_rho_bootstrap.tsv", ci_h1 + ci_h2,
              ["task", "hypothesis", "obs_rho", "ci_lo", "ci_hi",
               "n", "B_boot"])

    meta = {"iter": 120, "pillar": "P4-LengthBias",
            "n_w": n_w, "B_perm": B_perm, "seed_base": seed_base,
            "tasks": TASKS, "algos": list(ALGOS),
            "consensus_H1_sever_vs_bwd_GR": float(cons_h1_rho),
            "consensus_H1_p_param": float(cons_h1_p),
            "consensus_H2_sever_vs_fwd_GR": float(cons_h2_rho),
            "consensus_H2_p_param": float(cons_h2_p),
            "n_consensus_points": int(len(all_sev_h1)),
            "envelope_sever_vs_bwd_GR": float(env_sev_bz_rho),
            "envelope_sever_vs_bwd_GR_p": float(env_sev_bz_p),
            "envelope_sever_vs_fwd_GR": float(env_sev_fz_rho),
            "envelope_sever_vs_fwd_GR_p": float(env_sev_fz_p),
            "n_tasks_envelope": int(len(envelope_rows))}
    out = os.path.join(RES, "length_bias_iter120_meta.json")
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter120] wrote {out}")

    # ---------- headlines ----------
    print("\n[iter120 headline] H1: Spearman(severship, GR baseline "
          "|CCF_bwd|) per task")
    for r in pooled_h1:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_bwd_GR']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_bwd_GR={r['mean_bwd_GR']:.3f}")
    print(f"\n[iter120 headline] H1 consensus (n={len(all_sev_h1)}): "
          f"rho={cons_h1_rho:+.3f} p={cons_h1_p:.3f}")
    print("\n[iter120 headline] H2: Spearman(severship, "
          "GR baseline |CCF_fwd|) per task")
    for r in pooled_h2:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_fwd_GR']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_fwd_GR={r['mean_fwd_GR']:.3f}")
    print(f"\n[iter120 headline] H2 consensus (n={len(all_sev_h2)}): "
          f"rho={cons_h2_rho:+.3f} p={cons_h2_p:.3f}")
    print("\n[iter120 headline] H3: cross-task envelope")
    for r in envelope_rows:
        print(f"  {r['task']:14s}: mean_sever={r['mean_sever']:+.3f} "
              f"mean_bwd_GR={r['mean_bwd_GR']:.3f} "
              f"mean_fwd_GR={r['mean_fwd_GR']:.3f}")
    print(f"\n[iter120 headline] envelope rho(sever, bwd_GR) "
          f"= {env_sev_bz_rho:+.3f} (p={env_sev_bz_p:.3f}) "
          f"rho(sever, fwd_GR) = {env_sev_fz_rho:+.3f} "
          f"(p={env_sev_fz_p:.3f}) n_tasks={len(envelope_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

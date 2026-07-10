#!/usr/bin/env python3
"""Iter 116 -- Pillar 4 (Length Bias / Dr.GRPO): Severship x ZVF cross-pillar.

Closes the structural gap between iter108 (per-window Dr.GR - GR
|CCF(bwd)| severship ICCA), iter112 (sever-reward Spearman is
weakly negative, rho = -0.21 across n = 32), and the Pillar 2 ZVF
dose-response literature (iter110, iter114 anti-herding delta_d).

The motivating question is: **at what baseline ZVF level does the
Dr.GRPO severship mechanism actually fire, and does severship produce
a measurable ZVF-trajectory delta?**

Hypotheses tested (each sharp & falsifiable):

  H1 (sever-ZVF dose-response).  Per (task, window, seed), the
      Dr.GR - GR severship intensity -delta_bwd is positively
      correlated with the *baseline* (GRPO) window-mean ZVF.
      Mechanistic reading: severship is most active where there is
      the most spurious signal to sever.

  H2 (sever -> delta_ZVF).  Per (task, window, seed), the severship
      intensity correlates with the *paired* Dr.GR - GR delta ZVF
      (i.e. did severship actually lower ZVF relative to GRPO in
      the same window?).  This is the operational "does the lever
      work" test, complementing iter112's "does the lever improve
      reward" test.

  H3 (cross-task dose envelope).  Per-task mean severship vs
      per-task mean (GR) ZVF:  does the cross-task pattern form a
      monotone dose envelope (more baseline ZVF -> more severship)?
      This is the cross-pillar (P4 x P2) summary.

  H4 (permutation null).  Permute (algo, seed) coupling within
      (task, window) for H2 to test whether the observed rho
      survives a 10^5-shuffle chance distribution.

INPUTS :
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json
  platform_hybrid/experiments/results/length_bias_iter108_perrun_progress.tsv
OUTPUTS: 6 TSVs + meta under platform_hybrid/experiments/results/length_bias_iter116_*
USAGE  : python3 platform_modal/scripts/length_bias_iter116.py [--n_w 4 --B_perm 50000]
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
def load_step_log(path: str, task_label: str) -> list[dict[str, Any]]:
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        sl = r.get("step_log") or []
        if len(sl) < 5:
            continue
        Z = np.array([float(s.get("zvf", float("nan")))
                      for s in sl], dtype=np.float64)
        L = np.array([float(s["mean_comp_len"]) for s in sl], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in sl], dtype=np.float64)
        out.append({"task": task_label, "algo": r["algo"],
                    "seed": int(r["seed"]), "n": int(len(sl)),
                    "Z": Z, "L": L, "R": R})
    return out


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
            for k in ("phi_L", "phi_R", "bwd", "fwd", "bwd_signed",
                      "fwd_signed"):
                row[k] = float(row[k])
            out.append(row)
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
# Build long table with sever intensity (per iter108 bwd), windowed
# GR ZVF, paired delta ZVF.
# ---------------------------------------------------------------------------
def build_long(step_runs: list[dict], perrun108: list[dict],
               n_w: int) -> list[dict[str, Any]]:
    bwd_idx: dict[tuple, dict[int, float]] = {}
    for r in perrun108:
        key = (r["task"], r["algo"], r["seed"])
        bwd_idx.setdefault(key, {})[r["window"]] = r["bwd"]

    by_tas: dict[tuple, dict[int, dict]] = {}
    for r in step_runs:
        by_tas.setdefault((r["task"], r["algo"]), {})[r["seed"]] = r

    out: list[dict[str, Any]] = []
    for (task, algo), seeds_map in by_tas.items():
        for seed, run in seeds_map.items():
            Z_w = window_mean(run["Z"], n_w=n_w)
            L_w = window_mean(run["L"], n_w=n_w)
            R_w = window_mean(run["R"], n_w=n_w)
            for w in range(n_w):
                bwd = bwd_idx.get((task, algo, seed), {}).get(w)
                if bwd is None:
                    continue
                out.append({"task": task, "algo": algo, "seed": seed,
                            "window": w,
                            "bwd": float(bwd),
                            "Z_w": float(Z_w[w]),
                            "L_w": float(L_w[w]),
                            "R_w": float(R_w[w]),
                            "n_total": int(run["n"])})
    return out


def pair_table(long_tab: list[dict], tasks: list[str],
               algos: tuple, n_w: int) -> tuple[dict, dict]:
    by_seed: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in long_tab:
        by_seed.setdefault((r["task"], r["algo"]), {}).setdefault(
            r["seed"], {})[r["window"]] = r
    common: dict[tuple, list[int]] = {}
    for task in tasks:
        s_g = set(by_seed.get((task, algos[0]), {}).keys())
        s_d = set(by_seed.get((task, algos[1]), {}).keys())
        common[(task, "pair")] = sorted(s_g & s_d)
    out: dict[tuple, dict[str, float]] = {}
    for task in tasks:
        for s in common[(task, "pair")]:
            for w in range(n_w):
                g = by_seed[(task, algos[0])][s].get(w)
                d = by_seed[(task, algos[1])][s].get(w)
                if g is None or d is None:
                    continue
                out[(task, w, s)] = {
                    "delta_bwd": float(d["bwd"] - g["bwd"]),
                    "delta_Z": float(d["Z_w"] - g["Z_w"]),
                    "delta_L": float(d["L_w"] - g["L_w"]),
                    "delta_R": float(d["R_w"] - g["R_w"]),
                    "g_bwd": float(g["bwd"]),
                    "d_bwd": float(d["bwd"]),
                    "g_Z": float(g["Z_w"]),
                    "d_Z": float(d["Z_w"]),
                }
    return out, common


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
    print(f"[iter116] wrote {out}  ({len(rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Iter116 severship x ZVF cross-pillar unification")
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
    long_tab = build_long(step_runs, perrun108, n_w=n_w)
    delta_tab, common = pair_table(long_tab, TASKS, ALGOS, n_w=n_w)

    # ---------- H1: sever ~ baseline ZVF (per task pooled) ----------
    pooled_h1 = []
    perm_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])   # severship intensity
                ys.append(row["g_Z"])          # baseline GR ZVF
        x = np.array(xs)
        y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h1", task)) % 100000)
        pooled_h1.append({"task": task,
                          "spearman_sever_vs_baseline_ZVF": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_baseline_ZVF": float(y.mean())})
        perm_h1.append({"task": task, "hypothesis": "H1_sever_vs_baseline_ZVF",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H2: sever ~ paired delta_ZVF (per task pooled) ----------
    pooled_h2 = []
    perm_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(-row["delta_Z"])   # ZVF *reduction* by Dr.GR
        x = np.array(xs)
        y = np.array(ys)
        rho, p_p = spearman(x, y)
        null = permutation_null(x, y, B=B_perm,
                                seed=seed_base + hash(("h2", task)) % 100000)
        pooled_h2.append({"task": task,
                          "spearman_sever_vs_neg_delta_ZVF": rho,
                          "spearman_p_param": p_p,
                          "p_perm_two_sided": null["p_perm"],
                          "null_mean": null["null_mean"],
                          "null_std": null["null_std"],
                          "null_q025": null["null_q025"],
                          "null_q500": null["null_q500"],
                          "null_q975": null["null_q975"],
                          "n_points": null["n"], "B_perm": B_perm,
                          "mean_sever": float(x.mean()),
                          "mean_neg_delta_ZVF": float(y.mean())})
        perm_h2.append({"task": task, "hypothesis": "H2_sever_vs_neg_delta_ZVF",
                        **null, "spearman_p_param": p_p,
                        "point_rho": rho})

    # ---------- H3: cross-task dose envelope ----------
    envelope_rows = []
    for task in TASKS:
        sever_seeds = []
        baseline_Z_seeds = []
        delta_Z_seeds = []
        for s in common[(task, "pair")]:
            # mean over windows for this seed
            sevs = [-delta_tab[(task, w, s)]["delta_bwd"] for w in range(n_w)
                    if (task, w, s) in delta_tab]
            bzs = [delta_tab[(task, w, s)]["g_Z"] for w in range(n_w)
                   if (task, w, s) in delta_tab]
            dzs = [-delta_tab[(task, w, s)]["delta_Z"] for w in range(n_w)
                   if (task, w, s) in delta_tab]
            sever_seeds.append(float(np.mean(sevs)))
            baseline_Z_seeds.append(float(np.mean(bzs)))
            delta_Z_seeds.append(float(np.mean(dzs)))
        envelope_rows.append({"task": task,
                              "n_seeds": len(sever_seeds),
                              "mean_sever": float(np.mean(sever_seeds)),
                              "std_sever": float(np.std(sever_seeds)),
                              "mean_baseline_ZVF":
                                  float(np.mean(baseline_Z_seeds)),
                              "std_baseline_ZVF":
                                  float(np.std(baseline_Z_seeds)),
                              "mean_neg_delta_ZVF":
                                  float(np.mean(delta_Z_seeds)),
                              "std_neg_delta_ZVF":
                                  float(np.std(delta_Z_seeds))})
    # cross-task envelope slope
    se = np.array([r["mean_sever"] for r in envelope_rows])
    bz = np.array([r["mean_baseline_ZVF"] for r in envelope_rows])
    nz = np.array([r["mean_neg_delta_ZVF"] for r in envelope_rows])
    env_sev_bz_rho = float("nan")
    env_sev_bz_p = float("nan")
    env_sev_nz_rho = float("nan")
    env_sev_nz_p = float("nan")
    if len(se) >= 3:
        env_sev_bz_rho, env_sev_bz_p = spearman(se, bz)
        env_sev_nz_rho, env_sev_nz_p = spearman(se, nz)

    # ---------- H4: per-window detail ----------
    per_window_rows = []
    for task in TASKS:
        for w in range(n_w):
            xs, ys_sev_bz, ys_sev_dz = [], [], []
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys_sev_bz.append(row["g_Z"])
                ys_sev_dz.append(-row["delta_Z"])
            if len(xs) < 3:
                continue
            x = np.array(xs)
            y_bz = np.array(ys_sev_bz)
            y_dz = np.array(ys_sev_dz)
            rho_bz, p_bz = spearman(x, y_bz)
            rho_dz, p_dz = spearman(x, y_dz)
            per_window_rows.append({"task": task, "window": w,
                                    "n_seeds": len(xs),
                                    "mean_sever": float(x.mean()),
                                    "mean_baseline_ZVF": float(y_bz.mean()),
                                    "mean_neg_delta_ZVF": float(y_dz.mean()),
                                    "rho_sever_vs_baseline_ZVF": float(rho_bz),
                                    "p_sever_vs_baseline_ZVF": float(p_bz),
                                    "rho_sever_vs_neg_delta_ZVF":
                                        float(rho_dz),
                                    "p_sever_vs_neg_delta_ZVF": float(p_dz)})

    # ---------- consensus / bootstrap on H1+H2 ----------
    all_sev_h1, all_bz = [], []
    all_sev_h2, all_neg_dz = [], []
    for task in TASKS:
        for w in range(n_w):
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                all_sev_h1.append(-row["delta_bwd"])
                all_bz.append(row["g_Z"])
                all_sev_h2.append(-row["delta_bwd"])
                all_neg_dz.append(-row["delta_Z"])
    cons_h1_rho, cons_h1_p = spearman(np.array(all_sev_h1),
                                      np.array(all_bz))
    cons_h2_rho, cons_h2_p = spearman(np.array(all_sev_h2),
                                      np.array(all_neg_dz))

    rng = np.random.default_rng(seed_base + 7919)
    ci_h1 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(row["g_Z"])
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
                      "hypothesis": "H1_sever_vs_baseline_ZVF"})

    rng2 = np.random.default_rng(seed_base + 11587)
    ci_h2 = []
    for task in TASKS:
        xs, ys = [], []
        for w in range(n_w):
            for s in common[(task, "pair")]:
                row = delta_tab.get((task, w, s))
                if row is None:
                    continue
                xs.append(-row["delta_bwd"])
                ys.append(-row["delta_Z"])
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
                      "hypothesis": "H2_sever_vs_neg_delta_ZVF"})

    # ---------- write outputs ----------
    write_tsv("length_bias_iter116_pooled_h1_sever_vs_baseline_ZVF.tsv",
              pooled_h1,
              ["task", "spearman_sever_vs_baseline_ZVF", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_baseline_ZVF"])
    write_tsv("length_bias_iter116_pooled_h2_sever_vs_neg_delta_ZVF.tsv",
              pooled_h2,
              ["task", "spearman_sever_vs_neg_delta_ZVF", "spearman_p_param",
               "p_perm_two_sided", "null_mean", "null_std",
               "null_q025", "null_q500", "null_q975",
               "n_points", "B_perm", "mean_sever", "mean_neg_delta_ZVF"])
    write_tsv("length_bias_iter116_permutation_null.tsv",
              perm_h1 + perm_h2,
              ["task", "hypothesis", "obs_rho", "abs_obs", "p_perm",
               "null_mean", "null_std", "null_q025", "null_q500",
               "null_q975", "n", "B",
               "spearman_p_param", "point_rho"])
    write_tsv("length_bias_iter116_envelope.tsv", envelope_rows,
              ["task", "n_seeds", "mean_sever", "std_sever",
               "mean_baseline_ZVF", "std_baseline_ZVF",
               "mean_neg_delta_ZVF", "std_neg_delta_ZVF"])
    write_tsv("length_bias_iter116_per_window.tsv", per_window_rows,
              ["task", "window", "n_seeds",
               "mean_sever", "mean_baseline_ZVF", "mean_neg_delta_ZVF",
               "rho_sever_vs_baseline_ZVF", "p_sever_vs_baseline_ZVF",
               "rho_sever_vs_neg_delta_ZVF", "p_sever_vs_neg_delta_ZVF"])
    write_tsv("length_bias_iter116_rho_bootstrap.tsv", ci_h1 + ci_h2,
              ["task", "hypothesis", "obs_rho", "ci_lo", "ci_hi",
               "n", "B_boot"])

    meta = {"iter": 116, "pillar": "P4-LengthBias",
            "n_w": n_w, "B_perm": B_perm, "seed_base": seed_base,
            "tasks": TASKS, "algos": list(ALGOS),
            "consensus_H1_sever_vs_baseline_ZVF": float(cons_h1_rho),
            "consensus_H1_p_param": float(cons_h1_p),
            "consensus_H2_sever_vs_neg_delta_ZVF": float(cons_h2_rho),
            "consensus_H2_p_param": float(cons_h2_p),
            "n_consensus_points": int(len(all_sev_h1)),
            "envelope_sever_vs_baseline_ZVF": float(env_sev_bz_rho),
            "envelope_sever_vs_baseline_ZVF_p": float(env_sev_bz_p),
            "envelope_sever_vs_neg_delta_ZVF": float(env_sev_nz_rho),
            "envelope_sever_vs_neg_delta_ZVF_p": float(env_sev_nz_p),
            "n_tasks_envelope": int(len(envelope_rows))}
    out = os.path.join(RES, "length_bias_iter116_meta.json")
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter116] wrote {out}")

    # headlines
    print("\n[iter116 headline] H1: Spearman(severship, baseline GR ZVF) "
          "per task")
    for r in pooled_h1:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_baseline_ZVF']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_ZVF_GR={r['mean_baseline_ZVF']:.3f}")
    print(f"\n[iter116 headline] H1 consensus (n={len(all_sev_h1)}): "
          f"rho={cons_h1_rho:+.3f} p={cons_h1_p:.3f}")
    print("\n[iter116 headline] H2: Spearman(severship, -delta_ZVF) "
          "per task")
    for r in pooled_h2:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_neg_delta_ZVF']:+.3f} "
              f"p_param={r['spearman_p_param']:.3f} "
              f"p_perm={r['p_perm_two_sided']:.4f} "
              f"n={r['n_points']} mean_sever={r['mean_sever']:+.3f} "
              f"mean_neg_dZ={r['mean_neg_delta_ZVF']:+.3f}")
    print(f"\n[iter116 headline] H2 consensus (n={len(all_sev_h2)}): "
          f"rho={cons_h2_rho:+.3f} p={cons_h2_p:.3f}")
    print("\n[iter116 headline] H3: cross-task envelope")
    for r in envelope_rows:
        print(f"  {r['task']:14s}: mean_sever={r['mean_sever']:+.3f} "
              f"baseline_ZVF={r['mean_baseline_ZVF']:.3f} "
              f"neg_delta_ZVF={r['mean_neg_delta_ZVF']:+.3f}")
    print(f"\n[iter116 headline] envelope rho(sever, baseline_ZVF) "
          f"= {env_sev_bz_rho:+.3f} (p={env_sev_bz_p:.3f}) "
          f"rho(sever, neg_dZ) = {env_sev_nz_rho:+.3f} "
          f"(p={env_sev_nz_p:.3f}) n_tasks={len(envelope_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
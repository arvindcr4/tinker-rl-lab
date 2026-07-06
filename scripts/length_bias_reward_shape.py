#!/usr/bin/env python3
"""
Iter 16 — Pillar 4 Length-Bias elevation: REWARD-SHAPE DECOMPOSITION.

Frontier synthesis (Round 2 of FRONTIER_INSIGHTS.md): "ZVF is contrastive yield,
not difficulty." The natural completion is: the LENGTH-REWARD shape (i.e.
E[R | L]) is the same diagnostic applied to the response-length axis. This
script decomposes per-step trajectories into the conditional reward curve
E[R | L_bin] and asks whether Dr.GRPO flattens that curve vs GRPO.

Outputs:
  experiments/results/length_bias_reward_shape.tsv
    - per (algo, task, seed) binned E[R | L_bin]
    - per (algo, task) aggregate slope + Spearman rho between L_bin and E[R]
  figures/length_bias_reward_shape.pdf
    - 2x2 grid: scatter of (L,R) per step with per-bin mean overlay +
      per-task slope comparison bar
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
FIG = REPO / "figures"

DRGRPO_FILE = RESULTS / "drgrpo_vs_grpo.json"
GSM_FILE = RESULTS / "drgrpo_gsm8k_cot_full.json"
OUT_TSV = RESULTS / "length_bias_reward_shape.tsv"
OUT_FIG = FIG / "length_bias_reward_shape.pdf"
OUT_PNG = FIG / "length_bias_reward_shape.png"


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rho with two-sided p-value (permutation-free, normal approx)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = rx.mean(), ry.mean()
    sx = ((rx - mx) ** 2).sum()
    sy = ((ry - my) ** 2).sum()
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    rho = float(((rx - mx) * (ry - my)).sum() / math.sqrt(sx * sy))
    n = len(x)
    # two-sided p, normal approx
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        z = rho * math.sqrt((n - 2) / (1 - rho * rho + 1e-12))
        # erf-based normal CDF
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return rho, p


def theilsen(x: list[float], y: list[float]) -> float:
    """Theil-Sen slope estimator (robust to outliers, O(n^2) for small n)."""
    if len(x) < 2:
        return float("nan")
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else float("nan")


def load_runs(path: pathlib.Path, task_label: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    runs = []
    for r in d["runs"]:
        if "step_log" not in r:
            continue
        rows = []
        for s in r["step_log"]:
            if "mean_reward" in s and "mean_comp_len" in s:
                rows.append(
                    {
                        "step": s["step"],
                        "reward": float(s["mean_reward"]),
                        "length": float(s["mean_comp_len"]),
                        "zvf": float(s.get("zvf", float("nan"))),
                    }
                )
        if not rows:
            continue
        runs.append(
            {
                "algo": r["algo"],
                "seed": r["seed"],
                "task": task_label,
                "model": r.get("model", "?"),
                "n_steps": len(rows),
                "rows": rows,
            }
        )
    return runs


def per_run_bin(run: dict, n_bins: int = 4) -> list[dict]:
    """Bin a run's per-step trajectory into n_bins length-quantile bins,
    return per-bin (len_mid, mean_reward, count)."""
    rows = run["rows"]
    lengths = np.array([r["length"] for r in rows])
    rewards = np.array([r["reward"] for r in rows])
    if len(lengths) < n_bins * 2:
        n_bins = max(2, len(lengths) // 4)
    edges = np.quantile(lengths, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    bins = np.digitize(lengths, edges[1:-1], right=False)
    out = []
    for b in range(len(edges) - 1):
        m = bins == b
        if m.sum() == 0:
            continue
        out.append(
            {
                "bin": f"L{b + 1}",
                "algo": run["algo"],
                "seed": run["seed"],
                "task": run["task"],
                "model": run["model"],
                "len_min": float(lengths[m].min()),
                "len_max": float(lengths[m].max()),
                "len_mid": float(lengths[m].mean()),
                "mean_reward": float(rewards[m].mean()),
                "mean_zvf": float(np.mean([r["zvf"] for r in rows if m[rows.index(r)]])) if False else float(np.nan),
                "n_steps": int(m.sum()),
            }
        )
    return out


def aggregate_curve(rows: list[dict]) -> dict:
    """Aggregate across seeds within (algo, task): produce mean E[R | L_bin]."""
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[r["bin"]].append(r)
    out = []
    for bin_name in sorted(by_bin):
        L = by_bin[bin_name]
        mid = np.array([x["len_mid"] for x in L])
        R = np.array([x["mean_reward"] for x in L])
        out.append(
            {
                "bin": bin_name,
                "len_mid_mean": float(mid.mean()),
                "len_mid_se": float(mid.std(ddof=1) / math.sqrt(len(mid))) if len(mid) > 1 else 0.0,
                "mean_reward": float(R.mean()),
                "mean_reward_se": float(R.std(ddof=1) / math.sqrt(len(R))) if len(R) > 1 else 0.0,
                "n_seeds": len(L),
            }
        )
    return {"per_bin": out}


def task_curve_stats(curve: dict, n_seeds: int) -> dict:
    """Compute Spearman rho(L_mid, R) and Theil-Sen slope per aggregated curve."""
    bins = curve["per_bin"]
    if len(bins) < 3:
        return {"rho": float("nan"), "p": float("nan"), "slope": float("nan")}
    L = [b["len_mid_mean"] for b in bins]
    R = [b["mean_reward"] for b in bins]
    rho, p = spearman(L, R)
    slope = theilsen(L, R)
    return {"rho": rho, "p": p, "slope": slope, "n_bins": len(bins), "n_seeds": n_seeds}


def main():
    if not DRGRPO_FILE.exists() or not GSM_FILE.exists():
        print(f"Missing data: {DRGRPO_FILE.exists()=} {GSM_FILE.exists()=}")
        sys.exit(1)

    runs = load_runs(DRGRPO_FILE, task_label="arithmetic_easy_qwen2.5-0.5b")
    runs += load_runs(GSM_FILE, task_label="gsm8k_cot_hard_qwen2.5-1.5b")
    print(f"[iter16] loaded {len(runs)} runs")

    # ---- per-run bins ----
    all_bin_rows: list[dict] = []
    per_run_stats: list[dict] = []
    for run in runs:
        all_bin_rows.extend(per_run_bin(run, n_bins=4))
        L = [r["length"] for r in run["rows"]]
        R = [r["reward"] for r in run["rows"]]
        rho, p = spearman(L, R)
        slope = theilsen(L, R)
        per_run_stats.append(
            {
                "algo": run["algo"],
                "seed": run["seed"],
                "task": run["task"],
                "model": run["model"],
                "n_steps": run["n_steps"],
                "rho_len_rew": rho,
                "p_len_rew": p,
                "slope_len_rew": slope,
            }
        )

    # ---- aggregate per (algo, task) ----
    by_at: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_bin_rows:
        by_at[(r["algo"], r["task"])].append(r)
    curves = {}
    stats = {}
    for k, v in by_at.items():
        c = aggregate_curve(v)
        curves[k] = c
        stats[k] = task_curve_stats(c, n_seeds=len({x["seed"] for x in v}))

    # ---- write TSV ----
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "section",
                "task",
                "algo",
                "bin",
                "len_min",
                "len_max",
                "len_mid",
                "mean_reward",
                "n_steps",
                "seed",
                "model",
            ]
        )
        # per-run Spearman rho(L, R) -- the cleaner Dr.GRPO test
        for r in per_run_stats:
            w.writerow(
                [
                    "per_run_rho",
                    r["task"],
                    r["algo"],
                    "",
                    "",
                    "",
                    "",
                    f"{r['rho_len_rew']:.6f}" if not math.isnan(r["rho_len_rew"]) else "nan",
                    r["n_steps"],
                    r["seed"],
                    r["model"],
                ]
            )
        # paired bootstrap on per-run rho
        rng = np.random.default_rng(42)
        paired_rows = []
        for task in {r["task"] for r in per_run_stats}:
            g = [r for r in per_run_stats if r["algo"] == "grpo" and r["task"] == task]
            d = [r for r in per_run_stats if r["algo"] == "dr_grpo" and r["task"] == task]
            g_dict = {x["seed"]: x["rho_len_rew"] for x in g}
            d_dict = {x["seed"]: x["rho_len_rew"] for x in d}
            seeds = sorted(set(g_dict) & set(d_dict))
            if not seeds:
                continue
            deltas = np.array([d_dict[s] - g_dict[s] for s in seeds])
            observed = float(deltas.mean())
            # paired bootstrap CI (B=5000)
            B = 5000
            n = len(deltas)
            idx = rng.integers(0, n, size=(B, n))
            boot = deltas[idx].mean(axis=1)
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            # permutation p-value
            signs = np.array([1.0, -1.0])
            null = []
            for _ in range(2000):
                flip = rng.choice(signs, size=n)
                null.append(float((flip * deltas).mean()))
            null = np.array(null)
            if observed >= 0:
                p_perm = float((null >= observed).mean())
            else:
                p_perm = float((null <= observed).mean())
            paired_rows.append(
                {
                    "task": task,
                    "algo": "dr_grpo",
                    "n_pairs": n,
                    "delta_rho": observed,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_perm": p_perm,
                }
            )
        for r in paired_rows:
            w.writerow(
                [
                    "paired_delta_rho_drgrpo_minus_grpo",
                    r["task"],
                    r["algo"],
                    "",
                    "",
                    "",
                    "",
                    f"{r['delta_rho']:.6f}",
                    "",
                    "",
                    "",
                ]
            )
        for r in paired_rows:
            w.writerow(
                [
                    "paired_delta_ci",
                    r["task"],
                    r["algo"],
                    f"ci_lo={r['ci_lo']:.4f}",
                    f"ci_hi={r['ci_hi']:.4f}",
                    f"p_perm={r['p_perm']:.4f}",
                    "",
                    "",
                    r["n_pairs"],
                    "",
                    "",
                ]
            )
        for r in all_bin_rows:
            w.writerow(
                [
                    "per_run_bin",
                    r["task"],
                    r["algo"],
                    r["bin"],
                    f"{r['len_min']:.4f}",
                    f"{r['len_max']:.4f}",
                    f"{r['len_mid']:.4f}",
                    f"{r['mean_reward']:.6f}",
                    r["n_steps"],
                    r["seed"],
                    r["model"],
                ]
            )
        for (algo, task), c in curves.items():
            for b in c["per_bin"]:
                w.writerow(
                    [
                        "aggregate_curve",
                        task,
                        algo,
                        b["bin"],
                        "",
                        "",
                        f"{b['len_mid_mean']:.4f}",
                        f"{b['mean_reward']:.6f}",
                        "",
                        "",
                        "",
                    ]
                )
        # paired stats block
        for (algo, task), s in stats.items():
            w.writerow(
                [
                    "curve_stats",
                    task,
                    algo,
                    "",
                    "",
                    "",
                    f"{s['slope']:.6f}" if not math.isnan(s["slope"]) else "nan",
                    f"{s['rho']:.4f}" if not math.isnan(s["rho"]) else "nan",
                    "",
                    "",
                    "",
                ]
            )
            # add paired-delta row immediately after the second algo per task
        # paired-delta rows
        for task in {t for (_, t) in by_at.keys()}:
            g = stats.get(("grpo", task))
            d = stats.get(("dr_grpo", task))
            if g and d:
                d_slope = d["slope"] - g["slope"]
                d_rho = d["rho"] - g["rho"]
                w.writerow(
                    [
                        "paired_delta_drgrpo_minus_grpo",
                        task,
                        "dr_grpo",
                        "",
                        "",
                        "",
                        f"{d_slope:.6f}",
                        f"{d_rho:.4f}",
                        "",
                        "",
                        "",
                    ]
                )
    print(f"[iter16] wrote {OUT_TSV} ({OUT_TSV.stat().st_size} bytes)")

    # ---- figure ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[iter16] matplotlib unavailable, skipping figure: {e}")
        return

    FIG.mkdir(parents=True, exist_ok=True)
    tasks = sorted({t for (_, t) in by_at.keys()})
    algos = ["grpo", "dr_grpo"]
    colors = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, task in zip(axes, tasks):
        # scatter all per-run points
        for run in [r for r in runs if r["task"] == task]:
            L = [row["length"] for row in run["rows"]]
            R = [row["reward"] for row in run["rows"]]
            ax.scatter(
                L,
                R,
                s=12,
                alpha=0.18,
                color=colors[run["algo"]],
                edgecolors="none",
            )
        # overlay per-bin mean curve per algo
        for algo in algos:
            c = curves.get((algo, task))
            if not c:
                continue
            xs = [b["len_mid_mean"] for b in c["per_bin"]]
            ys = [b["mean_reward"] for b in c["per_bin"]]
            ax.plot(xs, ys, "o-", color=colors[algo], lw=2.0, ms=6, label=algo.upper())
        ax.set_title(task.replace("_", " "), fontsize=10)
        ax.set_xlabel("completion length (tokens)")
        ax.set_ylabel("per-step mean reward")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
        s_g = stats.get(("grpo", task), {})
        s_d = stats.get(("dr_grpo", task), {})
        if s_g and s_d:
            ax.text(
                0.02,
                0.98,
                f"rho_GRPO={s_g.get('rho', float('nan')):.3f}  slope={s_g.get('slope', float('nan')):.4f}\n"
                f"rho_DrGRPO={s_d.get('rho', float('nan')):.3f}  slope={s_d.get('slope', float('nan')):.4f}",
                transform=ax.transAxes,
                fontsize=7.5,
                va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", lw=0.4),
            )
    fig.suptitle(
        "Iter 16 — Length-Reward Shape E[R | L]: Dr.GRPO flattens the per-step curve",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"[iter16] wrote {OUT_FIG} ({OUT_FIG.stat().st_size} bytes)")
    print(f"[iter16] wrote {OUT_PNG} ({OUT_PNG.stat().st_size} bytes)")

    # ---- print summary ----
    print("\n[iter16] per-run Spearman rho(L, R) by (algo, task):")
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in per_run_stats:
        agg[(r["task"], r["algo"])].append(r["rho_len_rew"])
        print(
            f"  {r['task']:36s} {r['algo']:7s} seed={r['seed']:<6}  "
            f"rho={r['rho_len_rew']:+.3f}  slope={r['slope_len_rew']:+.4f}"
        )
    print("\n[iter16] aggregate mean rho per (task, algo):")
    for k, v in sorted(agg.items()):
        print(f"  {k[0]:36s} {k[1]:7s}  mean_rho={np.mean(v):+.3f}  sd={np.std(v, ddof=1):.3f}  n={len(v)}")

    print("\n[iter16] paired bootstrap Delta(Dr.GRPO - GRPO) rho(L,R):")
    for r in paired_rows:
        sig = "***" if r["p_perm"] < 0.01 else ("**" if r["p_perm"] < 0.05 else ("*" if r["p_perm"] < 0.1 else ""))
        print(
            f"  {r['task']:36s}  delta_rho={r['delta_rho']:+.3f}  CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
            f"p_perm={r['p_perm']:.3f} {sig}"
        )

    print("\n[iter16] bin-aggregated curve stats:")
    for (algo, task), s in sorted(stats.items()):
        print(
            f"  {task:36s} {algo:7s}  rho={s['rho']:+.3f}  p={s['p']:.3g}  "
            f"slope={s['slope']:+.4f}  n_seeds={s['n_seeds']}  n_bins={s['n_bins']}"
        )


if __name__ == "__main__":
    main()
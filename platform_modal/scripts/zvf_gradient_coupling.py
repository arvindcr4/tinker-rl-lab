#!/usr/bin/env python3
"""
zvf_gradient_coupling.py — Pillar 2 (ZVF) cross-pillar elevation, iter18.

Couples per-step ZVF with the gradient-flow proxies
(advantage_variance, mean_reward, entropy, grad_norm) recorded in
experiments/results/group_size_advantage_variance.tsv (G-sweep on
Qwen2.5-0.5B arithmetic, G in {2,4,8,16}, 3 seeds, 40 steps each).

Produces:
  - experiments/results/zvf_gradient_coupling.tsv   (per-G regression table)
  - experiments/results/zvf_gradient_coupling_pivot.tsv  (per-(G,step) stats)
  - figures/zvf_gradient_coupling.pdf
  - paper/sections/zvf_gradient.tex

The empirical claim is narrow and falsifiable: within the G-sweep on
Qwen2.5-0.5B, the per-step correlation between ZVF and grad_norm is
strongly negative (more zero-variance groups = less gradient flow), the
per-step correlation between ZVF and entropy is strongly negative
(less diversity = lower entropy), and the per-step correlation between
ZVF and mean_reward is non-monotone (U-shaped: ZVF is high both at
near-zero rewards and at near-one rewards, lowest at the learning
frontier). The first two are textbook predictions; the third is the
non-obvious one.
"""
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER = ROOT / "paper" / "sections"
FIGS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

SRC = RESULTS / "group_size_advantage_variance.tsv"


def load_rows():
    rows = []
    with SRC.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(
                dict(
                    G=int(r["G"]),
                    seed=int(r["seed"]),
                    step=int(r["step"]),
                    zvf=float(r["zvf"]),
                    advantage_variance=float(r["advantage_variance"]),
                    mean_reward=float(r["mean_reward"]),
                    entropy=float(r["entropy"]),
                    grad_norm=float(r["grad_norm"]),
                )
            )
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan"), float("nan"), float("nan")
    r = num / (dx * dy)
    return r, float("nan"), float("nan")  # CI is computed at the group level


def per_seed_corr_for_G(rows, G, col):
    """Compute the per-seed Pearson r between ZVF and `col` for a fixed G.

    Returns the list of per-seed r values. The bootstrap CI is then
    computed by resampling over seeds (which respects the temporal
    structure within each seed)."""
    out = []
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["G"] == G and r["seed"] == seed]
        if len(sub) < 5:
            continue
        xs = [r["zvf"] for r in sub]
        ys = [r[col] for r in sub]
        r, _, _ = pearson(xs, ys)
        out.append(r)
    return out


def block_bootstrap_ci(values, B=2000, seed=0):
    """Resample-with-replacement over the supplied list (per-seed correlations)."""
    import random
    random.seed(seed)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    boots = []
    for _ in range(B):
        sample = [values[random.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return lo, hi


def per_step_corr_table(rows):
    """Pooled per-(G,seed) Pearson between ZVF and each gradient-flow proxy."""
    out = []
    for G in sorted({r["G"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows}):
            sub = [r for r in rows if r["G"] == G and r["seed"] == seed]
            if len(sub) < 5:
                continue
            xs = [r["zvf"] for r in sub]
            for col, label in [
                ("grad_norm", "grad_norm"),
                ("advantage_variance", "advantage_variance"),
                ("entropy", "entropy"),
                ("mean_reward", "mean_reward"),
            ]:
                ys = [r[col] for r in sub]
                r, lo, hi = pearson(xs, ys)
                out.append(
                    dict(
                        G=G,
                        seed=seed,
                        proxy=label,
                        pearson_r=round(r, 4),
                        ci_lo=round(lo, 4),
                        ci_hi=round(hi, 4),
                        n_steps=len(sub),
                    )
                )
    return out


def per_G_pooled(rows):
    """Pool across seeds within G. Point estimate = mean of per-seed Pearson r.
    CI = block bootstrap over seeds (preserves per-seed temporal structure)."""
    out = []
    for G in sorted({r["G"] for r in rows}):
        for col, label in [
            ("grad_norm", "grad_norm"),
            ("advantage_variance", "advantage_variance"),
            ("entropy", "entropy"),
            ("mean_reward", "mean_reward"),
        ]:
            per_seed = per_seed_corr_for_G(rows, G, col)
            per_seed = [v for v in per_seed if not math.isnan(v)]
            if not per_seed:
                continue
            mean_r = sum(per_seed) / len(per_seed)
            lo, hi = block_bootstrap_ci(per_seed, B=2000, seed=G * 31 + 7)
            out.append(
                dict(
                    G=G,
                    proxy=label,
                    pearson_r=round(mean_r, 4),
                    ci_lo=round(lo, 4),
                    ci_hi=round(hi, 4),
                    n_obs=len(sub for sub in [r for r in rows if r["G"] == G])
                    if False
                    else len([r for r in rows if r["G"] == G]),
                    n_seeds=len(per_seed),
                )
            )
    return out


def per_step_stats(rows):
    """For figure: per (G, step) mean and SEM of each variable across seeds."""
    out = []
    by_g_step = defaultdict(list)
    for r in rows:
        by_g_step[(r["G"], r["step"])].append(r)
    for (G, step), sub in sorted(by_g_step.items()):
        d = dict(G=G, step=step, n_seeds=len(sub))
        for col, label in [
            ("zvf", "zvf"),
            ("grad_norm", "grad_norm"),
            ("entropy", "entropy"),
            ("mean_reward", "mean_reward"),
            ("advantage_variance", "advantage_variance"),
        ]:
            vs = [r[col] for r in sub]
            d[f"mean_{label}"] = round(statistics.mean(vs), 4)
            d[f"sem_{label}"] = round(
                statistics.stdev(vs) / math.sqrt(len(vs)) if len(vs) > 1 else 0.0, 4
            )
        out.append(d)
    return out


def write_tsv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tex(out_path, per_g_pooled_rows, per_step_stats_rows, header=""):
    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return "NA"
            return f"{x:+.3f}"
        return str(x)

    lines = []
    lines.append(header)
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{lrrrr}"
    )
    lines.append(r"\toprule")
    lines.append(r"$G$ & proxy & $r_{\text{Pearson}}$ & 95\% boot.\ CI & $n$ \\")
    lines.append(r"\midrule")
    for r in per_g_pooled_rows:
        lines.append(
            f"{r['G']} & {r['proxy']} & {fmt(r['pearson_r'])} & "
            f"$[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]$ & {r['n_obs']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Per-step Pearson correlation between ZVF and "
        r"gradient-flow proxies, pooled across seeds within each $G$. "
        r"Source: \texttt{experiments/results/zvf\_gradient\_coupling.tsv}.}"
    )
    lines.append(r"\label{tab:zvf-gradient-coupling}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def make_figure(rows, out_pdf):
    by_g_step = defaultdict(list)
    for r in rows:
        by_g_step[(r["G"], r["step"])].append(r)

    Gs = sorted({r["G"] for r in rows})
    colors = {2: "#1f77b4", 4: "#2ca02c", 8: "#d62728", 16: "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # Panel A: ZVF vs grad_norm scatter (per-G), per-step means
    ax = axes[0]
    for G in Gs:
        pts = sorted(by_g_step.items(), key=lambda kv: (kv[0][1] if kv[0][0] == G else 0))
        ps = [
            (statistics.mean([r["zvf"] for r in sub]),
             statistics.mean([r["grad_norm"] for r in sub]))
            for (gg, _), sub in pts
            if gg == G
        ]
        ps.sort()
        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        ax.plot(xs, ys, "-o", color=colors[G], label=f"G={G}", markersize=4)
    ax.set_xlabel("mean ZVF (per step)")
    ax.set_ylabel("mean grad_norm (per step)")
    ax.set_title("(A) ZVF vs grad_norm")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Panel B: ZVF vs entropy
    ax = axes[1]
    for G in Gs:
        ps = sorted(
            [
                (statistics.mean([r["zvf"] for r in sub]),
                 statistics.mean([r["entropy"] for r in sub]))
                for (gg, _), sub in by_g_step.items()
                if gg == G
            ]
        )
        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        ax.plot(xs, ys, "-o", color=colors[G], label=f"G={G}", markersize=4)
    ax.set_xlabel("mean ZVF (per step)")
    ax.set_ylabel("mean entropy (per step)")
    ax.set_title("(B) ZVF vs entropy")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Panel C: ZVF vs mean_reward (the U-shape claim)
    ax = axes[2]
    for G in Gs:
        ps = sorted(
            [
                (statistics.mean([r["zvf"] for r in sub]),
                 statistics.mean([r["mean_reward"] for r in sub]))
                for (gg, _), sub in by_g_step.items()
                if gg == G
            ]
        )
        xs = [p[0] for p in ps]
        ys = [p[1] for p in ps]
        ax.plot(xs, ys, "-o", color=colors[G], label=f"G={G}", markersize=4)
    ax.set_xlabel("mean ZVF (per step)")
    ax.set_ylabel("mean reward (per step)")
    ax.set_title("(C) ZVF vs mean reward (U-shape)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_rows()
    per_step = per_step_corr_table(rows)
    per_g = per_G_pooled(rows)
    stats = per_step_stats(rows)
    write_tsv(
        RESULTS / "zvf_gradient_coupling.tsv",
        per_step,
        ["G", "seed", "proxy", "pearson_r", "ci_lo", "ci_hi", "n_steps"],
    )
    write_tsv(
        RESULTS / "zvf_gradient_coupling_pooled.tsv",
        per_g,
        ["G", "proxy", "pearson_r", "ci_lo", "ci_hi", "n_obs", "n_seeds"],
    )
    write_tsv(
        RESULTS / "zvf_gradient_coupling_pivot.tsv",
        stats,
        [
            "G",
            "step",
            "n_seeds",
            "mean_zvf",
            "sem_zvf",
            "mean_grad_norm",
            "sem_grad_norm",
            "mean_entropy",
            "sem_entropy",
            "mean_mean_reward",
            "sem_mean_reward",
            "mean_advantage_variance",
            "sem_advantage_variance",
        ],
    )
    make_figure(rows, FIGS / "zvf_gradient_coupling.pdf")
    write_tex(
        PAPER / "zvf_gradient.tex",
        per_g,
        stats,
        header=(
            "% paper/sections/zvf_gradient.tex\n"
            "%\n"
            "% Pillar 2 (ZVF) cross-pillar elevation, iter18: ZVF x gradient-flow\n"
            "% coupling on the G-sweep (Qwen2.5-0.5B / synthetic arithmetic).\n"
            "% Source: scripts/zvf_gradient_coupling.py\n"
        ),
    )
    # Quick console summary
    print("Per-G pooled Pearson(ZVF, proxy):")
    for r in per_g:
        print(
            f"  G={r['G']:2d}  {r['proxy']:>20s}  r={r['pearson_r']:+.3f}  "
            f"CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={r['n_obs']}"
        )


if __name__ == "__main__":
    main()

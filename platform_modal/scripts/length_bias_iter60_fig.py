"""Iter 60 figure — length-elasticity of reward visualisation.

Three panels:

(A) Per-step elasticity trace on GSM8K CoT. For each seed, scatter
    (L_t, epsilon_t) where epsilon = dR/dL (centred, smoothed).
    GRPO = blue, Dr.GRPO = red. Identifies the "productive region"
    (epsilon > 0) vs "destructive region" (epsilon < 0).

(B) Quadratic R(L) fit on GSM8K CoT. For each (algo, seed), plot the
    fitted parabola and the actual (L_t, R_t) points. Highlight L*
    (parabolic optimum) and the iso-reward band.

(C) Negative-elasticity fraction paired bar chart. Per-task bar of
    GRPO vs Dr.GRPO negative-elasticity fraction with bootstrap CI
    error bars.

Outputs:
  figures/length_bias_iter60_elasticity.pdf
  paper/figures/length_bias_iter60_elasticity.pdf (mirror)
"""
from __future__ import annotations
import csv
import json
import math
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
PAPER_FIG = ROOT / "paper" / "figures"
DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"

RNG_SEED = 60
SMOOTH_K = 3
N_BOOT = 4000

ALGO_COLORS = {"grpo": "#1f77b4", "dr_grpo": "#d62728", "drgrpo": "#d62728"}


def centred_smooth(xs, k=SMOOTH_K):
    n = len(xs)
    half = k // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def compute_elasticity(R, L):
    n = len(R)
    eps = [0.0] * n
    for i in range(n):
        if i == 0:
            dR, dL = R[1] - R[0], L[1] - L[0]
        elif i == n - 1:
            dR, dL = R[-1] - R[-2], L[-1] - L[-2]
        else:
            dR = (R[i + 1] - R[i - 1]) / 2.0
            dL = (L[i + 1] - L[i - 1]) / 2.0
        if abs(dL) < 1e-6:
            eps[i] = float("nan")
        else:
            eps[i] = dR / dL
    return eps


def load_runs(path, task):
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        ts = [int(s["step"]) for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        out.append({"task": task, "algo": r["algo"], "seed": r["seed"],
                    "t": ts, "R": rs, "L": ls})
    return out


def fit_quadratic_R_of_L(R, L):
    A = [[ls * ls, ls, 1.0] for ls in L]
    n = len(R)
    AtA = [[0.0] * 3 for _ in range(3)]
    Atb = [0.0, 0.0, 0.0]
    for i in range(n):
        for r in range(3):
            for c in range(3):
                AtA[r][c] += A[i][r] * A[i][c]
            Atb[r] += A[i][r] * R[i]
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det3(AtA)
    if abs(d) < 1e-12:
        return {"a": 0.0, "L_star": float("nan"), "b": 0.0}
    cof = [[0.0] * 3 for _ in range(3)]
    cof[0][0] = AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]
    cof[0][1] = -(AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0])
    cof[0][2] = AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]
    cof[1][0] = -(AtA[0][1] * AtA[2][2] - AtA[0][2] * AtA[2][1])
    cof[1][1] = AtA[0][0] * AtA[2][2] - AtA[0][2] * AtA[2][0]
    cof[1][2] = -(AtA[0][0] * AtA[2][1] - AtA[0][1] * AtA[2][0])
    cof[2][0] = AtA[0][1] * AtA[1][2] - AtA[0][2] * AtA[1][1]
    cof[2][1] = -(AtA[0][0] * AtA[1][2] - AtA[0][2] * AtA[1][0])
    cof[2][2] = AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0]
    inv = [[cof[c][r] / d for c in range(3)] for r in range(3)]
    x = [sum(inv[r][c] * Atb[c] for c in range(3)) for r in range(3)]
    alpha, beta, gamma = x
    a = alpha
    L_star = -beta / (2.0 * a) if abs(a) > 1e-12 else float("nan")
    return {"a": a, "L_star": L_star, "b": gamma}


def main():
    rows = []
    rows.extend(load_runs(DRGRPO_JSON, "arithmetic_easy"))
    rows.extend(load_runs(GSM_JSON, "gsm8k_cot"))

    gsm = [r for r in rows if r["task"] == "gsm8k_cot"]
    ari = [r for r in rows if r["task"] == "arithmetic_easy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: elasticity trace (L_t, epsilon_t) on GSM8K CoT
    ax = axes[0]
    for r in gsm:
        algo = "grpo" if r["algo"] == "grpo" else "dr_grpo"
        R_s = centred_smooth(r["R"], SMOOTH_K)
        L_s = centred_smooth(r["L"], SMOOTH_K)
        eps = compute_elasticity(R_s, L_s)
        # filter finite
        L_f = [r["L"][i] for i in range(len(eps))
               if math.isfinite(eps[i]) and abs(eps[i]) < 1.0]
        e_f = [eps[i] for i in range(len(eps))
               if math.isfinite(eps[i]) and abs(eps[i]) < 1.0]
        ax.scatter(L_f, e_f, c=ALGO_COLORS[algo], s=18, alpha=0.4,
                   edgecolors="none", label=algo if r["seed"] == 42 else None)
    ax.axhline(0, color="k", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Response length L (tokens)")
    ax.set_ylabel(r"Marginal elasticity $\epsilon = \Delta R / \Delta L$")
    ax.set_title("(A) GSM8K CoT: per-step elasticity")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=ALGO_COLORS["grpo"],
                          markersize=8, label="GRPO"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=ALGO_COLORS["dr_grpo"],
                          markersize=8, label="Dr.GRPO")]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel B: parabolic fit on GSM8K CoT
    ax = axes[1]
    L_min_data = min(min(r["L"]) for r in gsm)
    L_max_data = max(max(r["L"]) for r in gsm)
    L_grid = [L_min_data + (L_max_data - L_min_data) * i / 200
              for i in range(201)]
    fits = {}
    for r in gsm:
        algo = "grpo" if r["algo"] == "grpo" else "dr_grpo"
        fit = fit_quadratic_R_of_L(r["R"], r["L"])
        fits.setdefault(algo, []).append(fit)
        ax.scatter(r["L"], r["R"], c=ALGO_COLORS[algo], s=12, alpha=0.4,
                   edgecolors="none")
    # plot mean parabola per algo
    for algo, color in (("grpo", ALGO_COLORS["grpo"]),
                        ("dr_grpo", ALGO_COLORS["dr_grpo"])):
        if algo not in fits:
            continue
        a_mean = sum(f["a"] for f in fits[algo]) / len(fits[algo])
        b_mean = sum(f["b"] for f in fits[algo]) / len(fits[algo])
        # reconstruct mean beta from mean L_star and mean a
        L_star_mean = sum(f["L_star"] for f in fits[algo]
                          if math.isfinite(f["L_star"])) / max(
            1, sum(1 for f in fits[algo] if math.isfinite(f["L_star"])))
        # mean parabola using mean a and mean L_star
        if math.isfinite(L_star_mean):
            R_grid = [a_mean * (l - L_star_mean) ** 2
                      + max(0.0, 0.5 - a_mean * L_star_mean ** 2)
                      for l in L_grid]
        else:
            R_grid = [a_mean * l * l + b_mean for l in L_grid]
        ax.plot(L_grid, R_grid, c=color, linewidth=2.0, alpha=0.8,
                label=f"{algo} (mean fit, $L^\\star$={L_star_mean:.0f})"
                if math.isfinite(L_star_mean) else f"{algo} (mean fit)")
    ax.set_xlabel("Response length L (tokens)")
    ax.set_ylabel("Reward R")
    ax.set_title("(B) GSM8K CoT: $R(L)$ quadratic fit")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: negative-elasticity fraction paired bar (cross-task)
    ax = axes[2]
    # read from TSV
    summary = {}
    with open(RES / "length_bias_iter60_summary.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["metric"] == "neg_frac":
                summary[row["task"]] = row
    tasks = ["arithmetic_easy", "gsm8k_cot"]
    x = list(range(len(tasks)))
    width = 0.35
    grpo_vals = [float(summary[t]["mean_grpo"]) for t in tasks
                 if t in summary]
    drgrpo_vals = [float(summary[t]["mean_drgrpo"]) for t in tasks
                   if t in summary]
    grpo_errs = [
        float(summary[t]["mean_grpo"]) - float(summary[t]["ci_lo"])
        for t in tasks if t in summary]
    drgrpo_errs = [
        float(summary[t]["mean_drgrpo"]) - float(summary[t]["ci_lo"])
        for t in tasks if t in summary]
    ax.bar([xi - width / 2 for xi in x], grpo_vals, width,
           color=ALGO_COLORS["grpo"], alpha=0.7,
           yerr=grpo_errs, label="GRPO", capsize=4)
    ax.bar([xi + width / 2 for xi in x], drgrpo_vals, width,
           color=ALGO_COLORS["dr_grpo"], alpha=0.7,
           yerr=drgrpo_errs, label="Dr.GRPO", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.set_ylabel("Negative-elasticity fraction")
    ax.set_title("(C) Negative-$\\epsilon$ steps (higher = worse)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(grpo_vals + drgrpo_vals) * 1.2)

    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / "length_bias_iter60_elasticity.pdf", dpi=150)
    plt.savefig(FIG / "length_bias_iter60_elasticity.png", dpi=150)
    plt.savefig(PAPER_FIG / "length_bias_iter60_elasticity.pdf", dpi=150)
    plt.close(fig)
    print("Wrote figures/length_bias_iter60_elasticity.pdf")


if __name__ == "__main__":
    main()
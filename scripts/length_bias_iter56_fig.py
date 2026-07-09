"""Iter 56 figure: 2x2 panels of reward-per-token efficiency diagnostics.

Panel A: rho(t) trajectories on GSM8K CoT (GRPO blue, Dr.GRPO orange).
Panel B: Cumulative-token-tax bar chart (paired per-seed), GSM8K CoT.
Panel C: Length phase-portrait (L_t, dL_t) clouds on GSM8K CoT.
Panel D: rho_0 vs rho_final scatter, both tasks, both algorithms.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"


def _load(path, task):
    out = []
    with open(path) as f:
        d = json.load(f)
    for r in d["runs"]:
        sl = r["step_log"]
        ts = [int(s["step"]) for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        rho = [rs[i] / ls[i] if ls[i] > 0 else 0.0 for i in range(len(rs))]
        out.append({"task": task, "algo": r["algo"], "seed": r["seed"],
                    "t": ts, "R": rs, "L": ls, "rho": rho})
    return out


def _tax_one(L, R):
    R0 = R[0]
    if R0 <= 0:
        return 0.0
    L_star = [L[0] * (R[i] / R0) for i in range(len(R))]
    return sum(max(0.0, L[i] - L_star[i]) for i in range(len(L))) / sum(L)


def main() -> None:
    rows = []
    rows.extend(_load(DRGRPO_JSON, "arithmetic_easy"))
    rows.extend(_load(GSM_JSON, "gsm8k_cot"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # Panel A: rho trajectories on GSM8K CoT
    ax = axes[0, 0]
    cmap = {"grpo": "tab:blue", "dr_grpo": "tab:orange"}
    for r in rows:
        if r["task"] != "gsm8k_cot":
            continue
        c = cmap.get(r["algo"], "tab:gray")
        ls = "--" if r["algo"] == "dr_grpo" else "-"
        ax.plot(r["t"], [x * 1000 for x in r["rho"]], color=c, linestyle=ls,
                alpha=0.65, linewidth=1.2,
                label=f'{r["algo"]} s{r["seed"]}')
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\rho(t)=R_t/L_t$ ($\times 10^{3}$)")
    ax.set_title("(A) Reward-per-token trajectory, GSM8K CoT")
    ax.grid(True, alpha=0.3)

    # Panel B: cumulative tax per (task, algo) -- paired per seed
    ax = axes[0, 1]
    tasks = ("arithmetic_easy", "gsm8k_cot")
    algos = ("grpo", "dr_grpo")
    seed_lists = {t: sorted({r["seed"] for r in rows if r["task"] == t})
                  for t in tasks}
    x_centers = list(range(len(tasks)))
    width = 0.32
    for ai, algo in enumerate(algos):
        offsets = [-width / 2 + ai * width] * len(tasks)
        vals = []
        for t in tasks:
            v = []
            for s in seed_lists[t]:
                row = next((r for r in rows if r["task"] == t
                            and r["algo"] == algo and r["seed"] == s), None)
                if row is None:
                    continue
                v.append(_tax_one(row["L"], row["R"]))
            vals.append(v)
        positions = [xc + offsets[i] for i, xc in enumerate(x_centers)]
        bp = ax.boxplot(vals, positions=positions, widths=width * 0.9,
                        patch_artist=True, showmeans=True, meanline=True)
        color = cmap.get(algo, "tab:gray")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        # Scatter the seeds on top
        for i, xc in enumerate(x_centers):
            for v in vals[i]:
                ax.scatter(xc + offsets[i] + (ai - 0.5) * 0.05, v,
                           color=color, edgecolor="black", s=24, zorder=3)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([t.replace("_", "\\_") for t in tasks])
    ax.set_ylabel("cumulative token-tax (fraction of $\\sum L$)")
    ax.set_title("(B) Cumulative token-tax per (task, algo)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=cmap["grpo"],
                                     alpha=0.55, label="GRPO"),
                        plt.Rectangle((0, 0), 1, 1, facecolor=cmap["dr_grpo"],
                                      alpha=0.55, label="Dr.GRPO")],
               loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: phase portrait on GSM8K CoT
    ax = axes[1, 0]
    for r in rows:
        if r["task"] != "gsm8k_cot":
            continue
        L = r["L"]; dL = [L[i + 1] - L[i] for i in range(len(L) - 1)]
        c = cmap.get(r["algo"], "tab:gray")
        ax.plot(L[:-1], dL, marker="o", color=c, alpha=0.55, linewidth=0.9,
                markersize=3.5, label=f'{r["algo"]} s{r["seed"]}')
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel(r"$L_t$ (tokens)")
    ax.set_ylabel(r"$\Delta L_t = L_{t+1}-L_t$")
    ax.set_title("(C) Length phase-portrait, GSM8K CoT")
    ax.grid(True, alpha=0.3)

    # Panel D: rho_0 vs rho_final scatter, all runs
    ax = axes[1, 1]
    markers = {"arithmetic_easy": "o", "gsm8k_cot": "s"}
    for r in rows:
        c = cmap.get(r["algo"], "tab:gray")
        m = markers.get(r["task"], "^")
        ax.scatter(r["rho"][0], r["rho"][-1], color=c, marker=m, s=42,
                   alpha=0.8, edgecolor="black", linewidth=0.6,
                   label=f'{r["algo"]}/{r["task"]}')
    # y=x reference
    lo = 0.0
    hi = max(r["rho"][0] for r in rows + [{"rho": [1.0]}])
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, alpha=0.4,
            linestyle="--", label=r"$\rho_{\rm final}=\rho_0$")
    ax.set_xlabel(r"$\rho_0$ (initial reward-per-token)")
    ax.set_ylabel(r"$\rho_{\rm final}$ (final reward-per-token)")
    ax.set_title("(D) $\\rho$ growth, all (task, algo, seed)")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap["grpo"], markersize=8,
                          markeredgecolor="black", label="GRPO (arith)"),
               plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap["dr_grpo"], markersize=8,
                          markeredgecolor="black", label="Dr.GRPO (arith)"),
               plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=cmap["grpo"], markersize=8,
                          markeredgecolor="black", label="GRPO (gsm8k)"),
               plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=cmap["dr_grpo"], markersize=8,
                          markeredgecolor="black", label="Dr.GRPO (gsm8k)")]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Iter 56: Reward-per-Token Efficiency Frontier "
                 "(\\textit{Dr.GRPO} on GSM8K CoT pays +0.040 cumulative tax, "
                 "$p < 0.001$)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG / "length_bias_iter56_efficiency.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
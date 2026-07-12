#!/usr/bin/env python3
"""Iter 80 figure: OU length-dynamics phase view + unit-root falsification.

Panel A: mean length trajectories (GRPO vs Dr.GRPO) on GSM8K CoT with the fitted
         OU equilibrium mu drawn as a dashed attractor line per algorithm.
Panel B: Dickey-Fuller stat per seed vs the 5% unit-root critical line -- shows
         GRPO sits closer to the unit-root boundary (weaker mean reversion) than
         Dr.GRPO on the hard CoT task, while both clear it decisively on arithmetic.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "experiments", "results")
FIG = os.path.join(W, "figures")
BURN = 2
CRIT5 = -2.93
CG, CD = "#c0392b", "#2471a3"  # grpo red, dr_grpo blue


def load(fname):
    d = json.load(open(os.path.join(RES, fname)))
    out = {}
    for r in d["runs"]:
        L = [s["mean_comp_len"] for s in r["step_log"]]
        out.setdefault(r["algo"], []).append(L)
    return out


def perrun():
    rows = []
    with open(os.path.join(RES, "length_bias_iter80_perrun.tsv")) as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(h, line.rstrip("\n").split("\t"))))
    return rows


def summ():
    rows = {}
    with open(os.path.join(RES, "length_bias_iter80_summary.tsv")) as f:
        h = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(h, line.rstrip("\n").split("\t")))
            rows[(r["task"], r["algo"])] = r
    return rows


def main():
    cot = load("drgrpo_gsm8k_cot_full.json")
    S = summ()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))

    # Panel A: trajectories + equilibrium
    for algo, col, lab in [("grpo", CG, "GRPO"), ("dr_grpo", CD, "Dr.GRPO")]:
        arr = np.array(cot[algo])
        t = np.arange(arr.shape[1])
        m = arr.mean(0)
        axA.plot(t, m, color=col, lw=2.2, label=lab)
        axA.fill_between(t, arr.min(0), arr.max(0), color=col, alpha=0.12)
        mu = float(S[("gsm8k_cot", algo)]["mu"])
        axA.axhline(mu, color=col, ls="--", lw=1.2, alpha=0.8)
        axA.text(t[-1], mu, f" $\\mu$={mu:.0f}", color=col, va="center", fontsize=8)
    axA.axvspan(0, BURN - 0.5, color="grey", alpha=0.10)
    axA.set_xlabel("training step")
    axA.set_ylabel("mean completion length (tokens)")
    axA.set_title("A  GSM8K CoT: length trajectories & OU equilibrium $\\mu$")
    axA.legend(loc="upper right", fontsize=9)

    # Panel B: DF stat per seed
    rows = perrun()
    tasks = ["arithmetic", "gsm8k_cot"]
    xpos = {"arithmetic": 0, "gsm8k_cot": 1}
    off = {"grpo": -0.12, "dr_grpo": +0.12}
    for r in rows:
        try:
            df = float(r["df_stat"])
        except ValueError:
            continue
        x = xpos[r["task"]] + off[r["algo"]]
        axB.scatter(x, df, color=(CG if r["algo"] == "grpo" else CD),
                    s=55, zorder=3, edgecolor="white", linewidth=0.6)
    # algo means
    for r in [S[(tk, al)] for tk in tasks for al in ("grpo", "dr_grpo")]:
        x = xpos[r["task"]] + off[r["algo"]]
        axB.plot([x - 0.06, x + 0.06], [float(r["df_stat"])] * 2,
                 color="black", lw=2.4, zorder=4)
    axB.axhline(CRIT5, color="darkgreen", ls="--", lw=1.4)
    axB.text(1.32, CRIT5 + 0.05, "5% unit-root\ncritical line", color="darkgreen",
             fontsize=8, va="bottom", ha="right")
    axB.annotate("closer to unit root\n(weaker reversion)", xy=(1.0, -3.0),
                 xytext=(0.35, -1.4), fontsize=8, color="dimgrey",
                 arrowprops=dict(arrowstyle="->", color="dimgrey"))
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["arithmetic\n(0.5B)", "GSM8K CoT\n(1.5B)"])
    axB.set_ylabel("Dickey-Fuller stat (lower = more stationary)")
    axB.set_title("B  Unit-root test on length level (per seed)")
    from matplotlib.lines import Line2D
    axB.legend(handles=[Line2D([], [], marker="o", ls="", color=CG, label="GRPO"),
                        Line2D([], [], marker="o", ls="", color=CD, label="Dr.GRPO")],
               loc="lower left", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"length_bias_iter80.{ext}"), dpi=140,
                    bbox_inches="tight")
    print("wrote figures/length_bias_iter80.pdf / .png")


if __name__ == "__main__":
    main()

"""Deep Review 1ed8fa8c audit of the measured P3 group-size sweep.

Recomputes, from the per-run step logs in groupsize_zvf_sweep.json:
  * the ZVF closed-form baseline on the same training rollouts that produce
    the empirical ZVF (per-step pass rate, not terminal held-out accuracy);
  * TOST equivalence on per-seed last-10 reward with seeds as the unit
    (n=3 per arm; Welch-Satterthwaite df, and paired df=2 for reference);
  * paired per-seed retention ratios vs G=2;
  * Table 4 power values for n1=5, n2=15;
  * the measured panels of the iter-27 figure (reward vs G, and the fraction of
    steps whose batch has a non-degenerate group, which the old "SNR(adv)"
    column transformed as sqrt(f/(1-f))).

Writes experiments/results/group_size_review_1ed8_*.tsv. No new training.
"""

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

RESULTS = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments/results"
FIGURES = RESULTS.parents[1] / "paper/figures"
G_VALUES = (2, 4, 8, 16)
EPS = (0.01, 0.02, 0.05)


def write(name, rows):
    path = RESULTS / f"group_size_review_1ed8_{name}.tsv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}")


def welch_tost(a, b, eps):
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    d = a.mean() - b.mean()
    p = max(stats.t.sf((d + eps) / se, df), stats.t.cdf((d - eps) / se, df))
    return d, se, df, p


def paired_tost(a, b, eps):
    diff = a - b
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    df = diff.size - 1
    d = diff.mean()
    return max(stats.t.sf((d + eps) / se, df), stats.t.cdf((d - eps) / se, df))


def main():
    runs = json.load(open(RESULTS / "groupsize_zvf_sweep.json"))["runs"]
    by_g = {
        g: sorted((r for r in runs if r["group_size"] == g), key=lambda r: r["seed"])
        for g in G_VALUES
    }

    # (1) ZVF: empirical vs closed form on the same per-step training rollouts.
    zrows = []
    for g in G_VALUES:
        emp, homog, term = [], [], []
        for r in by_g[g]:
            steps = r["step_log"]
            emp.append(np.mean([s["zvf"] for s in steps]))
            homog.append(
                np.mean([s["mean_reward"] ** g + (1 - s["mean_reward"]) ** g for s in steps])
            )
            term.append(r["heldout_acc"] ** g + (1 - r["heldout_acc"]) ** g)
        emp, homog, term = map(np.mean, (emp, homog, term))
        zrows.append(
            {
                "G": g,
                "mean_train_reward": round(
                    float(
                        np.mean(
                            [np.mean([s["mean_reward"] for s in r["step_log"]]) for r in by_g[g]]
                        )
                    ),
                    4,
                ),
                "zvf_empirical": round(float(emp), 4),
                "zvf_theory_per_step_train_p": round(float(homog), 4),
                "residual_per_step": round(float(emp - homog), 4),
                "zvf_theory_terminal_heldout_p_legacy": round(float(term), 4),
                "residual_legacy": round(float(emp - term), 4),
            }
        )
    write("zvf_theory", zrows)

    # (2) TOST with seeds as the unit.
    last10 = {
        g: np.array([np.mean([s["mean_reward"] for s in r["step_log"][-10:]]) for r in by_g[g]])
        for g in G_VALUES
    }
    trows = []
    for ga, gb in combinations(G_VALUES, 2):
        a, b = last10[ga], last10[gb]
        for eps in EPS:
            d, se, df, p = welch_tost(a, b, eps)
            trows.append(
                {
                    "G_a": ga,
                    "G_b": gb,
                    "n_a": a.size,
                    "n_b": b.size,
                    "mean_diff": round(float(d), 5),
                    "se": round(float(se), 5),
                    "welch_df": round(float(df), 2),
                    "epsilon": eps,
                    "tost_p_welch": f"{p:.4g}",
                    "tost_p_paired_df2": f"{paired_tost(a, b, eps):.4g}",
                    "equivalent_welch_0.05": "yes" if p < 0.05 else "no",
                }
            )
    write("tost", trows)

    # (3) Paired per-seed retention vs G=2 (G=2 vs itself is exactly 1).
    rrows = []
    for g in G_VALUES:
        ratio = last10[g] / last10[2]
        rrows.append(
            {
                "G": g,
                "mean_last10": round(float(last10[g].mean()), 4),
                "retention_vs_G2_mean": round(float(ratio.mean()), 4),
                "retention_min": round(float(ratio.min()), 4),
                "retention_max": round(float(ratio.max()), 4),
            }
        )
    write("retention", rrows)

    # (4) Table 4 power, n1=5 vs n2=15 (pooled-variance reference).
    pw = TTestIndPower()
    prow = [
        {
            "d": d,
            "power_5v5": round(pw.power(d, nobs1=5, ratio=1.0, alpha=0.05), 3),
            "power_5v15": round(pw.power(d, nobs1=5, ratio=3.0, alpha=0.05), 3),
        }
        for d in (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
    ]
    prow.append(
        {
            "d": "MDE",
            "power_5v5": round(pw.solve_power(nobs1=5, ratio=1.0, alpha=0.05, power=0.8), 3),
            "power_5v15": round(pw.solve_power(nobs1=5, ratio=3.0, alpha=0.05, power=0.8), 3),
        }
    )
    write("power", prow)

    # (5) Measured panels of the iter-27 figure.
    frac = {
        g: np.array(
            [np.mean([s["advantage_variance"] > 0.5 for s in r["step_log"]]) for r in by_g[g]]
        )
        for g in G_VALUES
    }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    gs = np.array(G_VALUES)
    m = np.array([last10[g].mean() for g in G_VALUES])
    se = np.array([last10[g].std(ddof=1) / np.sqrt(last10[g].size) for g in G_VALUES])
    for g in G_VALUES:
        ax_a.scatter([g] * last10[g].size, last10[g], color="#1f77b4", alpha=0.35, s=18)
    ax_a.errorbar(
        gs,
        m,
        yerr=1.96 * se,
        fmt="o-",
        color="#1f77b4",
        capsize=5,
        lw=2,
        label="last-10 reward (mean $\\pm$ 1.96 SE)",
    )
    ax_a.axhline(m[0] * 0.976, color="gray", ls="--", lw=1.3, label="97.6% of G=2 (Wu et al.)")
    for g, mi in zip(G_VALUES, m, strict=False):
        ax_a.annotate(
            f"{np.mean(last10[g] / last10[2]) * 100:.1f}%",
            (g, mi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_a.set(
        xscale="log",
        xlabel="Group size G",
        ylabel="Last-10 training reward",
        title="(A) Reward vs G (3 seeds per G)",
    )
    ax_a.set_xticks(gs, [str(g) for g in G_VALUES])
    ax_a.legend(fontsize=8, loc="lower left")
    ax_a.grid(alpha=0.3)
    for g in G_VALUES:
        ax_b.scatter([g] * frac[g].size, frac[g], color="#2ca02c", alpha=0.4, s=18)
    ax_b.plot(
        gs, [frac[g].mean() for g in G_VALUES], "o-", color="#2ca02c", lw=2, label="mean over seeds"
    )
    ax_b.set(
        xscale="log",
        xlabel="Group size G",
        ylabel="Fraction of steps",
        ylim=(0.5, 1.0),
        title="(B) Steps with $\\geq$1 non-degenerate group",
    )
    ax_b.set_xticks(gs, [str(g) for g in G_VALUES])
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.grid(alpha=0.3)
    for ax in (ax_a, ax_b):
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(FIGURES / "group_size_iter27_measured.png", dpi=200)
    print("wrote group_size_iter27_measured.png")


if __name__ == "__main__":
    main()

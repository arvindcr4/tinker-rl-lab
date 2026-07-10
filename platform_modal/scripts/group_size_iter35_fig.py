#!/usr/bin/env python3
"""Figure for iter35: 3-panel cross-scale G=4..G=64 audit.

Panels:
  (Left)  Retention vs G_a for fixed G_b=32 at all 4 budgets, with the
          Wu 97.6% band. Shows the retention curve as a function of
          G_a (so the per-G_a retention at G_b=32 is a single curve
          per T).
  (Mid)   DPO-equivalence score heatmap (G_a x G_b) at T=64M, with the
          Wu 97.6% cells annotated.
  (Right) Cost-effectiveness Pareto frontier: acc per million tokens
          vs G at all 4 budgets.

The figure is saved as figures/group_size_iter35.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

WU_RETENTION = 0.976


def main() -> None:
    pair = pd.read_csv(RES / "group_size_iter35_pair_sweep.tsv", sep="\t")
    pareto = pd.read_csv(RES / "group_size_iter35_pareto.tsv", sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # ----- Left panel: retention vs G_a at fixed G_b=32 (one curve per T) -----
    ax = axes[0]
    budgets = sorted(pair["T_tokens"].unique())
    cmap = plt.cm.viridis
    for i, T in enumerate(budgets):
        sub = pair[(pair["T_tokens"] == T) & (pair["G_b"] == 32)].sort_values("G_a")
        if sub.empty:
            continue
        x = sub["G_a"].values
        y = sub["retention"].values
        ylo = sub["retention_ci_low"].values
        yhi = sub["retention_ci_high"].values
        c = cmap(i / max(len(budgets) - 1, 1))
        ax.plot(x, y, "o-", color=c, label=f"T={T // 1_000_000}M", lw=2, ms=6)
        ax.fill_between(x, ylo, yhi, color=c, alpha=0.18)
    ax.axhline(WU_RETENTION, color="k", linestyle="--", lw=1.4,
               label=f"Wu 97.6%")
    ax.axhline(0.90, color="gray", linestyle=":", lw=1.0, label="90% threshold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel(r"$G_a$ (with fixed $G_b = 32$)")
    ax.set_ylabel("Retention $R = \\mathrm{acc}_{G_a} / \\mathrm{acc}_{G_b}$")
    ax.set_title(r"Retention of $G_a$ vs $G_b = 32$")
    ax.set_ylim(0.6, 1.30)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # ----- Mid panel: DPO-equivalence heatmap at T=64M -----
    ax = axes[1]
    sub64 = pair[pair["T_tokens"] == 64_000_000]
    gs = sorted(set(sub64["G_a"]).union(set(sub64["G_b"])))
    n = len(gs)
    dpo = np.full((n, n), np.nan)
    for _, r in sub64.iterrows():
        i = gs.index(int(r["G_a"]))
        j = gs.index(int(r["G_b"]))
        dpo[i, j] = float(r["dpo_equivalence_score"])
        # mirror
        if np.isnan(dpo[j, i]):
            dpo[j, i] = float(r["dpo_equivalence_score"])
    im = ax.imshow(dpo, cmap="RdYlGn", vmin=0.5, vmax=1.0, origin="lower")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(g) for g in gs])
    ax.set_yticklabels([str(g) for g in gs])
    ax.set_xlabel(r"$G_b$")
    ax.set_ylabel(r"$G_a$")
    ax.set_title(r"DPO-equivalence score at $T = 64$M")
    for i in range(n):
        for j in range(n):
            if not np.isnan(dpo[i, j]):
                txt = f"{dpo[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7,
                        color="white" if dpo[i, j] < 0.78 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"DPO-eq $= 1 - 2|\Delta|/(\mathrm{acc}_a + \mathrm{acc}_b)$")

    # ----- Right panel: Pareto cost-effectiveness frontier -----
    ax = axes[2]
    for i, T in enumerate(budgets):
        sub = pareto[pareto["T_tokens"] == T].sort_values("G")
        if sub.empty:
            continue
        x = sub["G"].values
        y = sub["acc_per_M_tokens"].values
        c = cmap(i / max(len(budgets) - 1, 1))
        ax.plot(x, y, "o-", color=c, label=f"T={T // 1_000_000}M", lw=2, ms=6)
        # annotate rank-1 cell
        best_row = sub[sub["rank_by_acc_per_M"] == 1]
        if not best_row.empty:
            ax.annotate(f"G={int(best_row['G'].iloc[0])} (best)",
                        xy=(int(best_row["G"].iloc[0]),
                            float(best_row["acc_per_M_tokens"].iloc[0])),
                        xytext=(5, 8), textcoords="offset points",
                        fontsize=7, color=c)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.set_xlabel(r"$G$ (group size)")
    ax.set_ylabel(r"Accuracy per million rollout tokens")
    ax.set_title("Cost-effectiveness Pareto frontier")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Pillar 3 (iter35) — G=4..G=64 cross-scale audit", fontsize=12, y=1.01)
    fig.tight_layout()
    out_pdf = FIG / "group_size_iter35.pdf"
    out_png = FIG / "group_size_iter35.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  -> wrote {out_pdf}")
    print(f"  -> wrote {out_png}")


if __name__ == "__main__":
    main()

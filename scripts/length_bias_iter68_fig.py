"""length_bias_iter68_fig.py — 3-panel matplotlib PDF for iter68 reversal finding."""
from __future__ import annotations

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG_DIR = os.path.join(ROOT, "figures")
PAPER_FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def load_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    summary = load_tsv(os.path.join(RES, "length_bias_iter68_summary.tsv"))
    auc = load_tsv(os.path.join(RES, "length_bias_iter68_auc.tsv"))
    rev_cond = load_tsv(os.path.join(RES, "length_bias_iter68_rev_cond.tsv"))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel 1: aggregate reversal rate per (experiment, algo)
    per_run = load_tsv(os.path.join(RES, "length_bias_iter68_per_run.tsv"))
    gsm = [r for r in per_run if r["experiment"] == "drgrpo_gsm8k_cot"]
    arith = [r for r in per_run if r["experiment"] == "drgrpo_vs_grpo"]
    for i, (data, label) in enumerate([(gsm, "GSM8K CoT (3 seeds)"), (arith, "Arith-easy (5 seeds)")]):
        ax = axes[0]
        algo_vals = {"grpo": [], "dr_grpo": []}
        for r in data:
            algo_vals[r["algo"]].append(float(r["reversal_rate"]))
        positions = [i * 4 + 1, i * 4 + 2]
        vals = [algo_vals["grpo"], algo_vals["dr_grpo"]]
        bp = ax.boxplot(vals, positions=positions, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="#ffe0b2" if i == 0 else "#c8e6c9"))
        ax.set_xticks([i * 4 + 1.5])
        ax.set_xticklabels([label], fontsize=9)
        # overlay individual points
        for j, vlist in enumerate(vals):
            xs = [positions[j] + (k - len(vlist) / 2) * 0.06 for k in range(len(vlist))]
            ax.scatter(xs, vlist, color="black", s=18, zorder=3)
    axes[0].set_ylabel("reversal rate (fraction of dL sign-flips)")
    axes[0].set_title("(a) Aggregate reversal rate")
    axes[0].set_ylim(0.3, 0.85)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: conditional reversal rate (pos_dR vs neg_dR) — MEAN per (experiment, cell)
    pos = [r for r in rev_cond if r["cell"] == "rev_on_pos_dR"]
    neg = [r for r in rev_cond if r["cell"] == "rev_on_neg_dR"]
    labels = ["GSM8K pos-dR", "GSM8K neg-dR", "Arith pos-dR", "Arith neg-dR"]
    grpo_vals, drgrpo_vals = [], []
    for src, exp in [(pos, "drgrpo_gsm8k_cot"), (neg, "drgrpo_gsm8k_cot"),
                     (pos, "drgrpo_vs_grpo"), (neg, "drgrpo_vs_grpo")]:
        rows = [r for r in src if r["experiment"] == exp]
        grpo_vals.append(sum(float(r["rev_grpo"]) for r in rows) / len(rows))
        drgrpo_vals.append(sum(float(r["rev_drgrpo"]) for r in rows) / len(rows))
    x = list(range(len(labels)))
    axes[1].bar([xi - 0.18 for xi in x], grpo_vals, width=0.34, color="#90caf9", label="GRPO")
    axes[1].bar([xi + 0.18 for xi in x], drgrpo_vals, width=0.34, color="#ef9a9a", label="Dr.GRPO")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8, rotation=15)
    axes[1].set_ylabel("reversal rate (conditional)")
    axes[1].set_title("(b) Reversal rate by reward sign")
    axes[1].set_ylim(0.4, 0.85)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: signed AUC by seed
    gsm_auc = [r for r in auc if r["experiment"] == "drgrpo_gsm8k_cot"]
    arith_auc = [r for r in auc if r["experiment"] == "drgrpo_vs_grpo"]
    axes[2].bar(["GSM8K CoT"] * len(gsm_auc), [float(r["auc_grpo_minus_drgrpo"]) for r in gsm_auc],
                color="#90caf9", label="GRPO - Dr.GRPO AUC")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("signed area (token-step units)")
    axes[2].set_title("(c) Signed AUC (positive = GRPO trace above)")
    axes[2].grid(True, alpha=0.3)
    # show arith too as inset overlay
    ax2 = axes[2].twinx()
    ax2.bar(["Arith-easy"] * len(arith_auc), [float(r["auc_grpo_minus_drgrpo"]) for r in arith_auc],
            color="#ef9a9a", width=0.4, alpha=0.6, label="Arith-easy")
    ax2.set_ylabel("Arith-easy AUC", color="#c62828")
    ax2.tick_params(axis="y", labelcolor="#c62828")

    fig.suptitle("Iter 68 — Pillar 4: Trajectory-divergence decomposition", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"length_bias_iter68_reversal.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        if ext == "pdf":
            fig.savefig(os.path.join(PAPER_FIG_DIR, f"length_bias_iter68_reversal.{ext}"),
                        bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
"""
Iter 64 — Pillar 4 figure: Conditional Length Response to Reward Direction.
3-panel: (a) per-step (ΔR, ΔL) scatter for GSM8K CoT GRPO vs Dr.GRPO with
sign-conditioned marginals; (b) compression-on-reward-up by ZVF tier;
(c) raw-drift paired bar with CI.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main():
    paired = read_tsv(RES / "length_bias_iter64_paired.tsv")
    summary = read_tsv(RES / "length_bias_iter64_summary.tsv")

    gsm = [r for r in paired if r["experiment"] == "drgrpo_gsm8k_cot"]
    gsm_sum = [r for r in summary if r["experiment"] == "drgrpo_gsm8k_cot"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ----- Panel (a): E[dL] by sign(dR) and ZVF tier
    ax = axes[0]
    zvf_tiers = ["high", "mid", "low"]
    sign_labels = ["dR>0", "dR<0"]
    grpo_vals = []
    drgrpo_vals = []
    grpo_ci_lo = []
    grpo_ci_hi = []
    drgrpo_ci_lo = []
    drgrpo_ci_hi = []
    for tier in zvf_tiers:
        for sign in ["pos", "neg"]:
            cell = f"{sign}_{tier}"
            row = next((r for r in gsm if r["cell"] == cell), None)
            if row:
                grpo_vals.append(float(row["mean_grpo"]))
                drgrpo_vals.append(float(row["mean_drgrpo"]))
    x = np.arange(len(grpo_vals))
    w = 0.38
    ax.bar(x - w / 2, grpo_vals, w, label="GRPO", color="#3a86ff")
    ax.bar(x + w / 2, drgrpo_vals, w, label="Dr.GRPO", color="#fb5607")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}/{s}" for t in zvf_tiers for s in sign_labels],
        rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("E[ΔL | sign(ΔR), ZVF tier] (tokens)")
    ax.set_title("(a) Length response by reward sign × ZVF\n(GSM8K CoT)")
    ax.legend(fontsize=8)

    # ----- Panel (b): compression on reward-up steps by ZVF tier
    ax = axes[1]
    g_pos, d_pos = [], []
    for tier in zvf_tiers:
        cell = f"pos_{tier}"
        row = next((r for r in gsm if r["cell"] == cell), None)
        if row:
            g_pos.append(-float(row["mean_grpo"]))   # compression = -ΔL
            d_pos.append(-float(row["mean_drgrpo"]))
    x = np.arange(len(zvf_tiers))
    ax.bar(x - w / 2, g_pos, w, label="GRPO", color="#3a86ff")
    ax.bar(x + w / 2, d_pos, w, label="Dr.GRPO", color="#fb5607")
    ax.set_xticks(x)
    ax.set_xticklabels(zvf_tiers)
    ax.set_xlabel("ZVF tier")
    ax.set_ylabel("Compression on reward-up steps\n(tokens/step, higher = more compression)")
    ax.set_title("(b) GRPO compresses MORE on reward-up\nsteps in low-ZVF regime (Dr.GRPO flatter)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5)

    # ----- Panel (c): raw drift paired summary
    ax = axes[2]
    metrics = ["alignment_all", "raw_drift_all", "compression_on_pos_dR"]
    short_labels = ["Alignment\n(E[dL|ΔR>0]−E[dL|ΔR<0])",
                    "Raw drift\n(E[ΔL])",
                    "Compression\non reward-up"]
    g_v, d_v, err_lo, err_hi = [], [], [], []
    for m in metrics:
        row = next((r for r in gsm_sum if r["metric"] == m), None)
        if row:
            g_v.append(float(row["mean_grpo"]))
            d_v.append(float(row["mean_drgrpo"]))
            err_lo.append(float(row["ci_lo"]))
            err_hi.append(float(row["ci_hi"]))
    x = np.arange(len(metrics))
    g_err_lo = [abs(v - lo) for v, lo in zip(g_v, err_lo)]
    g_err_hi = [abs(hi - v) for v, hi in zip(g_v, err_hi)]
    d_err_lo = [abs(v - lo) for v, lo in zip(d_v, err_lo)]
    d_err_hi = [abs(hi - v) for v, hi in zip(d_v, err_hi)]
    ax.bar(x - w / 2, g_v, w, yerr=[g_err_lo, g_err_hi],
           label="GRPO", color="#3a86ff", capsize=3)
    ax.bar(x + w / 2, d_v, w, yerr=[d_err_lo, d_err_hi],
           label="Dr.GRPO", color="#fb5607", capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_title("(c) Aggregate metrics with 95% CI\n(GSM8K CoT)")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Iter 64 — Conditional Length Response to Reward Direction (CLRRD):\n"
        "Dr.GRPO loses reward-direction coupling, esp. in heterogeneous-group (low-ZVF) regime",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_pdf = FIG / "length_bias_iter64_clrrd.pdf"
    out_png = FIG / "length_bias_iter64_clrrd.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Wrote {out_pdf} and {out_png}")

    # Mirror to paper/figures
    paper_fig = ROOT / "paper" / "figures"
    if paper_fig.exists():
        import shutil
        shutil.copyfile(out_pdf, paper_fig / "length_bias_iter64_clrrd.pdf")
        print(f"Mirrored to {paper_fig / 'length_bias_iter64_clrrd.pdf'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
length_zvf_figure.py — 3-panel figure for length×ZVF cross-pillar coupling.

Panel (A): Per-(task, algo) ρ(len, ZVF) bar plot with bootstrap 95% CI
Panel (B): Length-bin conditional ZVF (L1=low length, L3=high length)
Panel (C): Per-step trajectories showing len, zvf, reward co-evolution
           on one representative seed from each (task, algo) cell.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"


def main():
    drgrpo = json.load(open(RESULTS / "drgrpo_vs_grpo.json"))
    gsm = json.load(open(RESULTS / "drgrpo_gsm8k_cot_full.json"))

    # ----------------------------------------------------------------
    # Aggregate ρ(len, ZVF) per (task, algo) with bootstrap CIs
    # ----------------------------------------------------------------
    rng = np.random.default_rng(42)
    rho_per_cell = {}  # (task, algo) -> list of per-run ρ(len, ZVF)
    for src, task in ((drgrpo["runs"], "arithmetic_qwen2.5-0.5b"),
                       (gsm["runs"], "gsm8k_cot_qwen2.5-1.5b")):
        for r in src:
            from scipy.stats import spearmanr
            leng = np.array([e["mean_comp_len"] for e in r["step_log"]])
            zvf = np.array([e["zvf"] for e in r["step_log"]])
            rho, _ = spearmanr(leng, zvf)
            rho_per_cell.setdefault((task, r["algo"]), []).append(rho)

    cells = sorted(rho_per_cell.keys())
    means = []
    los = []
    his = []
    for cell in cells:
        vals = np.array(rho_per_cell[cell])
        means.append(vals.mean())
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(vals), size=len(vals))
            boots.append(vals[idx].mean())
        los.append(np.quantile(boots, 0.025))
        his.append(np.quantile(boots, 0.975))

    # ----------------------------------------------------------------
    # Length-bin conditional ZVF per (task, algo)
    # ----------------------------------------------------------------
    bin_zvf = {}  # (task, algo, bin) -> list
    bin_order = ["L1", "L2", "L3"]
    for src, task in ((drgrpo["runs"], "arithmetic_qwen2.5-0.5b"),
                       (gsm["runs"], "gsm8k_cot_qwen2.5-1.5b")):
        for r in src:
            leng = np.array([e["mean_comp_len"] for e in r["step_log"]])
            zvf = np.array([e["zvf"] for e in r["step_log"]])
            qs = np.quantile(leng, np.linspace(0, 1, 4))
            qs[0] -= 1e-9; qs[-1] += 1e-9
            for i, lab in enumerate(bin_order):
                mask = (leng >= qs[i]) & (leng < qs[i + 1])
                if mask.any():
                    bin_zvf.setdefault((task, r["algo"], lab), []).append(zvf[mask].mean())

    # ----------------------------------------------------------------
    # Representative trajectories — pick seed 42 (or first available)
    # ----------------------------------------------------------------
    rep_runs = {}
    for src, task in ((drgrpo["runs"], "arithmetic_qwen2.5-0.5b"),
                       (gsm["runs"], "gsm8k_cot_qwen2.5-1.5b")):
        for r in src:
            seed = int(r["seed"])
            # Prefer seed 42 if exists, else first
            if (task, r["algo"]) not in rep_runs or seed == 42:
                rep_runs[(task, r["algo"])] = r

    # ----------------------------------------------------------------
    # Build figure: 1 row, 3 columns
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: ρ(len, ZVF) bar plot
    ax = axes[0]
    labels = []
    for c in cells:
        if c[0] == "arithmetic_qwen2.5-0.5b":
            t = "Arith"
        else:
            t = "GSM8K"
        labels.append(f"{t}\n{c[1].upper()}")
    x = np.arange(len(cells))
    err_lo = np.array(means) - np.array(los)
    err_hi = np.array(his) - np.array(means)
    colors = ["#1f77b4" if "grpo" == c[1] else "#d62728" for c in cells]
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\rho(\mathrm{len}, \mathrm{ZVF})$", fontsize=10)
    ax.set_title(r"(A) Length-ZVF joint Spearman" + "\n(per-run, with 95% bootstrap CI)", fontsize=10)
    # Add n labels
    for i, c in enumerate(cells):
        ax.text(i, max(0, means[i]) + 0.05, f"n={len(rho_per_cell[c])}",
                ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Length-bin conditional ZVF
    ax = axes[1]
    bar_w = 0.18
    x0 = np.arange(4)  # 4 (task, algo) cells
    legend_handles = []
    legend_labels = []
    color_map = {"L1": "#2ca02c", "L2": "#ff7f0e", "L3": "#9467bd"}
    for k, lab in enumerate(bin_order):
        vals = [bin_zvf.get((c[0], c[1], lab), [np.nan])[0] for c in cells]
        sds = [np.std(bin_zvf.get((c[0], c[1], lab), [np.nan])) if len(bin_zvf.get((c[0], c[1], lab), [])) > 1 else 0
               for c in cells]
        bars = ax.bar(x0 + (k - 1) * bar_w, vals, bar_w, yerr=sds, capsize=2,
                      color=color_map[lab], edgecolor="black", linewidth=0.4, label=lab)
    ax.set_xticks(x0)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean ZVF (per length bin)", fontsize=10)
    ax.set_title("(B) Length-bin conditional ZVF\n(low/mid/high length tertiles)", fontsize=10)
    ax.legend(title="Length bin", fontsize=8, title_fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel C: representative trajectory (len & ZVF co-evolution)
    ax = axes[2]
    # Show arithmetic GRPO seed 42
    target = next(((t, a, r) for (t, a), r in rep_runs.items()
                   if t == "arithmetic_qwen2.5-0.5b" and a == "grpo"), None)
    if target is not None:
        t, a, r = target
        steps = [e["step"] for e in r["step_log"]]
        leng = [e["mean_comp_len"] for e in r["step_log"]]
        zvf = [e["zvf"] for e in r["step_log"]]
        # normalize length to [0,1]
        l_arr = np.array(leng); z_arr = np.array(zvf)
        l_norm = (l_arr - l_arr.min()) / (l_arr.max() - l_arr.min() + 1e-9)
        ax.plot(steps, l_norm, "-o", color="#1f77b4", label="mean_comp_len (norm)", markersize=3)
        ax.plot(steps, z_arr, "-s", color="#d62728", label="ZVF", markersize=3)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(f"(C) Trajectory co-evolution\n({t}, GRPO, seed 42)", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGS / f"length_zvf_coupling.{ext}"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
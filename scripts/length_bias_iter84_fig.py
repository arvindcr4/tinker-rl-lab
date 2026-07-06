#!/usr/bin/env python3
"""Iter 84 figure: 4-panel spectral + long-memory + Granger dashboard."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
FIGS = os.path.join(W, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

perrun = []
with open(os.path.join(RES, "length_bias_iter84_perrun.tsv")) as f:
    lines = f.read().strip().split("\n")
    header = lines[0].split("\t")
    for ln in lines[1:]:
        row = dict(zip(header, ln.split("\t")))
        for k in ["hurst_L", "hurst_R", "coh_vlow", "coh_low", "coh_mid", "coh_high",
                  "F_lr", "F_rl", "spearman_rho", "L_mean", "L_std", "R_mean", "R_std",
                  "last10_acc", "p_lr", "p_rl", "spearman_p", "r2_lr", "r2_rl"]:
            try:
                row[k] = float(row[k])
            except (KeyError, ValueError):
                row[k] = float("nan")
        perrun.append(row)

TASKS = ["arithmetic", "gsm8k_cot"]
ALGOS = ["grpo", "dr_grpo"]
ALGO_LABEL = {"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}
ALGO_CLR = {"grpo": "#3b6fb6", "dr_grpo": "#c14545"}
BANDS = [("vlow", "0.00-0.05"), ("low", "0.05-0.15"), ("mid", "0.15-0.30"), ("high", "0.30-0.50")]

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# Panel A: per-task coherence bands, GRPO vs Dr.GRPO side-by-side
ax = axes[0, 0]
band_keys = ["coh_vlow", "coh_low", "coh_mid", "coh_high"]
x = np.arange(len(BANDS))
w = 0.38
for ti, task in enumerate(TASKS):
    sub_ax = plt.subplot2grid((2, 4), (0, ti * 2), colspan=2, fig=fig) if False else ax
    if ti == 0:
        sub_ax = axes[0, 0]
    else:
        sub_ax = axes[0, 1]
    for ai, algo in enumerate(ALGOS):
        vals = []
        ses = []
        for bk in band_keys:
            vs = [r[bk] for r in perrun if r["task"] == task and r["algo"] == algo and np.isfinite(r[bk])]
            vals.append(np.mean(vs))
            ses.append(np.std(vs, ddof=1) / np.sqrt(max(1, len(vs))))
        pos = x + (ai - 0.5) * w
        sub_ax.bar(pos, vals, w, yerr=ses, color=ALGO_CLR[algo],
                   label=ALGO_LABEL[algo], alpha=0.85, capsize=2)
    sub_ax.set_xticks(x)
    sub_ax.set_xticklabels([b[1] for b in BANDS], fontsize=8)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_title(task, fontsize=10)
    sub_ax.set_ylabel("|C_xy(f)|^2  (length vs reward)" if ti == 0 else "")
    sub_ax.grid(axis="y", alpha=0.3)
    sub_ax.legend(fontsize=8, loc="upper right")

# Panel C: Hurst exponent (length)
ax = axes[1, 0]
metrics = [("hurst_L", "Hurst H(L_t)"), ("F_lr", "F-stat L->R"), ("F_rl", "F-stat R->L")]
for ai, algo in enumerate(ALGOS):
    xs = []
    ys = []
    es = []
    for ti, task in enumerate(TASKS):
        for mi, _ in enumerate(metrics):
            vs = [r[metrics[mi][0]] for r in perrun if r["task"] == task and r["algo"] == algo
                  and np.isfinite(r[metrics[mi][0]])]
            ys.append(np.mean(vs))
            es.append(np.std(vs, ddof=1) / np.sqrt(max(1, len(vs))))
            xs.append(ti * 3 + mi)
    pos = np.array(xs) + (ai - 0.5) * 0.30
    ax.bar(pos, ys, 0.30, yerr=es, color=ALGO_CLR[algo], label=ALGO_LABEL[algo], alpha=0.85, capsize=2)
ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(TASKS, fontsize=9)
ax.axhline(0.5, ls="--", color="grey", lw=0.7, label="H=0.5 (memoryless)")
ax.set_ylabel("value")
ax.set_title("Long-memory (H) and Granger F-stat", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7, loc="upper right", ncol=2)

# Panel D: Granger direction (delta F, Dr.GRPO - GRPO) per task
ax = axes[1, 1]
delta_F_lr = []
delta_F_rl = []
delta_coh_low = []
delta_hurst = []
for task in TASKS:
    for kind, store in [("F_lr", delta_F_lr), ("F_rl", delta_F_rl),
                        ("coh_low", delta_coh_low), ("hurst_L", delta_hurst)]:
        g = [r[kind] for r in perrun if r["task"] == task and r["algo"] == "grpo" and np.isfinite(r[kind])]
        d = [r[kind] for r in perrun if r["task"] == task and r["algo"] == "dr_grpo" and np.isfinite(r[kind])]
        n = min(len(g), len(d))
        store.append(np.mean(d[:n]) - np.mean(g[:n]))
x = np.arange(2)
w = 0.20
ax.bar(x - 1.5 * w, delta_F_lr, w, color="#234b6e", label="ΔF L->R", alpha=0.85)
ax.bar(x - 0.5 * w, delta_F_rl, w, color="#a83232", label="ΔF R->L", alpha=0.85)
ax.bar(x + 0.5 * w, delta_coh_low, w, color="#3aa856", label="Δcoh(0.05-0.15)", alpha=0.85)
ax.bar(x + 1.5 * w, delta_hurst, w, color="#a87232", label="ΔHurst(L)", alpha=0.85)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(TASKS, fontsize=9)
ax.set_ylabel("Dr.GRPO - GRPO")
ax.set_title("Direction of divergence (paired Δ)", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7, loc="upper right", ncol=2)

fig.suptitle("Iter 84 — Length×Reward: frequency-domain + long-memory decomposition", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_pdf = os.path.join(FIGS, "length_bias_iter84.pdf")
out_png = os.path.join(FIGS, "length_bias_iter84.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=130)
print("[iter84] wrote", out_pdf, "and", out_png)
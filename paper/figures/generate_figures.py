"""
Publication-quality figure generation for NeurIPS TinkerRL paper.
Nature/Science academic style — no titles, minimal chartjunk.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os

# ---------------------------------------------------------------------------
# Global rcParams — Nature/Science academic style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # embed fonts in PDF
    "ps.fonttype": 42,
    "axes.grid": False,
})

OUT = "/home/user/workspace/tinker-rl-lab/paper/figures"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open("/home/user/workspace/tinker-rl-lab/experiments/all_results_consolidated.json") as f:
    data = json.load(f)

tc = data["tinker_completed"]
mc = data["modal_completed"]
trl = data["old_modal_trl_grpo"]

# Colour palette (colorblind-friendly — Okabe-Ito inspired from tab10)
COLORS = {
    "deepseek":    "#1f77b4",   # blue
    "qwen3_grpo":  "#ff7f0e",   # orange
    "llama_grpo":  "#2ca02c",   # green
    "qwen3_ppo":   "#d62728",   # red
    "llama_ppo":   "#9467bd",   # purple
    "trl":         "#8c564b",   # brown
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save(fig, name):
    png = os.path.join(OUT, f"{name}.png")
    pdf = os.path.join(OUT, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  Saved {png}")
    print(f"  Saved {pdf}")
    plt.close(fig)


def smooth(y, w=3):
    """Simple moving average."""
    if len(y) < w:
        return np.array(y, dtype=float)
    kernel = np.ones(w) / w
    padded = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


# ===========================================================================
# Figure 1 — Learning curves (double-column, 7 in)
# ===========================================================================
print("Generating Figure 1: learning_curves")

deepseek_trace  = np.array(tc["frontier_gsm8k_deepseek-v3.1"]["reward_trace"], dtype=float)
qwen3g_trace    = np.array(tc["scale_gsm8k_qwen3-8b"]["reward_trace"], dtype=float)
# Llama GRPO: complete failure — all zeros, 30 steps
llama_grpo_trace = np.zeros(30, dtype=float)

qwen3p_trace    = np.array(mc["ppo_qwen3-8b"]["reward_trace"], dtype=float)
llamap_trace    = np.array(mc["ppo_llama-8b-inst"]["reward_trace"], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
fig.subplots_adjust(wspace=0.38)

# --- Panel A: Tinker GRPO ---
ax = axes[0]
steps_ds  = np.arange(1, len(deepseek_trace) + 1)
steps_q3g = np.arange(1, len(qwen3g_trace) + 1)
steps_lg  = np.arange(1, len(llama_grpo_trace) + 1)

ax.plot(steps_ds,  deepseek_trace,  color=COLORS["deepseek"],   lw=1.5, marker="o", ms=3,
        label="DeepSeek-V3.1 (~671B)")
ax.plot(steps_q3g, qwen3g_trace,    color=COLORS["qwen3_grpo"], lw=1.5, marker="s", ms=3,
        label="Qwen3-8B")
ax.plot(steps_lg,  llama_grpo_trace, color=COLORS["llama_grpo"], lw=1.5, marker="^", ms=3,
        ls="--", label="Llama-3.1-8B (tool_use)")

ax.set_xlabel("Training Step")
ax.set_ylabel("Mean Reward")
ax.set_ylim(-0.05, 1.10)
ax.set_xlim(0, 32)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.legend(loc="lower right", handlelength=1.5)
ax.text(-0.18, 1.0, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left")
ax.set_title("Tinker GRPO", fontsize=10, pad=4, loc="left", color="#444444")

# --- Panel B: Modal PPO ---
ax = axes[1]
steps_q3p = np.arange(1, len(qwen3p_trace) + 1)
steps_lp  = np.arange(1, len(llamap_trace) + 1)

ax.plot(steps_q3p, qwen3p_trace, color=COLORS["qwen3_ppo"], lw=1.5, marker="s", ms=3,
        label="Qwen3-8B")
ax.plot(steps_lp,  llamap_trace, color=COLORS["llama_ppo"], lw=1.5, marker="D", ms=3,
        label="Llama-3.1-8B")

ax.set_xlabel("Training Step")
ax.set_ylabel("Mean Reward")
ax.set_ylim(-0.05, 1.10)
ax.set_xlim(0, 32)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.legend(loc="lower right", handlelength=1.5)
ax.text(-0.18, 1.0, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left")
ax.set_title("Modal PPO-REINFORCE", fontsize=10, pad=4, loc="left", color="#444444")

plt.tight_layout()
save(fig, "learning_curves")


# ===========================================================================
# Figure 2 — Comparison bar chart (single-column, 3.5 in)
# ===========================================================================
print("Generating Figure 2: comparison_bars")

# Data for GSM8K comparisons
models = ["Qwen3-8B", "Llama-3.1-8B"]
grpo_peak   = [0.625,  0.0]
grpo_last10 = [0.344,  0.0]
ppo_peak    = [1.0,    1.0]
ppo_last10  = [0.35,   0.95]

x     = np.arange(len(models))
width = 0.18
gap   = 0.04

fig, ax = plt.subplots(figsize=(5.0, 3.6))

b1 = ax.bar(x - 1.5*width - gap, grpo_peak,   width, color=COLORS["qwen3_grpo"], label="GRPO Peak",         zorder=3)
b2 = ax.bar(x - 0.5*width,       grpo_last10, width, color=COLORS["qwen3_grpo"], label="GRPO Last-10 Avg",  alpha=0.55, hatch="///", zorder=3)
b3 = ax.bar(x + 0.5*width + gap, ppo_peak,    width, color=COLORS["qwen3_ppo"],  label="PPO Peak",          zorder=3)
b4 = ax.bar(x + 1.5*width + 2*gap, ppo_last10, width, color=COLORS["qwen3_ppo"], label="PPO Last-10 Avg",   alpha=0.55, hatch="///", zorder=3)

# Value labels
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

autolabel(b1); autolabel(b2); autolabel(b3); autolabel(b4)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Reward")
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.legend(loc="upper right", fontsize=8, ncol=2,
          handlelength=1.2, handletextpad=0.5, columnspacing=0.8)
ax.spines["left"].set_visible(True)

# Footnote
ax.text(0.01, -0.15, "Task: GSM8K math reasoning. GRPO = Tinker platform; PPO = Modal platform.",
        transform=ax.transAxes, fontsize=7.5, color="#666666", va="top")

plt.tight_layout()
save(fig, "comparison_bars")


# ===========================================================================
# Figure 3 — Scaling plot (single-column, 3.5 in)
# ===========================================================================
print("Generating Figure 3: scaling_plot")

# Points: (params_B, last10_avg, label, method, marker)
points = [
    (0.5,   0.734, "Qwen2.5-0.5B\n(TRL GRPO)", "GRPO",  "o", COLORS["trl"]),
    (8.0,   0.344, "Qwen3-8B\n(Tinker GRPO)", "GRPO",   "s", COLORS["qwen3_grpo"]),
    (671.0, 0.850, "DeepSeek-V3.1\n(Tinker GRPO)", "GRPO", "D", COLORS["deepseek"]),
    (8.0,   0.350, "Qwen3-8B\n(Modal PPO)",    "PPO",    "^", COLORS["qwen3_ppo"]),
    (8.0,   0.950, "Llama-3.1-8B\n(Modal PPO)", "PPO",  "P", COLORS["llama_ppo"]),
]

fig, ax = plt.subplots(figsize=(4.5, 3.8))

for (params, last10, label, method, mk, col) in points:
    ax.scatter(params, last10, s=70, marker=mk, color=col, zorder=5,
               edgecolors="white", linewidths=0.5)

# Annotations with offsets to avoid overlap
offsets = {
    "Qwen2.5-0.5B\n(TRL GRPO)":     (-0.02,  0.04),
    "Qwen3-8B\n(Tinker GRPO)":      ( 0.0,  -0.08),
    "DeepSeek-V3.1\n(Tinker GRPO)": (-0.3,   0.04),
    "Qwen3-8B\n(Modal PPO)":        ( 0.0,   0.04),
    "Llama-3.1-8B\n(Modal PPO)":    ( 0.0,   0.04),
}

for (params, last10, label, method, mk, col) in points:
    dx, dy = offsets.get(label, (0, 0.04))
    ax.annotate(label,
                xy=(params, last10),
                xytext=(params * (1 + 0.05), last10 + dy),
                fontsize=7.5,
                color="#333333",
                ha="left" if params < 100 else "right",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))

# Legend by method
grpo_patch = mpatches.Patch(color="#888888", label="GRPO (Tinker / TRL)")
ppo_patch  = mpatches.Patch(color=COLORS["qwen3_ppo"], alpha=0.7, label="PPO (Modal)")
ax.legend(handles=[grpo_patch, ppo_patch], loc="lower right", fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("Model Parameters (B)")
ax.set_ylabel("Last-10 Reward Average")
ax.set_xlim(0.2, 3000)
ax.set_ylim(0.0, 1.05)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{x:g}"))

# Disclaimer note
ax.text(0.02, 0.02,
        "Note: different tasks / platforms — trends are indicative only.",
        transform=ax.transAxes, fontsize=7, color="#888888", va="bottom")

plt.tight_layout()
save(fig, "scaling_plot")


# ===========================================================================
# Figure 4 — PPO vs GRPO detail, Qwen3-8B (double-column, 7 in)
# ===========================================================================
print("Generating Figure 4: ppo_vs_grpo_detail")

grpo = qwen3g_trace
ppo  = qwen3p_trace
steps = np.arange(1, 31)

fig, ax = plt.subplots(figsize=(5.5, 3.2))

# Raw traces (light / thin)
ax.plot(steps, grpo, color=COLORS["qwen3_grpo"], lw=0.8, alpha=0.35)
ax.plot(steps, ppo,  color=COLORS["qwen3_ppo"],  lw=0.8, alpha=0.35)

# Smoothed (bold)
ax.plot(steps, smooth(grpo, 5), color=COLORS["qwen3_grpo"], lw=2.0,
        label="Tinker GRPO (raw + smoothed)")
ax.plot(steps, smooth(ppo, 5),  color=COLORS["qwen3_ppo"],  lw=2.0,
        label="Modal PPO-REINFORCE (raw + smoothed)")

# Shaded last-10 window
last10_start = 21
ax.axvspan(last10_start, 30, alpha=0.08, color="grey", zorder=0)
ax.text(last10_start + 0.3, 1.04, "last-10 window",
        fontsize=7.5, color="#666666", va="bottom")

# Annotate final last-10 averages
grpo_l10 = np.mean(grpo[-10:])
ppo_l10  = np.mean(ppo[-10:])
ax.axhline(grpo_l10, xmin=(last10_start-1)/30, xmax=1,
           color=COLORS["qwen3_grpo"], lw=0.9, ls=":", alpha=0.7)
ax.axhline(ppo_l10,  xmin=(last10_start-1)/30, xmax=1,
           color=COLORS["qwen3_ppo"],  lw=0.9, ls=":", alpha=0.7)

ax.text(30.4, grpo_l10, f"{grpo_l10:.3f}", color=COLORS["qwen3_grpo"],
        va="center", fontsize=8)
ax.text(30.4, ppo_l10,  f"{ppo_l10:.3f}", color=COLORS["qwen3_ppo"],
        va="center", fontsize=8)

ax.set_xlabel("Training Step")
ax.set_ylabel("Mean Reward")
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(0, 33)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.legend(loc="upper left", handlelength=1.5)

plt.tight_layout()
save(fig, "ppo_vs_grpo_detail")


# ===========================================================================
# Figure 5 — TRL GRPO seeds box/violin (single-column, 3.5 in)
# ===========================================================================
print("Generating Figure 5: old_trl_seeds")

seeds_acc = np.array(trl["accuracies"])
mean_acc  = trl["mean_accuracy"]
std_acc   = trl["std"]
seed_ids  = trl["seeds"]

fig, ax = plt.subplots(figsize=(3.5, 3.6))

# Violin
parts = ax.violinplot([seeds_acc], positions=[0], widths=0.5,
                      showmeans=False, showmedians=False, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor(COLORS["trl"])
    pc.set_alpha(0.35)
    pc.set_edgecolor(COLORS["trl"])

# Box (IQR)
q25, q75 = np.percentile(seeds_acc, [25, 75])
ax.plot([0, 0], [q25, q75], lw=4, color=COLORS["trl"], solid_capstyle="round", zorder=4)
ax.plot(0, np.median(seeds_acc), "o", ms=8, color=COLORS["trl"], zorder=5, label="Median")

# Mean ± std
ax.errorbar(0, mean_acc, yerr=std_acc, fmt="D", ms=7, color="#333333",
            elinewidth=1.5, capsize=5, zorder=6, label=f"Mean ± SD ({mean_acc:.3f}±{std_acc:.3f})")

# Individual points (jittered)
np.random.seed(0)
jitter = np.random.uniform(-0.05, 0.05, len(seeds_acc))
for i, (acc, sid, jit) in enumerate(zip(seeds_acc, seed_ids, jitter)):
    ax.scatter(jit, acc, s=40, color=COLORS["trl"], zorder=7, alpha=0.85,
               edgecolors="white", linewidths=0.5)
    ax.annotate(f"s={sid}", (jit, acc), xytext=(0.12, acc),
                fontsize=7.5, color="#555555", va="center")

ax.set_xlim(-0.5, 0.8)
ax.set_ylim(0.55, 0.88)
ax.set_xticks([])
ax.set_ylabel("Accuracy (GSM8K)")
ax.legend(loc="lower right", fontsize=8, handlelength=1.2)

# Add thin horizontal reference lines at 0.7 and 0.8
for ref in [0.7, 0.8]:
    ax.axhline(ref, color="#cccccc", lw=0.8, ls="--", zorder=0)
    ax.text(-0.48, ref, f"{ref:.1f}", fontsize=7.5, color="#aaaaaa", va="center")

ax.text(0.03, 0.02,
        f"Qwen2.5-0.5B · TRL GRPO · {trl['steps']} steps · {trl['gpu']} GPU\nn=5 seeds",
        transform=ax.transAxes, fontsize=7.5, color="#666666", va="bottom")

plt.tight_layout()
save(fig, "old_trl_seeds")

print("\nAll figures generated successfully.")

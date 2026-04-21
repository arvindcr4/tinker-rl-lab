from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(label, width=width)) for label in labels]


def annotate_bar(ax, bar, text, dy=0.025, fontsize=10, weight="bold"):
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fontsize, weight=weight)


def style_axes(ax):
    ax.grid(axis="y", color="#d7dce2", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#88929d")
    ax.spines["bottom"].set_color("#88929d")
    ax.tick_params(colors="#26323f")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
        }
    )

    colors = {
        "blue": "#2f6fbb",
        "orange": "#d9822b",
        "green": "#2f8f5b",
        "red": "#b84a62",
        "gray": "#6d7885",
        "light_blue": "#dbeafe",
        "light_red": "#fde2e8",
        "light_green": "#dff3e8",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=True)
    fig.suptitle(
        "Chapter 6 empirical dashboard: GRPO helps only when the reward signal is usable",
        x=0.5,
        y=1.02,
    )

    # Panel A: tool-use format acquisition.
    ax = axes[0, 0]
    labels = ["Strict Tinker\n(no warm-up)", "SFT only", "SFT + GRPO"]
    values = [0.00, 0.72, 0.91]
    bars = ax.bar(labels, values, color=[colors["red"], colors["gray"], colors["green"]], width=0.62)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tool-call reward / validity")
    ax.set_title("A. Tool use: GRPO refines a learned schema")
    style_axes(ax)
    for bar, text in zip(bars, ["0.00", "0.72", "0.91"]):
        annotate_bar(ax, bar, text)
    ax.annotate(
        "Warm-up creates\nvalid samples",
        xy=(2, 0.91),
        xytext=(1.25, 1.0),
        arrowprops=dict(arrowstyle="->", color=colors["green"], lw=1.5),
        ha="center",
        fontsize=10,
        color="#1f5d3b",
    )

    # Panel B: held-out math generalization.
    ax = axes[0, 1]
    heldout_labels = ["Base Qwen3-8B", "GRPO checkpoints"]
    heldout_values = np.array([82.0, 83.3])
    interval_low = np.array([76.5, 80.6])
    interval_high = np.array([87.5, 86.0])
    yerr = np.vstack([heldout_values - interval_low, interval_high - heldout_values])
    bars = ax.bar(heldout_labels, heldout_values, color=[colors["gray"], colors["blue"]], width=0.58)
    ax.errorbar(
        np.arange(len(heldout_values)),
        heldout_values,
        yerr=yerr,
        fmt="none",
        ecolor="#26323f",
        elinewidth=1.4,
        capsize=5,
    )
    ax.set_ylim(74, 89)
    ax.set_ylabel("Held-out GSM8K accuracy (%)")
    ax.set_title("B. Math: training reward is not proof of generalization")
    style_axes(ax)
    for bar, text in zip(bars, ["82.0%", "83.3%"]):
        annotate_bar(ax, bar, text, dy=0.3)
    ax.text(
        0.5,
        75.2,
        "+1.3 pp, p = 0.26 (not significant)",
        ha="center",
        va="bottom",
        fontsize=10,
        color=colors["red"],
        weight="bold",
    )

    # Panel C: ZVF/GU diagnostic regimes.
    ax = axes[1, 0]
    ax.axvspan(0.72, 1.02, ymin=0.0, ymax=0.42, color=colors["light_red"], alpha=0.9)
    ax.axvspan(0.72, 1.02, ymin=0.72, ymax=1.0, color=colors["light_blue"], alpha=0.75)
    ax.axvspan(0.0, 0.72, ymin=0.55, ymax=1.0, color=colors["light_green"], alpha=0.65)
    zvf_points = [
        ("GSM8K\nLR=1e-4", 0.52, 1.000, colors["green"]),
        ("MATH-500", 0.81, 0.574, colors["orange"]),
        ("HumanEval", 0.98, 0.024, colors["red"]),
        ("Tool JSON", 0.55, 1.000, colors["blue"]),
    ]
    for label, zvf, reward, color in zvf_points:
        ax.scatter(zvf, reward, s=120, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(zvf + 0.025, reward, label, ha="left", va="center", fontsize=9)
    ax.text(0.83, 0.13, "Failure:\nflat wrong groups", ha="center", va="center", fontsize=9, color="#7a2538")
    ax.text(0.83, 0.92, "Saturation:\nflat correct groups", ha="center", va="center", fontsize=9, color="#254f87")
    ax.text(0.24, 0.86, "Usable:\nreward diversity", ha="center", va="center", fontsize=9, color="#1f5d3b")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean ZVF (1 - GU)")
    ax.set_ylabel("Last-10 / final reward")
    ax.set_title("C. ZVF/GU: diagnose whether rollouts can teach")
    style_axes(ax)

    # Panel D: PPO versus GRPO reversal.
    ax = axes[1, 1]
    models = ["Qwen3-8B", "Llama-3.1-8B\nInstruct"]
    grpo = np.array([0.344, 0.844])
    ppo = np.array([0.225, 0.975])
    x = np.arange(len(models))
    width = 0.34
    bars_grpo = ax.bar(x - width / 2, grpo, width, label="GRPO", color=colors["blue"])
    bars_ppo = ax.bar(x + width / 2, ppo, width, label="PPO", color=colors["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(models, width=14))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Last-10 reward")
    ax.set_title("D. Framework/model effects reverse the winner")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    for bars in [bars_grpo, bars_ppo]:
        for bar in bars:
            annotate_bar(ax, bar, f"{bar.get_height():.3f}", dy=0.025, fontsize=9)
    ax.text(x[0], 0.47, "GRPO wins", ha="center", fontsize=10, weight="bold", color=colors["blue"])
    ax.text(x[1], 1.02, "PPO wins", ha="center", fontsize=10, weight="bold", color=colors["orange"])

    for label, ax in zip(["A", "B", "C", "D"], axes.ravel()):
        ax.text(
            -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=14,
            weight="bold",
            va="top",
            ha="left",
            color="#111827",
        )

    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"results_dashboard.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(out)


if __name__ == "__main__":
    main()

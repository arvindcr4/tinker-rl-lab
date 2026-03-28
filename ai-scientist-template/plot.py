"""Visualization for GRPO GSM8K experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Map run folders to human-readable labels
labels = {
    "run_0": "Baseline",
}

# Auto-discover additional runs
for d in sorted(Path(".").glob("run_*")):
    if d.name not in labels and d.is_dir():
        labels[d.name] = d.name.replace("_", " ").title()


def load_run(run_dir: str) -> dict | None:
    """Load final_info.json from a run directory."""
    path = Path(run_dir) / "final_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_reward_traces(run_dir: str) -> list[np.ndarray]:
    """Load per-seed reward traces from a run directory."""
    traces = []
    for seed_dir in sorted(Path(run_dir).glob("seed_*")):
        trace_path = seed_dir / "reward_trace.npy"
        if trace_path.exists():
            traces.append(np.load(trace_path))
    return traces


def plot_accuracy_comparison():
    """Bar chart comparing last-10 accuracy across runs."""
    runs = []
    for run_name, label in labels.items():
        info = load_run(run_name)
        if info is None:
            continue
        gsm = info["gsm8k_training"]
        runs.append({
            "label": label,
            "mean": gsm["means"]["last_10_accuracy_mean"],
            "stderr": gsm["stderrs"]["last_10_accuracy_stderr"],
        })

    if not runs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(runs))
    means = [r["mean"] for r in runs]
    errs = [r["stderr"] for r in runs]
    bars = ax.bar(x, means, yerr=errs, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in runs], rotation=45, ha="right")
    ax.set_ylabel("Last-10 Training Accuracy")
    ax.set_title("GRPO GSM8K: Training Accuracy Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, err in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.02,
                f"{mean:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved accuracy_comparison.png")


def plot_training_curves():
    """Plot reward traces across training steps for all runs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for (run_name, label), color in zip(labels.items(), colors):
        traces = load_reward_traces(run_name)
        if not traces:
            continue

        # Pad to same length and compute mean/std
        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        steps = np.arange(max_len)

        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward (Accuracy)")
    ax.set_title("GRPO GSM8K: Training Reward Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")


def plot_metric_summary():
    """Summary table-style plot of all metrics."""
    runs = []
    for run_name, label in labels.items():
        info = load_run(run_name)
        if info is None:
            continue
        gsm = info["gsm8k_training"]
        runs.append({
            "label": label,
            "last_10": gsm["means"]["last_10_accuracy_mean"],
            "peak": gsm["means"]["peak_accuracy_mean"],
            "first_5": gsm["means"]["first_5_accuracy_mean"],
            "loss": gsm["means"]["training_loss_mean"],
        })

    if not runs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("last_10", "Last-10 Accuracy", axes[0]),
        ("peak", "Peak Accuracy", axes[1]),
        ("first_5", "First-5 Accuracy", axes[2]),
    ]

    for key, title, ax in metrics:
        vals = [r[key] for r in runs]
        lbls = [r["label"] for r in runs]
        ax.barh(range(len(vals)), vals, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(lbls)
        ax.set_title(title)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("metric_summary.png", dpi=150)
    plt.close()
    print("Saved metric_summary.png")


if __name__ == "__main__":
    plot_accuracy_comparison()
    plot_training_curves()
    plot_metric_summary()

"""
Figure Generation for TinkerRL Paper
=====================================
Generates publication-quality figures:
1. Learning curves with confidence bands
2. Performance profiles (Agarwal et al., 2021)
3. Scaling plots (performance vs compute)
4. Hyperparameter sensitivity heatmaps
5. Comparison bar chart

References:
    Agarwal et al., "Deep Reinforcement Learning at the Edge of the
    Statistical Precipice" (NeurIPS 2021)

Usage:
    python scripts/generate_figures.py --results-dir experiments/results/ --output-dir paper/figures/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib configuration (must happen before any pyplot import)
import matplotlib
matplotlib.use("Agg")        # non-interactive backend, safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, NullFormatter

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Publication style defaults
# ---------------------------------------------------------------------------

STYLE = {
    "font_size":       11,
    "title_size":      13,
    "label_size":      11,
    "tick_size":       9,
    "legend_size":     9,
    "linewidth":       1.8,
    "marker_size":     4,
    "dpi":             300,
    "fig_width":       6.0,
    "fig_height":      4.0,
    "grid_alpha":      0.25,
    "band_alpha":      0.20,
    "capsize":         4,
}

# Consistent colour palette across all figures (colourblind-friendly)
LIBRARY_COLORS = {
    "stable-baselines3": "#E87C56",   # warm orange
    "cleanrl":           "#5B9BD5",   # blue
    "rllib":             "#70B85E",   # green
    "tianshou":          "#9B72AF",   # purple
    "dopamine":          "#F0C14B",   # amber
    "default":           "#888888",   # grey fallback
}

PLACEHOLDER_ALPHA = 0.55      # visual dimming for placeholder figures

# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          STYLE["font_size"],
        "axes.titlesize":     STYLE["title_size"],
        "axes.labelsize":     STYLE["label_size"],
        "xtick.labelsize":    STYLE["tick_size"],
        "ytick.labelsize":    STYLE["tick_size"],
        "legend.fontsize":    STYLE["legend_size"],
        "lines.linewidth":    STYLE["linewidth"],
        "lines.markersize":   STYLE["marker_size"],
        "figure.dpi":         STYLE["dpi"],
        "savefig.dpi":        STYLE["dpi"],
        "savefig.bbox":       "tight",
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.alpha":         STYLE["grid_alpha"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def _save_figure(fig: plt.Figure, output_dir: str, stem: str) -> None:
    """Save figure as both PDF and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, format=ext)
        print(f"  Saved: {path}")
    plt.close(fig)


def _annotate_placeholder(ax: plt.Axes, message: str = "Data not found – placeholder") -> None:
    """Add a centred annotation to signal a placeholder figure."""
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, color="grey",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.7),
    )


def _library_color(name: str) -> str:
    return LIBRARY_COLORS.get(name, LIBRARY_COLORS["default"])


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_learning_curves(results_dir: str) -> Dict[str, dict]:
    """
    Attempt to load per-library learning curve data from JSON files.
    Expected format: {"steps": [...], "seeds": {"0": [...], "1": [...], ...}}
    Returns a dict keyed by library name.
    """
    data = {}
    for lib in LIBRARY_COLORS:
        if lib == "default":
            continue
        path = os.path.join(results_dir, lib, "learning_curves.json")
        loaded = _load_json(path)
        if loaded:
            data[lib] = loaded
    return data


def _load_sensitivity_csv(results_dir: str) -> Optional[dict]:
    """
    Load sensitivity results CSV from results_dir.
    Expected columns: hyperparameter, value, accuracy, seed
    Returns dict keyed by hyperparameter name → (values, accuracies)
    """
    import csv
    # Try a few common locations
    candidates = [
        os.path.join(results_dir, "sensitivity_seed42.csv"),
        os.path.join(results_dir, "sensitivity", "sensitivity_seed42.csv"),
        os.path.join(results_dir, "hyperparam_sensitivity.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            result = {}
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    param = row["hyperparameter"]
                    result.setdefault(param, {"values": [], "accuracies": []})
                    result[param]["values"].append(float(row["value"]))
                    result[param]["accuracies"].append(float(row["accuracy"]))
            return result
    return None


def _load_scaling_data(results_dir: str) -> Optional[dict]:
    """Load scaling data: {library: {flops: [...], accuracy: [...]}}"""
    path = os.path.join(results_dir, "scaling.json")
    return _load_json(path)


def _load_final_accuracies(results_dir: str) -> Optional[Dict[str, List[float]]]:
    """Load final accuracy per library across seeds: {library: [acc_seed0, ...]}"""
    path = os.path.join(results_dir, "final_accuracies.json")
    return _load_json(path)


# ---------------------------------------------------------------------------
# Synthetic fallback generators (for placeholder figures)
# ---------------------------------------------------------------------------

def _synth_learning_curves() -> Dict[str, dict]:
    """Generate plausible synthetic learning curve data for all libraries."""
    rng = np.random.default_rng(42)
    data = {}
    steps = np.linspace(0, 100_000, 50)

    for i, lib in enumerate(k for k in LIBRARY_COLORS if k != "default"):
        ceiling = 0.70 + i * 0.04
        seeds = {}
        for s in range(5):
            noise = rng.normal(0, 0.03, len(steps))
            curve = ceiling * (1 - np.exp(-steps / 30_000)) + noise
            curve = np.clip(curve, 0, 1)
            seeds[str(s)] = curve.tolist()
        data[lib] = {"steps": steps.tolist(), "seeds": seeds}
    return data


def _synth_sensitivity() -> Dict[str, dict]:
    """Synthetic sensitivity data matching SWEEPS in hyperparam_sensitivity.py."""
    rng   = np.random.default_rng(7)
    sweeps = {
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "clip_range":    [0.05, 0.1,  0.2,  0.3,  0.5],
        "entropy_coef":  [0.0,  0.001, 0.01, 0.05, 0.1],
        "gamma":         [0.9,  0.95,  0.99, 0.995, 1.0],
        "gae_lambda":    [0.8,  0.9,   0.95, 0.98,  1.0],
    }
    result = {}
    for param, vals in sweeps.items():
        base = rng.uniform(0.45, 0.75)
        accs = []
        for _ in vals:
            accs.append(float(np.clip(rng.normal(base, 0.08), 0, 1)))
        result[param] = {"values": vals, "accuracies": accs}
    return result


def _synth_scaling() -> Dict[str, dict]:
    rng = np.random.default_rng(13)
    data = {}
    for i, lib in enumerate(k for k in LIBRARY_COLORS if k != "default"):
        n = 8
        flops = sorted(rng.uniform(1e7, 1e10, n).tolist())
        accs  = [min(1.0, 0.30 + 0.40 * np.log10(f / 1e7) / 3 + rng.normal(0, 0.03))
                 for f in flops]
        data[lib] = {"flops": flops, "accuracy": accs}
    return data


def _synth_final_accuracies() -> Dict[str, List[float]]:
    rng = np.random.default_rng(99)
    return {
        lib: rng.uniform(0.55, 0.90, 5).tolist()
        for lib in LIBRARY_COLORS if lib != "default"
    }


# ---------------------------------------------------------------------------
# Figure 1: Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    data: Dict[str, dict],
    output_dir: str,
    is_placeholder: bool = False,
) -> None:
    """Mean ± std learning curves per library with confidence bands."""
    fig, ax = plt.subplots(figsize=(STYLE["fig_width"], STYLE["fig_height"]))

    for lib, d in data.items():
        steps  = np.array(d["steps"])
        curves = np.array(list(d["seeds"].values()))   # (n_seeds, n_steps)
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        color  = _library_color(lib)

        ax.plot(steps, mean, label=lib, color=color, linewidth=STYLE["linewidth"])
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=STYLE["band_alpha"])

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curves (ArithmeticEnv)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
    )
    ax.legend(loc="lower right", framealpha=0.85)

    if is_placeholder:
        _annotate_placeholder(ax, "Synthetic data – run experiments to populate")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig1_learning_curves")


# ---------------------------------------------------------------------------
# Figure 2: Performance profiles
# ---------------------------------------------------------------------------

def plot_performance_profiles(
    final_accs: Dict[str, List[float]],
    output_dir: str,
    is_placeholder: bool = False,
) -> None:
    """
    CDF of normalised scores across algorithms (Agarwal et al., NeurIPS 2021).
    x-axis: normalised score threshold τ
    y-axis: fraction of runs with score ≥ τ
    """
    # Normalise scores to [0, 1] using min-max across all runs
    all_scores = np.concatenate(list(final_accs.values()))
    s_min, s_max = all_scores.min(), all_scores.max()
    if s_max == s_min:
        s_max = s_min + 1e-6

    taus = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(STYLE["fig_width"], STYLE["fig_height"]))

    for lib, accs in final_accs.items():
        norm = (np.array(accs) - s_min) / (s_max - s_min)
        profile = [(norm >= tau).mean() for tau in taus]
        ax.plot(taus, profile, label=lib, color=_library_color(lib))

    ax.set_xlabel(r"Normalised Score Threshold $\tau$")
    ax.set_ylabel(r"Fraction of Runs with Score $\geq \tau$")
    ax.set_title("Performance Profiles")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.85)

    if is_placeholder:
        _annotate_placeholder(ax, "Synthetic data – run experiments to populate")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig2_performance_profiles")


# ---------------------------------------------------------------------------
# Figure 3: Scaling plot
# ---------------------------------------------------------------------------

def plot_scaling(
    scaling_data: Dict[str, dict],
    output_dir: str,
    is_placeholder: bool = False,
) -> None:
    """Final accuracy vs total training compute (log-scale x-axis)."""
    fig, ax = plt.subplots(figsize=(STYLE["fig_width"], STYLE["fig_height"]))

    for lib, d in scaling_data.items():
        flops = d["flops"]
        accs  = d["accuracy"]
        color = _library_color(lib)
        ax.semilogx(flops, accs, "o-", label=lib, color=color,
                    linewidth=STYLE["linewidth"], markersize=STYLE["marker_size"])

    ax.set_xlabel("Total Training Compute (FLOPs)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Scaling: Accuracy vs Compute")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="lower right", framealpha=0.85)

    if is_placeholder:
        _annotate_placeholder(ax, "Synthetic data – run experiments to populate")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig3_scaling")


# ---------------------------------------------------------------------------
# Figure 4: Hyperparameter sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(
    sensitivity_data: Dict[str, dict],
    output_dir: str,
    is_placeholder: bool = False,
) -> None:
    """
    Grid heatmap: rows = hyperparameters, columns = sweep values,
    cell colour = accuracy.
    """
    params = list(sensitivity_data.keys())
    n_params = len(params)
    # All sweeps must have the same number of values; if not, pad with NaN
    max_vals = max(len(v["values"]) for v in sensitivity_data.values())

    grid = np.full((n_params, max_vals), np.nan)
    col_labels_per_row = []

    for i, param in enumerate(params):
        vals = sensitivity_data[param]["values"]
        accs = sensitivity_data[param]["accuracies"]
        for j, acc in enumerate(accs):
            grid[i, j] = acc
        col_labels_per_row.append([f"{v:.0e}" if v < 0.01 else f"{v}" for v in vals])

    fig, ax = plt.subplots(figsize=(max(7, max_vals * 1.2), max(3, n_params * 0.8)))

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Accuracy", fraction=0.046, pad=0.04)

    ax.set_yticks(range(n_params))
    ax.set_yticklabels(params, fontsize=STYLE["tick_size"])
    ax.set_xticks(range(max_vals))
    # Use row 0's labels as generic x-labels (sweep indices)
    ax.set_xticklabels([f"v{j+1}" for j in range(max_vals)], fontsize=STYLE["tick_size"])
    ax.set_title("Hyperparameter Sensitivity (Accuracy)")
    ax.set_xlabel("Sweep Value Index")

    # Annotate cells with accuracy values
    for i in range(n_params):
        for j in range(max_vals):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    if is_placeholder:
        _annotate_placeholder(ax, "Synthetic data – run experiments to populate")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig4_sensitivity_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    final_accs: Dict[str, List[float]],
    output_dir: str,
    is_placeholder: bool = False,
) -> None:
    """Side-by-side bars with error bars (mean ± std) for each library."""
    libs  = list(final_accs.keys())
    means = [np.mean(final_accs[lib]) for lib in libs]
    stds  = [np.std(final_accs[lib])  for lib in libs]
    colors = [_library_color(lib) for lib in libs]

    x = np.arange(len(libs))
    fig, ax = plt.subplots(figsize=(max(5, len(libs) * 1.4), STYLE["fig_height"]))

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=STYLE["capsize"],
                  width=0.6, edgecolor="white", linewidth=0.6)

    # Value labels on top of bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=STYLE["tick_size"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(libs, rotation=20, ha="right")
    ax.set_ylabel("Final Accuracy (mean ± std)")
    ax.set_title("Library Comparison – Final Accuracy")
    ax.set_ylim(0, 1.15)

    if is_placeholder:
        _annotate_placeholder(ax, "Synthetic data – run experiments to populate")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig5_comparison_bars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the TinkerRL paper."
    )
    parser.add_argument(
        "--results-dir", type=str, default="experiments/results/",
        help="Root directory containing experiment result files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="paper/figures/",
        help="Directory where figures will be saved (PDF + PNG)",
    )
    parser.add_argument(
        "--no-placeholder", action="store_true",
        help="Skip generating placeholder figures when data is missing",
    )
    args = parser.parse_args()

    _apply_style()

    print(f"\n{'='*60}")
    print(f"  TinkerRL – Figure Generation")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*60}\n")

    # ---- Figure 1: Learning curves ----------------------------------------
    print("[1/5] Learning curves...")
    lc_data = _load_learning_curves(args.results_dir)
    if lc_data:
        is_ph = False
        print(f"  Loaded data for: {', '.join(lc_data.keys())}")
    else:
        print("  No learning curve data found – using synthetic placeholder.")
        lc_data = _synth_learning_curves()
        is_ph = True
    if not args.no_placeholder or not is_ph:
        plot_learning_curves(lc_data, args.output_dir, is_placeholder=is_ph)

    # ---- Figure 2 & 5: Performance profiles + bar chart ------------------- 
    print("[2/5] Performance profiles & bar chart...")
    final_accs = _load_final_accuracies(args.results_dir)
    if final_accs:
        is_ph_fa = False
        print(f"  Loaded final accuracies for: {', '.join(final_accs.keys())}")
    else:
        print("  No final accuracy data found – using synthetic placeholder.")
        final_accs = _synth_final_accuracies()
        is_ph_fa = True

    if not args.no_placeholder or not is_ph_fa:
        plot_performance_profiles(final_accs, args.output_dir, is_placeholder=is_ph_fa)
        plot_comparison_bars(final_accs, args.output_dir, is_placeholder=is_ph_fa)

    # ---- Figure 3: Scaling ------------------------------------------------
    print("[3/5] Scaling plot...")
    scaling_data = _load_scaling_data(args.results_dir)
    if scaling_data:
        is_ph_sc = False
        print(f"  Loaded scaling data for: {', '.join(scaling_data.keys())}")
    else:
        print("  No scaling data found – using synthetic placeholder.")
        scaling_data = _synth_scaling()
        is_ph_sc = True

    if not args.no_placeholder or not is_ph_sc:
        plot_scaling(scaling_data, args.output_dir, is_placeholder=is_ph_sc)

    # ---- Figure 4: Sensitivity heatmap ------------------------------------
    print("[4/5] Sensitivity heatmap...")
    sens_data = _load_sensitivity_csv(args.results_dir)
    if sens_data:
        is_ph_s = False
        print(f"  Loaded sensitivity data for: {', '.join(sens_data.keys())}")
    else:
        print("  No sensitivity CSV found – using synthetic placeholder.")
        sens_data = _synth_sensitivity()
        is_ph_s = True

    if not args.no_placeholder or not is_ph_s:
        plot_sensitivity_heatmap(sens_data, args.output_dir, is_placeholder=is_ph_s)

    print(f"\nAll figures saved to: {args.output_dir}")
    print("  Formats: PDF (vector) + PNG (raster @ 300 DPI)\n")


if __name__ == "__main__":
    main()

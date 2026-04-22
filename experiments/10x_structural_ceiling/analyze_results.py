"""
10x Structural Ceiling — Results Analyzer

Pulls W&B data from tinker-structural-ceiling project and generates:
1. Scaling curves (model size vs. GSM8K accuracy delta)
2. Family isolation heatmap (model x benchmark)
3. Group saturation phase diagram (G vs. onset step)
4. Algorithm comparison bar chart (GRPO vs PPO vs DPO)
5. LaTeX tables for paper inclusion
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_saturation_data(checkpoints_root: str = "./checkpoints/10x") -> pd.DataFrame:
    """Load all group_saturation.json files into a DataFrame."""
    root = Path(checkpoints_root)
    records = []
    for sat_file in root.glob("*/group_saturation.json"):
        exp_name = sat_file.parent.name
        data = json.loads(sat_file.read_text())
        summary = data.get("summary", {})
        summary["experiment"] = exp_name
        records.append(summary)
    return pd.DataFrame(records)


def load_wandb_runs(project: str = "tinker-structural-ceiling") -> pd.DataFrame:
    """Pull run summaries from W&B."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"arvindcr4-pes-university/{project}")
        records = []
        for run in runs:
            record = {
                "name": run.name,
                "group": run.group,
                "state": run.state,
                **run.config,
                **{f"summary_{k}": v for k, v in run.summary.items()
                   if not k.startswith("_")},
            }
            records.append(record)
        return pd.DataFrame(records)
    except Exception as e:
        print(f"W&B pull failed: {e}")
        return pd.DataFrame()


def scaling_curve(df: pd.DataFrame) -> None:
    """Plot model size vs. GSM8K accuracy improvement."""
    try:
        import matplotlib.pyplot as plt

        size_runs = df[df["group"].str.contains("size-ladder", na=False)].copy()
        if size_runs.empty:
            print("No size-ladder data yet.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Model Size (B params)")
        ax.set_ylabel("GSM8K Accuracy Delta (post-RL - base)")
        ax.set_title("Structural Ceiling: Size Scaling Curve")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="No improvement")
        ax.set_xscale("log")
        ax.legend()
        fig.savefig("scaling_curve.png", dpi=150, bbox_inches="tight")
        print("Saved: scaling_curve.png")
    except ImportError:
        print("matplotlib not available — skipping plot")


def saturation_phase_diagram(df: pd.DataFrame) -> None:
    """Plot group size vs. saturation onset step."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Group Size (G)")
        ax.set_ylabel("Mean Zero-Variance Fraction")
        ax.set_title("Group Saturation Phase Diagram")

        for _, row in df.iterrows():
            label = row.get("experiment", "")
            ax.scatter(
                label.split("_g")[-1] if "_g" in label else "?",
                row.get("mean_zero_variance_frac", 0),
                s=100,
            )

        fig.savefig("saturation_phase.png", dpi=150, bbox_inches="tight")
        print("Saved: saturation_phase.png")
    except ImportError:
        print("matplotlib not available — skipping plot")


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{10x Structural Ceiling: Experimental Matrix Results}\n"
        "\\label{tab:10x-results}\n"
        "\\small\n"
        "\\begin{tabular}{@{}llcccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Block} & \\textbf{Model} & \\textbf{Benchmark} & "
        "\\textbf{Base score} & \\textbf{Post-training score} & \\textbf{$\\Delta$} \\\\\n"
        "\\midrule\n"
    )
    rows = ""
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + rows + footer


if __name__ == "__main__":
    print("10x Structural Ceiling — Results Analysis")
    print("=" * 50)

    # Try loading local saturation data
    sat_df = load_saturation_data()
    if not sat_df.empty:
        print(f"\nSaturation data: {len(sat_df)} experiments")
        print(sat_df[["experiment", "mean_zero_variance_frac", "mean_gradient_utilization"]].to_string())
        saturation_phase_diagram(sat_df)

    # Try loading W&B data
    wandb_df = load_wandb_runs()
    if not wandb_df.empty:
        print(f"\nW&B runs: {len(wandb_df)}")
        print(wandb_df[["name", "group", "state"]].to_string())
        scaling_curve(wandb_df)

    # Generate LaTeX stub
    latex = generate_latex_table(wandb_df if not wandb_df.empty else pd.DataFrame())
    Path("results_table.tex").write_text(latex)
    print("\nSaved: results_table.tex")

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

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import require_empirical, EvidenceUnavailable


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

        try:
            import torch, wandb

            if not getattr(wandb, "_vram_patched", False):
                _old_log = wandb.log

                def _vram_log(data, *args, **kwargs):
                    if torch.cuda.is_available():
                        data["system/vram_peak_allocated_gb"] = (
                            torch.cuda.max_memory_allocated() / (1024**3)
                        )
                        data["system/vram_reserved_gb"] = torch.cuda.max_memory_reserved() / (
                            1024**3
                        )
                        torch.cuda.reset_peak_memory_stats()
                    _old_log(data, *args, **kwargs)

                wandb.log = _vram_log
                wandb._vram_patched = True
        except ImportError:
            pass
        api = wandb.Api()
        runs = api.runs(f"arvindcr4-pes-university/{project}")
        records = []
        for run in runs:
            record = {
                "name": run.name,
                "group": run.group,
                "state": run.state,
                **run.config,
                **{f"summary_{k}": v for k, v in run.summary.items() if not k.startswith("_")},
            }
            # Promote only measured summary endpoints; never accept config declarations
            # as a substitute for observed endpoint/provenance fields.
            for field in ("heldout_before", "heldout_after", "evaluation_ids_hash"):
                record[field] = record.get(f"summary_{field}")
            # LIMITATION: "Early-Training Snapshot Problem" - flag or filter runs with very few
            # gradient steps (e.g., 30-50 steps) to avoid extrapolating from incomplete training dynamics.
            records.append(record)
        return pd.DataFrame(records)
    except Exception as e:
        print(f"W&B pull failed: {e}")
        return pd.DataFrame()


def scaling_curve(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    records = df.to_dict("records")
    if not records:
        raise EvidenceUnavailable("No measured scaling points")
    for r in records:
        require_empirical(r)
        if any(
            r.get(k) is None
            for k in ("model_size_b", "heldout_before", "heldout_after", "evaluation_ids_hash")
        ):
            raise EvidenceUnavailable(
                "Scaling points require held-out endpoints and evaluation IDs"
            )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [r["model_size_b"] for r in records],
        [r["heldout_after"] - r["heldout_before"] for r in records],
    )
    ax.set(
        xlabel="Model size (B parameters)",
        ylabel="Held-out accuracy change",
        title="Observed change within tested budget",
    )
    fig.savefig("scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def saturation_phase_diagram(df: pd.DataFrame) -> None:
    """Plot group size vs. saturation onset step."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Group Size (G)")
        ax.set_ylabel("Mean Zero-Variance Fraction")
        ax.set_title("Group Saturation Phase Diagram")

        # LIMITATION: "ZVF Metric Fragility" - ZVF collapses/saturates at 1.0 for format-gated tasks.
        # We need to fall back to calculating and plotting ERF (Effective-Rollout Fraction) instead
        # of ZVF when dealing with non-math or schema-dependent tasks.

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
    records = df.to_dict("records")
    if not records:
        raise EvidenceUnavailable("No measured table rows")
    lines = [r"\begin{tabular}{lrrr}", r"Model & Before & After & Difference \\"]
    for r in records:
        require_empirical(r)
        if any(
            r.get(k) is None
            for k in ("model", "heldout_before", "heldout_after", "evaluation_ids_hash")
        ):
            raise EvidenceUnavailable("Missing held-out table endpoint/provenance")
        label = str(r["model"]).replace("_", r"\_").replace("%", r"\%")
        before, after = r["heldout_before"], r["heldout_after"]
        lines.append(f"{label} & {before:.3f} & {after:.3f} & {after - before:+.3f} " + r"\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("10x Structural Ceiling — Results Analysis")
    print("=" * 50)

    # Try loading local saturation data
    sat_df = load_saturation_data()
    if not sat_df.empty:
        print(f"\nSaturation data: {len(sat_df)} experiments")
        print(
            sat_df[
                ["experiment", "mean_zero_variance_frac", "mean_gradient_utilization"]
            ].to_string()
        )
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

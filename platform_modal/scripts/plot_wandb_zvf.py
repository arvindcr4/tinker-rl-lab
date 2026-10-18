#!/usr/bin/env python3
"""Plot full W&B ZVF history, reject bad units, and archive exact plotted rows."""

import argparse
from _plot_provenance import archive_history, metric_spec


def main():
    import matplotlib.pyplot as plt
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path")
    parser.add_argument("--metric", default="zero_variance_frac")
    parser.add_argument("--label")
    parser.add_argument("--unit", choices=["fraction", "percent", "scalar"])
    parser.add_argument("--output", default="zvf_plot.png")
    args = parser.parse_args()
    spec = metric_spec(args.metric, args.label, args.unit)
    run = wandb.Api().run(args.run_path)
    rows, metadata = archive_history(run, spec, args.output)
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(
        [r["step"] for r in rows], [r["plotted_value"] for r in rows], marker="o", label=run.name
    )
    axes.set_xlabel("W&B logging step (_step)")
    axes.set_ylabel(f"{spec['label']} ({spec['unit']})")
    axes.set_title(f"{spec['label']} over logging steps")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()

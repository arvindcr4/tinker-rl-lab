#!/usr/bin/env python3
import argparse
import sys

try:
    import wandb
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure wandb, pandas, and matplotlib are installed (e.g., pip install wandb pandas matplotlib)")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot ZVF (Zero-Variance Fraction) from a W&B run.")
    parser.add_argument(
        "run_path", 
        type=str, 
        help="W&B run path in the format <entity>/<project>/<run_id>"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="zero_variance_frac", 
        help="Exact metric name in W&B to plot (e.g., 'zvf' or 'zero_variance_frac')"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="zvf_plot.png", 
        help="Path to save the output plot (default: zvf_plot.png)"
    )
    
    args = parser.parse_args()

    api = wandb.Api()
    
    try:
        run = api.run(args.run_path)
    except Exception as e:
        print(f"Error fetching run '{args.run_path}': {e}")
        sys.exit(1)

    print(f"Fetching history for metric '{args.metric}' from run '{run.name}' ({args.run_path})...")
    
    # Fetch run history. We request a large number of samples to ensure we get un-downsampled data if possible.
    try:
        history = run.history(keys=["_step", args.metric], samples=100000, pandas=True)
    except Exception as e:
        print(f"Error fetching history: {e}")
        sys.exit(1)
    
    if history.empty or args.metric not in history.columns:
        print(f"Error: Metric '{args.metric}' not found in the run history.")
        
        # Try to suggest similar metric names available in the summary
        try:
            available_metrics = list(run.summary.keys())
            candidates = [k for k in available_metrics if "zvf" in k.lower() or "var" in k.lower() or "frac" in k.lower()]
            if candidates:
                print(f"Did you mean one of these metrics? {candidates}")
                print(f"You can specify it using: --metric <name>")
            else:
                print("No obvious alternative metrics found in run summary.")
        except Exception:
            pass
            
        sys.exit(1)

    # Drop rows where the metric or step might be missing
    history = history.dropna(subset=[args.metric, "_step"])

    if history.empty:
        print(f"Error: Metric '{args.metric}' exists but contains no non-null values.")
        sys.exit(1)

    # Sort history chronologically
    history = history.sort_values(by="_step")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        history["_step"], 
        history[args.metric], 
        marker='o', 
        linestyle='-', 
        markersize=3, 
        alpha=0.8,
        label=args.metric
    )
    
    plt.xlabel("Training Step")
    plt.ylabel("Zero-Variance Fraction (ZVF)")
    plt.title(f"ZVF over Time\nRun: {run.name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # ZVF is usually a fraction between 0 and 1, so we lock the y-axis boundaries.
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(args.output, dpi=150)
        print(f"Success! Plot saved to {args.output}")
    except Exception as e:
        print(f"Error saving plot to {args.output}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

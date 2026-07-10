#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Effective Rollout Fraction (ERF) over time from W&B.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name (e.g., 'tinker-rl-scaling' or 'entity/project')")
    parser.add_argument("--metric", type=str, default="train/erf", help="Name of the ERF metric in W&B (e.g., train/erf, eval/erf, or train/accuracy)")
    parser.add_argument("--output", type=str, default="erf_plot.png", help="Output plot filename")
    parser.add_argument("--group", type=str, default=None, help="Filter runs by a specific group")
    parser.add_argument("--runs", nargs="+", help="Specific run IDs to plot. If not provided, plots all matching runs.")
    parser.add_argument("--smooth", type=int, default=1, help="Smoothing window size for the plot")
    
    args = parser.parse_args()
    
    api = wandb.Api()
    
    filters = {}
    if args.group:
        filters["group"] = args.group
        
    if args.runs:
        # Fetch specific runs
        runs = []
        for run_id in args.runs:
            # Handle if the user passes full path or just ID
            path = run_id if "/" in run_id else f"{args.project}/{run_id}"
            try:
                runs.append(api.run(path))
            except Exception as e:
                print(f"Could not find run {path}: {e}")
    else:
        # Fetch runs by project (and optionally group)
        print(f"Fetching runs from project '{args.project}'...")
        runs = api.runs(args.project, filters=filters)
        
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    data_found = False
    
    for run in runs:
        # Check if metric exists in the run's summary or if we should just try fetching history anyway
        print(f"Processing run: {run.name} ({run.id})")
        
        # Fetch the metric history
        # We also fetch _step to plot against time
        try:
            history = run.history(keys=["_step", args.metric], samples=10000)
            if not history.empty and args.metric in history.columns:
                history = history.dropna(subset=[args.metric])
                if not history.empty:
                    # Apply smoothing if requested
                    if args.smooth > 1:
                        history[args.metric] = history[args.metric].rolling(window=args.smooth, min_periods=1).mean()
                    
                    plt.plot(history["_step"], history[args.metric], label=run.name, alpha=0.8)
                    data_found = True
                else:
                    print(f"  -> Metric '{args.metric}' is all NaNs for this run.")
            else:
                print(f"  -> Metric '{args.metric}' not found in run history.")
        except Exception as e:
            print(f"  -> Error fetching history for {run.name}: {e}")
                
    if not data_found:
        print(f"\nNo data found for metric '{args.metric}'. Please check the metric name.")
        print("Tip: If ERF is logged under a different name (like train/accuracy), use --metric train/accuracy")
        return
        
    plt.xlabel("Step")
    plt.ylabel(f"Effective Rollout Fraction ({args.metric})")
    plt.title("Effective Rollout Fraction (ERF) over Time")
    
    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Runs")
    plt.tight_layout()
    
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nPlot successfully saved to {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot a declared W&B metric with a checksummed full-history input export."""
import argparse
from _plot_provenance import archive_history, metric_spec


def main():
    import matplotlib.pyplot as plt
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--metric', default='train/erf')
    parser.add_argument('--label')
    parser.add_argument('--unit', choices=['fraction', 'percent', 'scalar'])
    parser.add_argument('--output', default='erf_plot.png')
    parser.add_argument('--group')
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--smooth', type=int, default=1)
    args = parser.parse_args()
    spec = metric_spec(args.metric, args.label, args.unit)
    api = wandb.Api()
    runs = ([api.run(r if '/' in r else f'{args.project}/{r}') for r in args.runs]
            if args.runs else api.runs(args.project, filters={'group': args.group} if args.group else {}))
    figure, axes = plt.subplots(figsize=(10, 6))
    count = 0
    for run in runs:
        rows, metadata = archive_history(run, spec, args.output, args.smooth)
        axes.plot([r['step'] for r in rows], [r['plotted_value'] for r in rows], label=run.name)
        count += 1
    if not count:
        raise ValueError('no eligible runs; no plot created')
    axes.set_xlabel('W&B logging step (_step)')
    axes.set_ylabel(f"{spec['label']} ({spec['unit']})")
    axes.set_title(f"{spec['label']} over logging steps")
    axes.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=300)
    plt.close(figure)


if __name__ == '__main__':
    main()

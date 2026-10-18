"""Review28_v2: render unmodified measured correlations and current verdicts."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments'/'results'
FIG_DIR=ROOT/'figures'/'review28_v2'


def read_rows(path):
    with Path(path).open() as fh:
        return list(csv.DictReader((line for line in fh if not line.startswith('#')),delimiter='\t'))


def render(lag_rows, prediction_rows, output_dir):
    """Return plotted point values and annotation text for auditability."""
    grouped=defaultdict(list)
    for row in lag_rows:
        value=float(row['r'])
        if np.isfinite(value): grouped[(row['method'],int(row['lag']))].append(value)
    values={key:float(np.mean(points)) for key,points in grouped.items()}
    fig,ax=plt.subplots(figsize=(10,7))
    for method in sorted({key[0] for key in values}):
        points=sorted((lag,value) for (m,lag),value in values.items() if m==method)
        ax.plot([x for x,_ in points],[y for _,y in points],marker='o',label=method)
    ax.set(ylim=(-1,1),xlabel='Lag (training observations)',ylabel='Mean measured Pearson correlation',
           title='Retrospective reward/ZVF lag correlations; no interval supplied')
    if values: ax.legend()
    text='\n'.join(f"{row['id']}: {row.get('pass','NOT_EVALUABLE')} (value={row.get('value','unavailable')})" for row in prediction_rows)
    fig.text(.1,.02,text or 'No prediction records available',fontsize=8)
    fig.tight_layout(rect=(0,.2,1,1))
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(output_dir/f'zvf_iter50.{ext}',dpi=150)
    plt.close(fig)
    return values,text


def main():
    render(read_rows(RESULTS/'zvf_iter50_lagged_corr_review28_v2.tsv'),
           read_rows(RESULTS/'zvf_iter50_predictions_review28_v2.tsv'),FIG_DIR)


if __name__=='__main__':main()

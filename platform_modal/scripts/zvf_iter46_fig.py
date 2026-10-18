"""Review28_v2 Iso-G sensitivity scenarios, using the analysis implementation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zvf_iter46_isog import yield_iid, yield_emp
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'/'review28_v2'


def make_pdf(output_dir=None):
    output_dir=Path(output_dir or FIG);output_dir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5))
    gs=list(range(1,33))
    for p in (.1,.5,.9):
        ax.plot(gs,[yield_iid(p,g) for g in gs],label=f'iid p={p}')
        for delta in (-.122,.122):
            ax.plot(gs,[yield_emp(p,g,delta) for g in gs],ls='--',alpha=.5,
                    label=f'sensitivity p={p}, signed correction={delta}')
    ax.set(xlabel='Group size',ylabel='Projected yield',ylim=(0,1),
           title='Sensitivity scenarios; corrections are not measured transfer effects')
    ax.legend(fontsize=7);fig.tight_layout()
    paths=[output_dir/f'zvf_iter46.{ext}' for ext in ('pdf','png')]
    for path in paths:fig.savefig(path,dpi=150)
    plt.close(fig)
    return tuple(map(str,paths))


if __name__=='__main__':make_pdf()

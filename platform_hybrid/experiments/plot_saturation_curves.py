
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import require_empirical, EvidenceUnavailable

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def saturation_model(t, R_max, lam, R_0):
    return R_0 + (R_max - R_0) * (1 - np.exp(-lam * t))

def plot_saturation(results_file="experiments/master_results.json", output="observed_trajectories_v2.pdf"):
    records = json.loads(Path(results_file).read_text()).get("experiments", [])
    if not records:
        raise EvidenceUnavailable("No measured trajectories")
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for exp in records:
        require_empirical(exp)
        trace = exp.get("reward_trace", [])
        if not trace:
            raise EvidenceUnavailable(f"Missing observed trajectory: {exp['run_id']}")
        ax.plot(range(len(trace)), trace, label=exp["run_id"])
        plotted += 1
    ax.set(title="Observed trajectories within tested budgets", xlabel="Logged training step", ylabel="Observed training reward")
    ax.legend(bbox_to_anchor=(1.05,1),loc="upper left")
    fig.tight_layout(); fig.savefig(output); plt.close(fig)
    return plotted

if __name__ == '__main__':
    plot_saturation()

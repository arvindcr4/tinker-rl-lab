import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import require_empirical, EvidenceUnavailable

import json
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

# Ignore optimize warnings for poorly fitting data
warnings.filterwarnings("ignore")


def saturation_model(t, R_max, lam, R_0):
    return R_0 + (R_max - R_0) * (1 - np.exp(-lam * t))


def analyze_saturation(results_file):
    path = Path(results_file)
    if not path.exists():
        raise EvidenceUnavailable("Missing measured run file")
    rows = []
    for exp in json.loads(path.read_text()).get("experiments", []):
        require_empirical(exp)
        trace = exp.get("reward_trace", [])
        row = dict(
            run_id=exp["run_id"],
            interpretation="observed finite-budget trajectory; no structural ceiling identified",
        )
        if len(trace) < 5:
            row.update(fit_status="insufficient_points")
        else:
            try:
                popt, _ = curve_fit(
                    saturation_model,
                    np.arange(len(trace)),
                    np.asarray(trace),
                    bounds=([0, 0, 0], [1, 1, 1]),
                )
                row.update(
                    fit_status="fitted_descriptive",
                    parameters=popt.tolist(),
                    budget_steps=len(trace),
                )
            except (ValueError, RuntimeError) as exc:
                row.update(fit_status="failed", reason=type(exc).__name__)
        rows.append(row)
    if not rows:
        raise EvidenceUnavailable("No measured trajectories")
    print(json.dumps(rows, indent=2, allow_nan=False))
    return rows


if __name__ == "__main__":
    analyze_saturation("/Users/arvind/research/paper/tinker-rl-lab/experiments/master_results.json")

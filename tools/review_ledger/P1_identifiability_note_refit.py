"""Reproduce the P1 note's three-parameter refit and sensitivity audit.

Adapted from the prior P1 review's scratch refit.py. This is the note's
unconstrained-direction offset model, NOT P1_scaling's monotone offset fit.
One observation per single-seed anchor; no step-level inferential replication.
OLS intervals and permutations are descriptive, conditional on the supplied
traces; neither propagates fit uncertainty nor resolves recipe confounding.
Run: python tools/review_ledger/P1_identifiability_note_refit.py
"""

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
from scipy.optimize import curve_fit

from P1_scaling_rawtrace_refit import ANCHORS, perm_group_p

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "platform_hybrid/experiments/tinker-runs/results"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P1_identifiability_note"
SEED = 1092026
PERMUTATIONS = 10000


def saturation(t, r0, rinf, lam):
    return r0 + (rinf - r0) * (1 - np.exp(-lam * t))


def fit(y, lo, hi):
    t = np.arange(1, len(y) + 1, dtype=float)
    p0 = [
        float(np.clip(y[0], lo[0], hi[0])),
        float(np.clip(max(y.max(), y.mean() + 0.05), lo[1], hi[1] - 1e-9)),
        0.3,
    ]
    p, _ = curve_fit(saturation, t, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    return p


def at_bound(value, lo, hi, tol=1e-3):
    return bool(value <= lo + tol or value >= hi - tol)


def regression(rows, key):
    x = np.log10([r["N"] for r in rows])
    y = np.array([r[key] for r in rows])
    result = stats.linregress(x, y)
    df = len(x) - 2
    critical = stats.t.ppf(0.975, df)
    # Algebraically the OLS slope; permute x as in the original scratch audit.
    xc, yc = x - x.mean(), y - y.mean()
    denominator = xc @ xc
    if len(x) <= 8:
        shuffles = (xc[list(p)] for p in itertools.permutations(range(len(x))))
        count = math.factorial(len(x))
        correction = 0
    else:
        rng = np.random.default_rng(SEED)
        shuffles = (rng.permutation(xc) for _ in range(PERMUTATIONS))
        count = PERMUTATIONS
        correction = 1
    exceedances = sum(abs(p @ yc / denominator) >= abs(result.slope) - 1e-12 for p in shuffles)
    return dict(
        anchors=[r["name"] for r in rows],
        n=len(x),
        df=df,
        slope=result.slope,
        se=result.stderr,
        ci=[result.slope - critical * result.stderr, result.slope + critical * result.stderr],
        p_perm=(exceedances + correction) / (count + correction),
        permutations=count,
        exact=(correction == 0),
        mds80_approx=(critical + stats.t.ppf(0.80, df)) * result.stderr,
    )


def main():
    rows = []
    for name, (filename, total, active, arch) in ANCHORS.items():
        path = TRACES / filename
        data = json.loads(path.read_text())
        y = np.asarray(data["reward_trace"], float)
        old = fit(y, [-1.5, 0, 1e-4], [1.5, 1.5, 50])
        new = fit(y, [0, 0, 1e-4], [1, 1, 50])
        rows.append(
            dict(
                name=name,
                source=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model=data["model"],
                run_id=data["run_id"],
                N=total,
                N_active=active,
                arch=arch,
                n=len(y),
                lr=data["lr"],
                group=data["group"],
                rank=data["rank"],
                seed=data["seed"],
                partial=str(data["run_id"]).startswith("partial"),
                first=y[0],
                first5=y[:5].mean(),
                mean=y.mean(),
                last10=y[-10:].mean(),
                peak=y.max(),
                peak_step=int(np.argmax(y)) + 1,
                zeros=int(np.sum(y == 0)),
                rinf_historical=old[1],
                log10lam_historical=np.log10(old[2]),
                r0=new[0],
                rinf=new[1],
                log10lam=np.log10(new[2]),
                rinf_at_bound=at_bound(new[1], 0, 1),
                lambda_at_bound=at_bound(new[2], 1e-4, 50, 1e-4),
            )
        )
    long = [r for r in rows if r["n"] >= 20]
    pools = {
        "all12": rows,
        "long7": long,
        "without_Nemotron11": [r for r in rows if r["name"] != "Nemotron-120B"],
        "without_Nemotron6": [r for r in long if r["name"] != "Nemotron-120B"],
        "interior4": [r for r in long if not r["rinf_at_bound"] and not r["lambda_at_bound"]],
    }
    regressions = {
        pool: {key: regression(members, key) for key in ("rinf", "log10lam", "mean")}
        for pool, members in pools.items()
    }
    groups = {}
    for pool, members in pools.items():
        if pool == "interior4":
            continue
        values = [r["mean"] for r in members]
        labels = [int(r["arch"] == "moe") for r in members]
        gap, p1, count = perm_group_p(values, labels)
        _, p2, _ = perm_group_p(values, labels, one_sided=False)
        groups[pool] = dict(
            n_moe=sum(labels),
            n_dense=len(labels) - sum(labels),
            gap=gap,
            p_one_sided=p1,
            p_two_sided=p2,
            partitions=count,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=SEED,
        numpy=np.__version__,
        scipy=scipy.__version__,
        rows=rows,
        regressions=regressions,
        groups=groups,
    )
    (OUT / "refit.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(dict(regressions=regressions, groups=groups), indent=2))


if __name__ == "__main__":
    main()

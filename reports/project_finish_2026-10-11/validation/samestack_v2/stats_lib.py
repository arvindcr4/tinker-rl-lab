import itertools
import numpy as np
from scipy import stats


def signflip(d):
    d = np.asarray(d, float)
    obs = abs(d.sum())
    hits = sum(abs((np.array(s) * d).sum()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(d)))
    return hits / 2 ** len(d)


def boot_ci(d, B=20000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return list(np.percentile(m, [2.5, 97.5]))


def paired(a, b):
    """a, b: dict seed->acc. Returns stats over shared seeds."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    md, sd = d.mean(), d.std(ddof=1)
    t = stats.ttest_1samp(d, 0)
    tc = stats.t.ppf(0.975, n - 1)
    return {"n": n, "seeds": seeds, "diffs": [round(x, 4) for x in d], "mean": md, "sd": sd,
            "t_ci": [md - tc * sd / np.sqrt(n), md + tc * sd / np.sqrt(n)],
            "boot_ci": boot_ci(d), "p_t": float(t.pvalue), "p_signflip": signflip(d)}


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def mde(sd, n, power=0.8, alpha=0.05):
    """Paired t MDE via noncentral t (exact)."""
    from scipy.optimize import brentq
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    f = lambda delta: (1 - stats.nct.cdf(tc, df, delta / (sd / np.sqrt(n)))) - power
    return brentq(f, 1e-9, 10 * sd)


def n_needed(sd, effect, power=0.8, alpha=0.05):
    for n in range(2, 1000):
        df = n - 1
        tc = stats.t.ppf(1 - alpha / 2, df)
        if 1 - stats.nct.cdf(tc, df, effect / (sd / np.sqrt(n))) >= power:
            return n

"""Review28 v2 exploratory estimators with explicit ties and eligibility."""
from itertools import product
import numpy as np
from scipy.stats import rankdata

PROTOCOL = 'review28_v2'


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    mask = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores, labels = scores[mask], labels[mask]
    n1, n0 = sum(labels == 1), sum(labels == 0)
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(scores)[labels == 1].sum() - n1*(n1+1)/2)/(n1*n0))


def average_precision(scores, labels):
    pairs = sorted(zip(scores, labels), reverse=True)
    positive = sum(label == 1 for _, label in pairs)
    if not positive:
        return float('nan')
    tp = fp = 0
    area = 0.0
    i = 0
    while i < len(pairs):
        j = i
        added = 0
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            added += pairs[j][1] == 1
            j += 1
        tp += added
        fp += j-i-added
        area += added / positive * tp/(tp+fp)
        i = j
    return float(area)


def signed_rank(deltas):
    """Wilcox zero convention and exact two-sided sign randomization, n<=20."""
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float('nan'), float('nan')
    d = d[d != 0]
    if not len(d):
        return 0.0, 1.0
    ranks = rankdata(abs(d))
    w = float(ranks[d > 0].sum())
    if len(d) > 20:
        raise ValueError('Exact signed-rank protocol supports at most 20 nonzero pairs')
    center = ranks.sum()/2
    extreme = sum(abs(np.dot(signs, ranks)-center) >= abs(w-center)-1e-12
                  for signs in product([0,1], repeat=len(d)))
    return w, extreme / 2**len(d)


def quantile_partitions(rows, key, n_bins):
    if n_bins < 1:
        raise ValueError('n_bins must be positive')
    ordered = sorted(rows, key=lambda row: row[key])
    if not ordered:
        return []
    return [[ordered[int(i)] for i in part]
            for part in np.array_split(np.arange(len(ordered)), min(n_bins, len(ordered)))]


def partial_auc(scores, nuisance, labels):
    """Rank residualization of predictor with fixed increasing-score direction."""
    scores, nuisance, labels = map(lambda a: np.asarray(a, float), (scores,nuisance,labels))
    mask = np.isfinite(scores) & np.isfinite(nuisance) & np.isin(labels,[0,1])
    scores,nuisance,labels=scores[mask],nuisance[mask],labels[mask]
    if len(scores)<3: return float('nan')
    x=np.column_stack([np.ones(len(scores)),rankdata(nuisance)])
    y=rankdata(scores)
    residual=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    # Numerical roundoff cannot turn a constant residual into discrimination.
    if np.ptp(residual)<1e-10: residual=np.zeros(len(residual))
    return auc(residual,labels)


def partial_auc_bootstrap(scores, nuisance, labels, B=2000, seed=110):
    arrays=[np.asarray(a,float) for a in (scores,nuisance,labels)]
    mask=np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isin(arrays[2],[0,1])
    arrays=[a[mask] for a in arrays]
    point=partial_auc(*arrays)
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(B if len(arrays[0]) else 0):
        ix=rng.integers(0,len(arrays[0]),len(arrays[0]))
        value=partial_auc(*(a[ix] for a in arrays))
        if np.isfinite(value): values.append(value)
    lo,hi=np.quantile(values,[.025,.975]) if values else (float('nan'),float('nan'))
    return point,float(lo),float(hi)


def partial_correlation_bootstrap(z, outcome, nuisance, B=2000, seed=26):
    arrays=[np.asarray(a,float) for a in (z,outcome,nuisance)]
    def estimate(arrays):
        z,y,r=arrays
        if len(z)<3:return float('nan')
        x=np.column_stack([np.ones(len(r)),r])
        rz=z-x@np.linalg.lstsq(x,z,rcond=None)[0]
        ry=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
        if np.ptp(rz)<1e-10 or np.ptp(ry)<1e-10:return float('nan')
        return float(np.corrcoef(rz,ry)[0,1])
    point=estimate(arrays)
    rng=np.random.default_rng(seed); values=[]
    for _ in range(B if len(arrays[0]) else 0):
        ix=rng.integers(0,len(arrays[0]),len(arrays[0]))
        value=estimate([a[ix] for a in arrays])
        if np.isfinite(value):values.append(value)
    ci=tuple(np.quantile(values,[.025,.975])) if values else (float('nan'),float('nan'))
    return point,ci

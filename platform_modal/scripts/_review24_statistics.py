"""Corrected analysis protocol review24-v2; no artifact or provider side effects."""
import hashlib
import math
import numpy as np


def stable_seed(master, *identity):
    payload = repr((master, 'review24-v2', identity)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def first_crossing(log_budgets, accuracies, target):
    if len(log_budgets) != len(accuracies):
        raise ValueError('budget/accuracy length mismatch')
    pairs = sorted(zip(log_budgets, accuracies))
    if not pairs:
        return None
    if not all(math.isfinite(x) and math.isfinite(y) for x,y in pairs):
        raise ValueError('nonfinite frontier observation')
    if len({x for x,y in pairs}) != len(pairs):
        raise ValueError('duplicate budget; aggregate explicitly first')
    if pairs[0][1] >= target:
        return pairs[0][0]
    for (x0,y0),(x1,y1) in zip(pairs,pairs[1:]):
        if y0 < target <= y1:
            return x0 + (target-y0)/(y1-y0)*(x1-x0)
    return None


def conditional_ccf(x, y, mask, K):
    """Corr(x[t], y[t+lag]); BOTH endpoints must belong to the same bin.

    Original observation indices are preserved. Missing/constant/fewer-than-four
    pairs are unavailable, never measured zero. Caller must retain counts.
    """
    x,y,mask=np.asarray(x,float),np.asarray(y,float),np.asarray(mask,bool)
    if len(x)!=len(y) or len(x)!=len(mask):
        raise ValueError('unaligned observation indices')
    values,counts=[],[]
    n=len(x)
    for lag in range(-K,K+1):
        left=np.arange(max(0,-lag),min(n,n-lag))
        right=left+lag
        valid=mask[left]&mask[right]&np.isfinite(x[left])&np.isfinite(y[right])
        a,b=x[left[valid]],y[right[valid]]
        counts.append(len(a))
        values.append(float(np.corrcoef(a,b)[0,1]) if len(a)>=4 and np.std(a)>0 and np.std(b)>0 else float('nan'))
    return np.asarray(values),counts

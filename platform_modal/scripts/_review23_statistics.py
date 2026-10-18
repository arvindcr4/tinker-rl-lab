"""Corrected review23-v2 identity and inference primitives."""
import math
import numpy as np


def paired_seed_values(runs, groups, metric):
    indexed={g:{} for g in groups}
    for run in runs:
        g=int(run['group_size'])
        if g not in indexed: continue
        seed=int(run['seed'])
        if seed in indexed[g]: raise ValueError(f'duplicate group/seed {g}/{seed}')
        value=float(metric(run))
        if not math.isfinite(value): raise ValueError('nonfinite paired observation')
        indexed[g][seed]=value
    common=sorted(set.intersection(*(set(indexed[g]) for g in groups)))
    return {g:np.asarray([indexed[g][seed] for seed in common],float) for g in groups},common

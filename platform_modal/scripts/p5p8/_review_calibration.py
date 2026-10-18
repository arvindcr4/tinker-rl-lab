"""Label-excluding calibration primitives, protocol review21-v2."""
from __future__ import annotations
import hashlib
import random
import numpy as np
from sklearn.isotonic import IsotonicRegression


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], 'big')


def isotonic_predict(scores, labels, evaluation):
    return IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0, y_max=1).fit(scores, labels).predict(evaluation).tolist()


def crossfit(scores, labels, ids, method='isotonic', seed=42, n_folds=5):
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('at least two unique row IDs required')
    perm=list(ids)
    random.Random(seed).shuffle(perm)
    folds=[perm[i::min(n_folds,len(ids))] for i in range(min(n_folds,len(ids)))]
    output={}
    for held in folds:
        train=[i for i in ids if i not in set(held)]
        x=[scores[i] for i in train]; y=[labels[i] for i in train]
        evaluation=[scores[i] for i in held]
        if method == 'isotonic':
            predicted=isotonic_predict(x,y,evaluation)
        elif method == 'rate':
            scale=sum(y)/sum(x) if sum(x)>0 else 1
            predicted=[min(1.,max(0.,p*scale)) for p in evaluation]
        elif method == 'identity':
            predicted=evaluation
        else:
            raise ValueError('unknown calibration method')
        output.update(zip(held,predicted))
    return [output[i] for i in ids]


def stacked_crossfit(scores, labels, cohorts, seed=42, n_folds=5):
    """Both stages fit inside each outer fold; inner OOF for stage two."""
    n=len(scores); ids=list(range(n)); perm=ids[:]
    random.Random(seed).shuffle(perm)
    if n < 4:
        raise ValueError('stacked calibration requires at least four rows')
    out=[None]*n
    for fold in range(min(n_folds,n)):
        held=perm[fold::min(n_folds,n)]
        train=[i for i in ids if i not in set(held)]
        inner=list(scores)
        first_held={i:scores[i] for i in held}
        for cohort in set(cohorts):
            group=[i for i in train if cohorts[i]==cohort]
            targets=[i for i in held if cohorts[i]==cohort]
            if len(group)>=2:
                for i,p in zip(group,crossfit(scores,labels,group,seed=seed,n_folds=n_folds)):
                    inner[i]=p
                if targets:
                    preds=isotonic_predict([scores[i] for i in group],[labels[i] for i in group],[scores[i] for i in targets])
                    first_held.update(zip(targets,preds))
        preds=isotonic_predict([inner[i] for i in train],[labels[i] for i in train],[first_held[i] for i in held])
        for i,p in zip(held,preds): out[i]=p
    return out

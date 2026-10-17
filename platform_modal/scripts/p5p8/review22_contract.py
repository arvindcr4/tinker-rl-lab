"""Candidate analysis v2: explicit identities and independent-unit inference."""
import math
import random
import statistics

PROTOCOL_ID = 'p5p8-synthesis-v2-independent-units'


def bootstrap_paired_diff(a, b, n_boot=1000, seed=0):
    if len(a) != len(b) or not a or not all(math.isfinite(v) for v in [*a, *b]):
        raise ValueError('equal nonempty finite pairs required')
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 100:
        raise ValueError('at least 100 bootstrap draws required')
    diffs = [x-y for x,y in zip(a,b)]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return statistics.fmean(diffs), means[int(.025*n_boot)], means[min(n_boot-1,int(.975*n_boot))], len(diffs)


def density_ci(n_fire, n_total, n_boot=1500, seed=0):
    """Wilson interval under an explicit *assumed* independent-binomial model.

    Count-only inputs cannot verify independence. No confirmatory verdict follows.
    """
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (n_fire,n_total)) or not 0 <= n_fire <= n_total or n_total == 0:
        raise ValueError('valid integer counts required')
    p=n_fire/n_total; z=1.959963984540054
    center=(p+z*z/(2*n_total))/(1+z*z/n_total)
    half=z*math.sqrt(p*(1-p)/n_total+z*z/(4*n_total*n_total))/(1+z*z/n_total)
    return dict(rate=p,lo=max(0.,center-half),hi=min(1.,center+half),n_fire=n_fire,n_total=n_total,
                interval_model='independent-binomial Wilson; independence unverified',confirmatory=False)


def ratio_ci(num, denom, n_boot=1500, seed=0):
    """Conservative ratio envelope of 97.5% Wilson intervals (Bonferroni).

    Valid only under independent-binomial marginal sampling, not certified here.
    Zero denominator is explicitly undefined/infinite, never epsilon-smoothed.
    """
    def interval(d):
        n,k=d['n_total'],d['n_fire']; density_ci(k,n)
        p=k/n; z=2.241402727604947
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return max(0.,center-half),min(1.,center+half)
    nl,nh=interval(num); dl,dh=interval(denom)
    rn,rd=num['n_fire']/num['n_total'],denom['n_fire']/denom['n_total']
    point=rn/rd if rd else (math.inf if rn else math.nan)
    lo,hi=nl/dh if dh else 0.,nh/dl if dl else math.inf
    return {'ratio':point,'lo':lo,'hi':hi,'excludes_1.0':lo>1 or hi<1,
            'outside_practical_band':point<.1 or point>10,
            'interval_model':'Bonferroni Wilson ratio envelope; sampling assumptions unverified','confirmatory':False}


def highest_density(densities, domain='D4'):
    if domain not in densities or not all(math.isfinite(v) for v in densities.values()):
        raise ValueError('complete finite domain map required')
    return densities[domain] == max(densities.values())


def require_independent_unit_contract():
    raise ValueError('Publication blocked: legacy cross-domain aggregates lack verified independent sampling units and comparable metric definitions; use a separately validated candidate analysis')


def score_gradient_selection(scores, k_top, threshold):
    import numpy as np
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 1<=k_top<=len(values):
        raise ValueError('finite score vector and valid K required')
    order=np.argsort(-values,kind='stable')
    sorted_values=values[order]
    gradients=np.abs(np.diff(sorted_values,prepend=sorted_values[0]+1.0))
    selected=(np.arange(len(values))<k_top)&(gradients<threshold)
    original=np.zeros(len(values),dtype=bool); original[order]=selected
    return order,gradients,gradients[:k_top],original

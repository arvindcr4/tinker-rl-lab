"""Offline statistics for prospective colab-exploratory-v2; no model imports."""
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
from pathlib import Path

PROTOCOL = "colab-exploratory-v2"


def fingerprint(path, config):
    payload = {"protocol": PROTOCOL, "script_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               "config": config, "packages": {}}
    for name in ("torch", "transformers", "peft", "torchao"):
        try:
            payload["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            payload["packages"][name] = None
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def pearson(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("paired observations must have equal lengths")
    if len(xs) < 2 or any(not math.isfinite(x) for x in [*xs, *ys]):
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    denominator = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denominator if denominator else None


def interval(values):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    return [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]] if values else None


def clustered_correlation(rows, x, y, repeats=1000, seed=0):
    groups = {s: [r for r in rows if r['seed'] == s] for s in sorted({r['seed'] for r in rows})}
    def corr(rs):
        return pearson([r[x] for r in rs], [r[y] for r in rs])
    rng = random.Random(seed)
    boots = [corr([r for s in rng.choices(list(groups), k=len(groups)) for r in groups[s]])
             for _ in range(repeats)] if groups else []
    return {"estimate": corr(rows), "ci95": interval(boots), "cluster_unit": "seed",
            "n_independent_seeds": len(groups), "n_observations": len(rows),
            "n_valid_bootstraps": sum(v is not None for v in boots),
            "per_seed": {s: corr(rs) for s,rs in groups.items()},
            "evidence_tier": "exploratory; few-cluster intervals are coarse"}


def finite_pool_summary(pools, k, repeats=500, draws=256, seed=0):
    """Prompt/rollout hierarchical bootstrap for WITH-replacement conditional ZVF.

    Plug-in equals exact conditional expectation; agreement is not independent
    theory validation. Neither estimates known true-p ZVF without uncertainty.
    """
    if not pools or k < 2 or draws < 1 or repeats < 1:
        raise ValueError("nonempty pools, k>=2 and positive repeats/draws required")
    if any(not p or any(v not in (0, 1) for v in p) for p in pools):
        raise ValueError("nonempty binary rollout pools required")
    rng = random.Random(seed)
    def stats(ps):
        plugin = statistics.mean((sum(p)/len(p))**k + (1-sum(p)/len(p))**k for p in ps)
        empirical = statistics.mean(float(len(set(rng.choices(p, k=k))) == 1)
                                    for p in ps for _ in range(draws))
        return empirical, plugin, empirical-plugin
    estimate = stats(pools)
    boots = [stats([rng.choices(p, k=len(p)) for p in rng.choices(pools, k=len(pools))])
             for _ in range(repeats)]
    return {"emp_zvf": estimate[0], "pred_zvf": estimate[1], "difference": estimate[2],
            "emp_ci95": interval([b[0] for b in boots]), "plugin_ci95": interval([b[1] for b in boots]),
            "difference_ci95": interval([b[2] for b in boots]), "true_p_prediction": None,
            "independent_prompts": len(pools), "original_rollouts": sum(map(len,pools)),
            "resampling": "prompts_then_original_rollouts", "group_sampling": "with_replacement",
            "claim_scope": "conditional_finite_pool; plugin is conditional expectation"}

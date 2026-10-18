"""Review 17, protocol v2: descriptive models; no causal/predictive claims.

Categorical main effects are fitted jointly. Bootstrap units must be independent
cells/runs; trajectory steps do not establish independent training replication.
"""
import numpy as np


def additive_r2(rows, values, factors):
    """In-sample OLS R² with intercept and categorical main effects."""
    y = np.asarray(values, dtype=float)
    if len(rows) != len(y) or len(y) < 2 or not np.isfinite(y).all():
        return float('nan')
    columns = [np.ones(len(y))]
    for factor in factors:
        levels = sorted({str(row[factor]) for row in rows})
        columns.extend(np.array([str(row[factor]) == level for row in rows], dtype=float)
                       for level in levels[1:])
    x = np.column_stack(columns)
    total = np.sum((y-y.mean())**2)
    if total <= 1e-12:
        return float('nan')
    prediction = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return float(1 - np.sum((y-prediction)**2)/total)


def ratio_interval(draws, requested):
    """Never hide invalid/unbounded denominator draws in a finite interval."""
    if len(draws) != requested or not np.isfinite(draws).all():
        return float('nan'), float('nan')
    return tuple(float(v) for v in np.quantile(draws, [.025, .975]))


def json_safe(value):
    """Serialize unavailable numeric evidence as null, never NaN/Infinity."""
    import math
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value

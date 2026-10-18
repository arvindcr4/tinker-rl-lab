"""Protocol review18-v2: strict identity joins, no positional pairing."""

import math


def keyed_metric(rows, metric, keys):
    out = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        if key in out:
            raise ValueError(f"Duplicate observation identity: {key}")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError("Nonfinite paired metric")
        out[key] = value
    return out


def paired_values(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("Paired comparisons require explicit identity mappings")
    if not a or set(a) != set(b):
        raise ValueError("Incomplete paired observation identities")
    if not all(math.isfinite(float(v)) for v in list(a.values()) + list(b.values())):
        raise ValueError("Nonfinite paired metric")
    keys = sorted(a)
    return [a[k] for k in keys], [b[k] for k in keys]

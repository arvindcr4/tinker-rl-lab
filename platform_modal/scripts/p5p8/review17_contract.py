"""Prospective review17-v2 measurement contract; never infer missing provenance."""

import csv
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

VERSION = "review17-v2"


def load_seed_metric(path, metric):
    out = {}
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"method", "seed", metric}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing explicit source column {metric}")
        for row in reader:
            seed = row["seed"].strip()
            if seed.lower() in {"agg", "aggregate", "all", "mean"}:
                continue
            if not seed:
                raise ValueError("Missing seed identity")
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError("Nonfinite seed metric")
            arm = out.setdefault(row["method"].lower(), {})
            if seed in arm:
                raise ValueError("Duplicate method/seed")
            arm[seed] = value
    if not out:
        raise ValueError("No independent seed observations")
    return out


def bootstrap(values, n_boot, seed):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite observations required")
    if n_boot < 2:
        raise ValueError("At least two bootstrap draws required")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_result(base, method, n_boot=2000, seed=20260705):
    if not isinstance(base, Mapping) or not isinstance(method, Mapping):
        raise ValueError("Pairing requires explicit seed identities")
    if set(base) != set(method):
        raise ValueError("Incomplete seed pairing")
    ids = sorted(base)
    delta, lo, hi = bootstrap([method[s] - base[s] for s in ids], n_boot, seed)
    return {
        "delta": delta,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(ids),
        "sample_ids": ids,
        "significant": lo > 0 or hi < 0,
        "analysis_version": VERSION,
        "ci_method": {
            "method": "paired_seed_percentile_bootstrap",
            "n_boot": n_boot,
            "seed": seed,
            "ci_level": 0.95,
            "rng": "numpy.default_rng",
        },
    }


def validate_interval(lo, hi):
    return (
        isinstance(lo, (int, float))
        and isinstance(hi, (int, float))
        and math.isfinite(lo)
        and math.isfinite(hi)
        and lo <= hi
    )


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed_rows(rows, key, expected=None):
    out = {}
    for row in rows:
        identity = row[key]
        if identity in out:
            raise ValueError(f"Duplicate {key} identity")
        out[identity] = row
    if not out or (expected is not None and set(out) != set(expected)):
        raise ValueError(f"Incomplete {key} panel")
    return out


def upsert(rows, value, keys=("metric", "panel")):
    identity = tuple(value[k] for k in keys)
    matches = [i for i, row in enumerate(rows) if tuple(row.get(k) for k in keys) == identity]
    if len(matches) > 1:
        raise ValueError("Duplicate result identity")
    if matches:
        rows[matches[0]] = value
    else:
        rows.append(value)


def validate_proposals(records, schema_path):
    """Validate the complete staged snapshot before publishing any proposal."""
    import json
    import jsonschema

    schema = json.loads(Path(schema_path).read_text())
    if not records:
        raise ValueError("Empty proposal set")
    for record in records.values():
        jsonschema.validate(record, schema)
        json.dumps(record, allow_nan=False)

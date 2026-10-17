"""Strict JSON boundary shared by registry ingestion and provenance signing.

Canonicalization is Python-json-sortkeys-ascii/1.0, not RFC 8785/JCS. Inputs
must use unique property names, string object keys, finite JSON numbers, and
Unicode scalar strings. No consumer may parse permissively before verification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

def _validate_json(value):
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str:
        value.encode("utf-8")  # Reject unpaired surrogate code points.
        return
    if type(value) is list:
        for child in value:
            _validate_json(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, child in value.items():
            _validate_json(key)
            _validate_json(child)
        return
    raise ValueError("only finite, unambiguous JSON values are supported")


def strict_json_loads(text):
    """Reject duplicate names, NaN/Infinity, overflow to infinity and surrogates."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON property: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    _validate_json(result)
    return result


def load_json(path):
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def canonical_bytes(record):
    """Python JSON v1, deliberately not advertised as RFC 8785/JCS."""
    _validate_json(record)
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

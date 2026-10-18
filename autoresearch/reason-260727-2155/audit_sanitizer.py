"""Output sanitization contract v2. Never emits arbitrary object representations."""

import os
import re

SENSITIVE = re.compile(r"password|secret|token|api.?key|authorization|credential", re.I)
TOKEN_METRIC = re.compile(r"(?:train|eval)/(?:num|response|prompt|completion|total|generated|input|output)_tokens(?:_.*)?$")
PATTERNS = [
    r"(?:wandb_v1_|tml-|hf_|sk-)[A-Za-z0-9_-]+",
    r"(?i)(?:bearer|basic)\s+[A-Za-z0-9+/=_-]+",
    r"(?i)(?:[\w.-]*(?:api.?key|token|password|secret)[\w.-]*)\s*[:=]\s*[\"']?[^\s,;\"']+",
    r"(?i)https?://[^/\s@]+@",
    r"\b[0-9a-fA-F]{40}\b",
]


def sanitize(value, credentials=()):
    secrets = tuple(credentials) + tuple(
        v for k, v in os.environ.items() if SENSITIVE.search(k) and len(v) >= 8
    )
    if isinstance(value, dict):
        return {
            sanitize(str(k), secrets): "<REDACTED>"
            if (SENSITIVE.search(str(k)) and not (TOKEN_METRIC.fullmatch(str(k))
                and isinstance(v, (int, float)) and not isinstance(v, bool)))
            else sanitize(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "<REDACTED>")
        for pattern in PATTERNS:
            value = re.sub(pattern, "<REDACTED>", value)
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "<unsupported>"

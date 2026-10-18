"""Versioned receipt checks. Caller-supplied identities never constitute trust.

Trust roots are a separately provisioned issuer->Ed25519 public-key mapping;
there are deliberately no default issuers. Legacy receipts remain historical.
"""

from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from collections.abc import Mapping
from typing import Any

IDENTITY_FIELDS = ("campaign", "phase", "suite", "run", "checkpoint", "plan")
NEGATIVE = {"rejected", "denied", "revoked", "not approved", "failed", "blocked"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def artifact_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or not digest(value.get("sha256")):
        return False
    try:
        return hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest() == value["sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        return False


def approval_valid(
    value: Any,
    kind: str,
    *,
    trusted_issuers: Mapping[str, str] | None = None,
    expected_identity: Mapping[str, str] | None = None,
) -> bool:
    """Verify signed affirmative receipt bytes, artifacts, and caller's identity.

    A missing independently supplied trust root or expected context fails closed.
    The signed payload includes all identity, decision and artifact fields.
    """
    if not isinstance(value, Mapping) or value.get("schema_version") != "pavlov-approval-v2":
        return False
    if value.get("kind") != kind or str(value.get("status", "")).strip().lower() not in {
        "approved",
        "authorized",
    }:
        return False
    if value.get("approved") is not True or value.get("authorized") is False:
        return False
    identity = value.get("identity")
    if (
        not isinstance(identity, Mapping)
        or not expected_identity
        or set(identity) != set(IDENTITY_FIELDS)
        or any(not isinstance(identity.get(k), str) or not identity[k] for k in IDENTITY_FIELDS)
        or dict(identity) != dict(expected_identity)
    ):
        return False
    if kind == "budget":
        amount = value.get("maximum_usd")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (float, int))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            return False
    artifacts = value.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or not all(artifact_valid(x) for x in artifacts)
    ):
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = (trusted_issuers or {})[value["issuer"]]
        payload = {k: v for k, v in value.items() if k != "signature_hex"}
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(
            bytes.fromhex(value["signature_hex"]), canonical(payload)
        )
        return True
    except (KeyError, TypeError, ValueError, ImportError):
        return False
    except Exception:  # invalid signature; no service calls or implicit trust fallback
        return False

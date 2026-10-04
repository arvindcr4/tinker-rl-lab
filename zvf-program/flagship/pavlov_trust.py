"""Shared loader for explicit provider-issued lane trust roots (E10, E12, E13, E14)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

try:
    from .pavlov_native_contract import canonical_json, sha256_text
except ImportError:  # pragma: no cover - direct execution fallback
    from pavlov_native_contract import canonical_json, sha256_text

_REQUIRED = {"schema_version", "lane", "suite_id", "provider", "key_id", "public_key_hex"}


def load_trust_root(
    trust_root: Mapping[str, Any] | str | Path | None,
    *,
    lane: tuple[str, str, str, str],
    error: Callable[[str], Exception],
) -> dict[str, str]:
    """Load an explicit provider-issued trust root; no embedded key is trusted.

    ``lane`` is the expected ``(schema_version, lane, suite_id, provider)``.
    Any failure is re-raised as ``error`` with the lane named in the message.
    """
    try:
        raw = (
            json.loads(Path(trust_root).read_text())
            if isinstance(trust_root, (str, Path))
            else trust_root
        )
        if (
            not isinstance(raw, Mapping)
            or set(raw) != _REQUIRED
            or not all(isinstance(raw[key], str) for key in _REQUIRED)
        ):
            raise ValueError("trust-root schema")
        root = {key: str(value) for key, value in raw.items()}
        if (
            (root["schema_version"], root["lane"], root["suite_id"], root["provider"]) != lane
            or not root["key_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", root["public_key_hex"])
        ):
            raise ValueError("trust-root identity")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(root["public_key_hex"]))
        document = canonical_json(dict(raw))
        root["document_sha256"] = sha256_text(document)
        root["key_fingerprint"] = hashlib.sha256(bytes.fromhex(root["public_key_hex"])).hexdigest()
        return root
    except Exception as exc:
        raise error(f"explicit valid {lane[1]} provider trust root is required") from exc

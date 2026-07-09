#!/usr/bin/env python3
"""Tamper-evident signing for the P5 provenance protocol (SPEC Phase-2).

Provides detached signatures over provenance-record JSON so that any
post-hoc mutation of a record is detectable.

Signing backend (best -> fallback):
  1. ed25519 via the `cryptography` library  -> a real public-key signature.
  2. HMAC-SHA256                              -> a symmetric MAC. NOTE: a MAC is
     NOT a public-key signature; anyone holding the secret key can forge it. It
     still gives tamper-evidence against parties without the key, but it does
     not provide non-repudiation. The .sig file records which backend was used.

Canonicalization: records are serialized with json.dumps(sort_keys=True,
separators=(",",":")) so that signing/verification are insensitive to key
ordering and whitespace but sensitive to every value.

CLI:
  python sign.py keygen               # create keypair under ./keys/
  python sign.py sign   <record.json> # write detached <record.json>.sig
  python sign.py verify <record.json> # verify against <record.json>.sig
  python sign.py demo   [<record>]    # sign+verify+tamper demonstration
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY_DIR = HERE / "keys"
PRIV_ED = KEY_DIR / "provenance_ed25519.key"
PUB_ED = KEY_DIR / "provenance_ed25519.pub"
HMAC_KEY = KEY_DIR / "provenance_hmac.key"
DEFAULT_RECORD = HERE / "campaign.provenance.json"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization

    HAVE_ED25519 = True
except Exception:  # pragma: no cover - exercised only when lib missing
    HAVE_ED25519 = False

BACKEND = "ed25519" if HAVE_ED25519 else "hmac-sha256"


# ---------------------------------------------------------------------------
# Canonicalization / digest
# ---------------------------------------------------------------------------
def canonical_bytes(record: dict) -> bytes:
    """Deterministic byte serialization of a record."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_hex(record: dict) -> str:
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------
def keygen(force: bool = False) -> None:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if HAVE_ED25519:
        if PRIV_ED.exists() and not force:
            print(f"[keygen] ed25519 key already exists: {PRIV_ED} (use force to overwrite)")
            return
        priv = Ed25519PrivateKey.generate()
        priv_bytes = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        PRIV_ED.write_bytes(base64.b64encode(priv_bytes))
        PUB_ED.write_bytes(base64.b64encode(pub_bytes))
        os.chmod(PRIV_ED, 0o600)
        print(f"[keygen] backend=ed25519")
        print(f"[keygen] private key -> {PRIV_ED} (mode 600)")
        print(f"[keygen] public  key -> {PUB_ED}")
    else:
        if HMAC_KEY.exists() and not force:
            print(f"[keygen] hmac key already exists: {HMAC_KEY}")
            return
        secret = os.urandom(32)
        HMAC_KEY.write_bytes(base64.b64encode(secret))
        os.chmod(HMAC_KEY, 0o600)
        print("[keygen] backend=hmac-sha256 (MAC, not a public-key signature)")
        print(f"[keygen] secret key -> {HMAC_KEY} (mode 600)")


def _load_ed_private() -> "Ed25519PrivateKey":
    raw = base64.b64decode(PRIV_ED.read_bytes())
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_ed_public() -> "Ed25519PublicKey":
    raw = base64.b64decode(PUB_ED.read_bytes())
    return Ed25519PublicKey.from_public_bytes(raw)


def _load_hmac_secret() -> bytes:
    return base64.b64decode(HMAC_KEY.read_bytes())


def _ensure_keys() -> None:
    if HAVE_ED25519:
        if not PRIV_ED.exists():
            keygen()
    else:
        if not HMAC_KEY.exists():
            keygen()


# ---------------------------------------------------------------------------
# Core sign / verify on an in-memory record
# ---------------------------------------------------------------------------
def sign_record(record: dict) -> dict:
    """Return a detached signature envelope for a record dict."""
    _ensure_keys()
    msg = canonical_bytes(record)
    if HAVE_ED25519:
        sig = _load_ed_private().sign(msg)
        sig_b64 = base64.b64encode(sig).decode("ascii")
    else:
        sig = hmac.new(_load_hmac_secret(), msg, hashlib.sha256).digest()
        sig_b64 = base64.b64encode(sig).decode("ascii")
    return {
        "spec": "P5-PROVENANCE-SIG/1.0",
        "backend": BACKEND,
        "is_public_key_signature": HAVE_ED25519,
        "digest_sha256": hashlib.sha256(msg).hexdigest(),
        "signature_b64": sig_b64,
        "caveat": None
        if HAVE_ED25519
        else "HMAC-SHA256 is a symmetric MAC, not a public-key signature; "
        "holders of the secret key can forge it (no non-repudiation).",
    }


def verify_record(record: dict, envelope: dict) -> bool:
    """Verify an in-memory record against a signature envelope."""
    msg = canonical_bytes(record)
    # Digest guard first (fast, and catches the common tamper case).
    if envelope.get("digest_sha256") != hashlib.sha256(msg).hexdigest():
        return False
    sig = base64.b64decode(envelope["signature_b64"])
    backend = envelope.get("backend", BACKEND)
    if backend == "ed25519":
        if not HAVE_ED25519:
            raise RuntimeError("ed25519 signature present but cryptography lib unavailable")
        try:
            _load_ed_public().verify(sig, msg)
            return True
        except Exception:
            return False
    elif backend == "hmac-sha256":
        expected = hmac.new(_load_hmac_secret(), msg, hashlib.sha256).digest()
        return hmac.compare_digest(expected, sig)
    else:
        raise ValueError(f"unknown backend: {backend}")


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------
def sig_path(record_path: Path) -> Path:
    return record_path.with_suffix(record_path.suffix + ".sig")


def sign_file(record_path: Path) -> Path:
    record = json.loads(record_path.read_text())
    envelope = sign_record(record)
    out = sig_path(record_path)
    out.write_text(json.dumps(envelope, indent=2) + "\n")
    return out


def verify_file(record_path: Path) -> bool:
    record = json.loads(record_path.read_text())
    envelope = json.loads(sig_path(record_path).read_text())
    return verify_record(record, envelope)


# ---------------------------------------------------------------------------
# Demonstration of tamper-evidence
# ---------------------------------------------------------------------------
def demo(record_path: Path) -> int:
    print("=" * 68)
    print("P5 PROVENANCE TAMPER-EVIDENCE DEMONSTRATION")
    print("=" * 68)
    print(f"backend            : {BACKEND}")
    print(f"public-key sig     : {HAVE_ED25519}")
    print(f"record             : {record_path}")
    _ensure_keys()

    record = json.loads(record_path.read_text())

    # 1) Sign
    out = sign_file(record_path)
    print(f"\n[1] SIGNED  -> {out}   (signed OK)")

    # 2) Verify pristine
    ok = verify_file(record_path)
    print(f"[2] VERIFY  original record -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    unexpected: pristine record failed verification")
        return 1

    # 3) Tamper in memory and re-verify against the same signature
    envelope = json.loads(sig_path(record_path).read_text())
    tampered = json.loads(json.dumps(record))  # deep copy
    # mutate one meaningful field
    field_before = tampered["provenance"]["git_sha"]
    tampered["provenance"]["git_sha"] = "deadbeef"  # forged commit
    print(f"\n[3] TAMPER  provenance.git_sha: {field_before!r} -> 'deadbeef' (in memory)")
    ok_tampered = verify_record(tampered, envelope)
    print(f"    VERIFY  tampered record -> {'PASS' if ok_tampered else 'FAIL'}")

    print("\n" + "-" * 68)
    if ok and not ok_tampered:
        print("RESULT: signed OK, original verify PASS, tamper -> FAIL  ✓")
        return 0
    print("RESULT: tamper-evidence check did NOT behave as expected  ✗")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _usage() -> int:
    print(__doc__)
    return 2


def main(argv: list[str]) -> int:
    if not argv:
        return _usage()
    cmd = argv[0]
    if cmd == "keygen":
        keygen(force="--force" in argv or "force" in argv[1:])
        return 0
    if cmd == "sign":
        if len(argv) < 2:
            print("usage: sign.py sign <record.json>")
            return 2
        out = sign_file(Path(argv[1]))
        print(f"signed OK -> {out}")
        return 0
    if cmd == "verify":
        if len(argv) < 2:
            print("usage: sign.py verify <record.json>")
            return 2
        ok = verify_file(Path(argv[1]))
        print(f"verify -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if cmd == "demo":
        rec = Path(argv[1]) if len(argv) > 1 else DEFAULT_RECORD
        return demo(rec)
    return _usage()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

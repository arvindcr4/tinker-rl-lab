#!/usr/bin/env python3
"""Versioned, fail-closed Ed25519 provenance signatures.

No automatic key creation, algorithm fallback, embedded-key trust, or key rotation.
Use ``keygen --key-dir NEW_DIRECTORY`` to provision a distinct signing identity.
The signing primitive authenticates JSON; publication additionally requires the
schema and evidence checks in ``minreport.audit_file``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
KEY_DIR = HERE / "keys"
PRIV_ED = KEY_DIR / "provenance_ed25519.key"
PUB_ED = KEY_DIR / "provenance_ed25519.pub"
PROTOCOL = "P5-PROVENANCE-SIG/2.0"
CANONICALIZATION = "python-json-sortkeys-ascii/1.0"
BACKEND = "ed25519"
DOMAIN = b"P5-PROVENANCE-SIG/2.0\x00"
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
    HAVE_ED25519 = True
except ImportError:  # Never downgrade to a MAC when the dependency is absent.
    HAVE_ED25519 = False


try:
    from ..json_protocol import canonical_bytes, load_json, strict_json_loads as strict_json_loads
except ImportError:  # Direct script invocation.
    sys.path.insert(0, str(HERE.parent))
    from json_protocol import canonical_bytes, load_json, strict_json_loads as strict_json_loads


def _require_ed25519():
    if not HAVE_ED25519:
        raise RuntimeError("Ed25519 requires cryptography; no weaker fallback is permitted")


def _paths(key_dir=None):
    if key_dir is None:
        return KEY_DIR, PRIV_ED, PUB_ED
    directory = Path(key_dir)
    return directory, directory / PRIV_ED.name, directory / PUB_ED.name


def _exclusive_write(path, data, mode):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def key_fingerprint(raw_public_key):
    return "sha256:" + hashlib.sha256(raw_public_key).hexdigest()


def keygen(force=False, key_dir=None):
    """Create a new identity only; even force cannot replace a trust anchor."""
    _require_ed25519()
    if force:
        raise ValueError("in-place rotation is forbidden; choose a new --key-dir")
    directory, private_path, public_path = _paths(key_dir)
    if any(os.path.lexists(path) for path in (private_path, public_path)):
        raise FileExistsError("key generation refuses to replace either existing key")
    if directory.is_symlink():
        raise ValueError("key directory must not be a symlink")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise PermissionError("key directory must have mode 0700 before provisioning")
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(serialization.Encoding.Raw,
                                        serialization.PrivateFormat.Raw,
                                        serialization.NoEncryption())
    public_raw = private.public_key().public_bytes(serialization.Encoding.Raw,
                                                   serialization.PublicFormat.Raw)
    fingerprint = key_fingerprint(public_raw)
    archive = directory / (fingerprint.replace(":", "-") + ".pub")
    # Exclusive opens protect against racing writers and existing symlinks. A
    # failure leaves any newly created files intact for explicit operator review.
    _exclusive_write(private_path, base64.b64encode(private_raw), 0o600)
    _exclusive_write(public_path, base64.b64encode(public_raw), 0o644)
    _exclusive_write(archive, base64.b64encode(public_raw), 0o644)
    return fingerprint


def _decode_key(path):
    return base64.b64decode(Path(path).read_bytes(), validate=True)


def _load_ed_private(key_dir=None):
    _require_ed25519()
    return Ed25519PrivateKey.from_private_bytes(_decode_key(_paths(key_dir)[1]))


def _load_ed_public(trusted_key):
    _require_ed25519()
    raw = trusted_key if isinstance(trusted_key, bytes) else _decode_key(trusted_key)
    return Ed25519PublicKey.from_public_bytes(raw), raw


def _ensure_keys(key_dir=None):
    """Validate an existing pair; never generate or alter keys while signing."""
    directory, private_path, public_path = _paths(key_dir)
    if not private_path.is_file() or not public_path.is_file():
        raise FileNotFoundError("both signing keys must exist; provision a new identity explicitly")
    if directory.is_symlink() or private_path.is_symlink() or public_path.is_symlink():
        raise ValueError("signing keys and directory must not be symlinks")
    if directory.stat().st_mode & 0o077 or private_path.stat().st_mode & 0o077:
        raise PermissionError("signing requires directory 0700 and private key 0600")
    private = _load_ed_private(key_dir)
    derived = private.public_key().public_bytes(serialization.Encoding.Raw,
                                                serialization.PublicFormat.Raw)
    if derived != _decode_key(public_path):
        raise ValueError("private key does not match the configured public trust anchor")
    return private, derived


def _payload(record, protected):
    return DOMAIN + canonical_bytes({"protected": protected, "record": record})


def sign_record(record, key_dir=None):
    """Low-level JSON authentication, not evidence validation or publication."""
    if type(record) is not dict:
        raise ValueError("a signed record must be a JSON object")
    message = canonical_bytes(record)
    private, public = _ensure_keys(key_dir)
    protected = {
        "spec": PROTOCOL,
        "canonicalization": CANONICALIZATION,
        "algorithm": BACKEND,
        "key_id": key_fingerprint(public),
        "digest_sha256": hashlib.sha256(message).hexdigest(),
    }
    return {"protected": protected,
            "signature_b64": base64.b64encode(private.sign(_payload(record, protected))).decode("ascii")}


def verify_record(record, envelope, trusted_key=None, allowed_algorithms=("ed25519",)):
    """Verify against caller-supplied trust; the envelope never selects trust."""
    try:
        if trusted_key is None or type(record) is not dict or type(envelope) is not dict:
            return False
        if set(envelope) != {"protected", "signature_b64"}:
            return False
        header = envelope["protected"]
        if type(header) is not dict or set(header) != {
            "spec", "canonicalization", "algorithm", "key_id", "digest_sha256"
        }:
            return False
        if (header["spec"] != PROTOCOL or header["canonicalization"] != CANONICALIZATION
                or header["algorithm"] != "ed25519"
                or header["algorithm"] not in allowed_algorithms):
            return False
        public, raw = _load_ed_public(trusted_key)
        if header["key_id"] != key_fingerprint(raw):
            return False
        if header["digest_sha256"] != hashlib.sha256(canonical_bytes(record)).hexdigest():
            return False
        signature = base64.b64decode(envelope["signature_b64"], validate=True)
        public.verify(signature, _payload(record, header))
        return True
    except Exception:
        return False


def sig_path(record_path):
    path = Path(record_path)
    return path.with_name(path.name + ".sig")


def sign_file(record_path, key_dir=None):
    """Sign only records satisfying the publication evidence policy."""
    try:
        from . import minreport
    except ImportError:  # Direct script invocation.
        import minreport
    record = load_json(record_path)
    report = minreport.verify(record)
    if not report["evidence_valid"]:
        failed = [check["msg"] for check in report["checks"] if check["level"] != "PASS"]
        raise ValueError("record is not publication-ready: " + "; ".join(failed))
    envelope = sign_record(record, key_dir)
    out = sig_path(record_path)
    _exclusive_write(out, (json.dumps(envelope, indent=2, allow_nan=False) + "\n").encode(), 0o644)
    return out


def verify_file(record_path, trusted_key=None, allowed_algorithms=("ed25519",)):
    try:
        return verify_record(load_json(record_path), load_json(sig_path(record_path)),
                             trusted_key, allowed_algorithms)
    except (ValueError, OSError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    key_parser = commands.add_parser("keygen")
    key_parser.add_argument("--key-dir", type=Path)
    sign_parser = commands.add_parser("sign")
    sign_parser.add_argument("record", type=Path)
    sign_parser.add_argument("--key-dir", type=Path)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("record", type=Path)
    verify_parser.add_argument("--trusted-key", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "keygen":
            print(f"created identity {keygen(key_dir=args.key_dir)}")
            return 0
        if args.command == "sign":
            print(f"signed -> {sign_file(args.record, args.key_dir)}")
            return 0
        ok = verify_file(args.record, args.trusted_key)
        print("signature -> " + ("PASS (Ed25519; evidence not audited)" if ok else "FAIL"))
        return 0 if ok else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

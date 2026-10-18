"""Exclusive durable local claims; an ambiguous dispatch is never retried implicitly."""

import json
import os
from pathlib import Path
import uuid


def exclusive_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def claim(directory: Path, identity: dict) -> str:
    nonce = uuid.uuid4().hex
    exclusive_json(
        directory / "attempt.json",
        {"version": 2, "nonce": nonce, "state": "PREPARED", "identity": identity},
    )
    return nonce


def transition(directory: Path, nonce: str, state: str, **evidence):
    exclusive_json(
        directory / f"{state.lower()}.json", {"nonce": nonce, "state": state, **evidence}
    )

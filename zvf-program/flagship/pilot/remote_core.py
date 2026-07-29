from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import PilotProtocol, ProtocolError, sha256_file


GSM8K_INTEGER = re.compile(r"[-+]?\d[\d,]*")
SOURCE_FILES = (
    "zvf-program/flagship/pilot_preregistration.json",
    "zvf-program/flagship/pilot/protocol.py",
    "zvf-program/flagship/pilot/replay.py",
    "zvf-program/flagship/pilot/objective.py",
    "zvf-program/flagship/pilot/training.py",
    "zvf-program/flagship/pilot/artifacts.py",
    "zvf-program/flagship/pilot/remote_core.py",
    "zvf-program/flagship/pilot/flops.py",
    "zvf-program/flagship/pilot/checkpointing.py",
    "zvf-program/flagship/pilot/evaluation.py",
    "zvf-program/flagship/pilot/remote_unit.py",
    "zvf-program/flagship/pilot/remote_training.py",
    "zvf-program/flagship/pilot/runtime_install.py",
    "zvf-program/flagship/pilot/bootstrap.py",
    "zvf-program/flagship/pilot/provenance/r3-corpus-bindings.json",
    "zvf-program/flagship/pilot/provenance/r3-corpus-source.tar.gz",
    "zvf-program/flagship/pilot/provenance/r3-control-source.tar.gz",
)


class RemoteContractError(RuntimeError):
    """The remote runtime or dataset violates the frozen pilot contract."""


def parse_gsm8k_integer(text: str) -> int | None:
    if "####" not in text:
        return None
    tail = text.rsplit("####", 1)[-1]
    match = GSM8K_INTEGER.search(tail)
    if match is None:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def gsm8k_reward(response: str, answer: str) -> float:
    prediction = parse_gsm8k_integer(response)
    target = parse_gsm8k_integer(answer)
    return float(prediction is not None and target is not None and prediction == target)


def last_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed{")
    if index == -1:
        return None
    cursor, depth, output = index + 7, 1, []
    while cursor < len(text) and depth:
        character = text[cursor]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if not depth:
                break
        output.append(character)
        cursor += 1
    return "".join(output) if depth == 0 else None


def normalize_math(answer: str) -> str:
    value = answer.strip().replace("\\left", "").replace("\\right", "")
    for junk in ("\\!", "\\,", "\\;", " "):
        value = value.replace(junk, "")
    value = value.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value


def math500_reward(response: str, answer: str) -> float:
    prediction = last_boxed(response)
    return float(prediction is not None and normalize_math(prediction) == normalize_math(answer))


def prompt_messages(regime: str, question: str) -> list[dict[str, str]]:
    if regime == "balanced_equal_length":
        instruction = (
            "Solve the problem carefully. End with exactly one line of the form #### <integer>."
        )
    elif regime == "filtered_variable_length":
        instruction = (
            "Solve the competition mathematics problem carefully. Put the final answer "
            "inside exactly one \\boxed{...}."
        )
    else:
        raise RemoteContractError(f"unknown pilot regime: {regime}")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def canonical_order_hash(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    keys: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for position, index in enumerate(indices):
        row = rows[index]
        payload = {
            "position": position,
            "index": int(index),
            **{key: row[key] for key in keys},
        }
        digest.update(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def frozen_train_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligible_indices: Sequence[int],
    seed: int,
    keys: Sequence[str],
    count: int = 100,
) -> tuple[tuple[int, ...], str]:
    if len(eligible_indices) < count:
        raise RemoteContractError("eligible training partition is too small")
    generator = np.random.Generator(np.random.PCG64(seed))
    order = tuple(
        int(index)
        for index in generator.permutation(np.asarray(eligible_indices, dtype=np.int64))[:count]
    )
    return order, canonical_order_hash(rows, order, keys=keys)


def verified_train_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
) -> tuple[int, ...]:
    if regime == "balanced_equal_length":
        eligible = tuple(range(len(rows)))
        keys = ("question", "answer")
    elif regime == "filtered_variable_length":
        eligible = tuple(range(128, len(rows)))
        keys = ("unique_id", "problem", "answer")
    else:
        raise RemoteContractError(f"unknown pilot regime: {regime}")
    order, digest = frozen_train_order(
        rows,
        eligible_indices=eligible,
        seed=seed,
        keys=keys,
    )
    expected = protocol.payload["runtime"]["execution_contract"]["train_order_hash"][regime][
        str(seed)
    ]
    if digest != expected:
        raise RemoteContractError(
            f"training row-order hash mismatch for {regime} seed {seed}: "
            f"expected {expected}, got {digest}"
        )
    return order


def expected_runtime_versions(protocol: PilotProtocol) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pin in protocol.payload["runtime"]["package_pins"]:
        name, separator, version = pin.partition("==")
        if not separator or not name or not version:
            raise ProtocolError(f"runtime package is not exactly pinned: {pin}")
        versions[name] = version
    return versions


def verify_runtime_versions(protocol: PilotProtocol) -> dict[str, str]:
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        raise RemoteContractError(
            f"Python runtime mismatch: expected >=3.11,<3.13, got {sys.version.split()[0]}"
        )
    expected = expected_runtime_versions(protocol)
    actual: dict[str, str] = {}
    for name, version in expected.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RemoteContractError(f"required runtime package is missing: {name}") from exc
        if actual[name] != version:
            raise RemoteContractError(
                f"runtime version mismatch for {name}: expected {version}, got {actual[name]}"
            )
    if "numpy" in expected:
        try:
            from numpy._core.umath import _center  # noqa: F401
        except (ImportError, AttributeError) as exc:
            raise RemoteContractError(
                "numpy binary extension is stale or inconsistent with the installed "
                "files; restart the kernel after installation before running science"
            ) from exc
    return {"python": sys.version.split()[0], **actual}


def require_a100(torch_module: Any) -> str:
    if not torch_module.cuda.is_available():
        raise RemoteContractError("pilot requires CUDA but no CUDA device is available")
    name = str(torch_module.cuda.get_device_name(0))
    if "A100" not in name.upper():
        raise RemoteContractError(f"pilot requires A100; detected {name}")
    if not torch_module.cuda.is_bf16_supported():
        raise RemoteContractError("pilot requires A100 bfloat16 support")
    return name


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=False)


def source_manifest(repo_root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise RemoteContractError(f"required pilot source is missing: {relative}")
        manifest[relative] = sha256_file(path)
    return manifest

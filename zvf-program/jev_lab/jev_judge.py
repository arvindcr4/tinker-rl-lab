#!/usr/bin/env python3
"""Receipt-recording wrapper around the `jev` (TypeSafe AI) CLI.

Design contract (mirrors the repo's evidence culture):

- Every judgment writes a durable JSON receipt before its value is returned.
- Receipts bind the exact state bytes (inline + SHA-256), the exact question
  text, the raw CLI answer, and the model id.
- Arithmetic, hashing, counting and lookups stay in code.  Jev is used only
  for semantic judgments: classification, faithfulness, plausibility,
  rubric scoring, decision stress-testing.
- The wrapper fails closed: any non-JSON or non-zero CLI result raises
  and writes no receipt.

CLI:
    python3 zvf-program/jev_lab/jev_judge.py demo

Library:
    from jev_judge import run_ask, run_noul, run_score, run_choice
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT_DIR = REPO_ROOT / "outputs/jev_receipts"
SCHEMA_VERSION = "jev-judgment-receipt-v1"
DEFAULT_MODEL = "jev-latest"
MAX_INLINE_STATE_BYTES = 120_000

_JEV = os.environ.get("JEV_BIN", "jev")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _call_jev(args: list[str]) -> dict[str, Any]:
    """Run one jev CLI subcommand and parse its JSON stdout."""
    proc = subprocess.run(
        [_JEV, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"jev {' '.join(args[:1])} exited {proc.returncode}: {proc.stderr[-2000:]}"
        )
    # The CLI occasionally prints config noise before the JSON object;
    # parse from the first '{' to the matching end.
    out = proc.stdout.strip()
    start = out.find("{")
    if start < 0:
        raise RuntimeError(f"jev produced no JSON: {out[:500]}")
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"jev produced unparseable JSON: {out[:500]}") from exc


def _receipt_path(label: str, receipt_dir: Path) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    slug = label.replace(" ", "_").replace("/", "-") or "judgment"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = receipt_dir / f"{stamp}__{slug}.json"
    n = 1
    while path.exists():
        path = receipt_dir / f"{stamp}__{slug}__{n}.json"
        n += 1
    return path


def _state_bytes(state: Any) -> bytes:
    if isinstance(state, (str, bytes)):
        return state if isinstance(state, bytes) else state.encode()
    return json.dumps(state, indent=1, sort_keys=True).encode()


def _record(
    *,
    kind: str,
    label: str,
    state: Any,
    payload: dict[str, Any],
    answer: dict[str, Any],
    receipt_dir: Path,
) -> dict[str, Any]:
    raw = _state_bytes(state)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _utc_now(),
        "model": DEFAULT_MODEL,
        "kind": kind,
        "label": label,
        "state_sha256": _sha256_bytes(raw),
        "state_inline": (
            raw.decode("utf-8", "replace") if len(raw) <= MAX_INLINE_STATE_BYTES else None
        ),
        "state_truncated": len(raw) > MAX_INLINE_STATE_BYTES,
        "input": payload,
        "answer": answer,
    }
    path = _receipt_path(label, receipt_dir)
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    receipt["receipt_path"] = str(path)
    return receipt


def _state_file(state: Any, tmp_dir: Path) -> Path:
    raw = _state_bytes(state)
    p = tmp_dir / "state.json"
    p.write_bytes(raw)
    return p


def run_ask(
    state: Any,
    questions: dict[str, dict[str, Any]],
    label: str,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict[str, Any]:
    """Ask independent mixed questions over one shared state.

    questions: {"name": {"type": "noul"|"choice"|"score",
                          "instructions": "...", ["choices": [...]]}}
    Returns the full receipt; answers are in receipt["answer"]["answers"].
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        state_path = _state_file(state, Path(td))
        answer = _call_jev(["ask", str(state_path), "-q", json.dumps(questions)])
    receipt = _record(
        kind="ask",
        label=label,
        state=state,
        payload={"questions": questions},
        answer=answer,
        receipt_dir=receipt_dir,
    )
    return receipt


def run_noul(
    state: Any,
    instructions: str,
    label: str,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict[str, Any]:
    """Probability that a yes-condition holds, given state."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        state_path = _state_file(state, Path(td))
        answer = _call_jev(["noul", str(state_path), instructions])
    return _record(
        kind="noul", label=label, state=state,
        payload={"instructions": instructions}, answer=answer, receipt_dir=receipt_dir,
    )


def run_score(
    state: Any,
    instructions: str,
    label: str,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        state_path = _state_file(state, Path(td))
        answer = _call_jev(["score", str(state_path), instructions])
    return _record(
        kind="score", label=label, state=state,
        payload={"instructions": instructions}, answer=answer, receipt_dir=receipt_dir,
    )


def run_choice(
    state: Any,
    instructions: str,
    choices: list[str],
    label: str,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        state_path = _state_file(state, Path(td))
        answer = _call_jev(
            ["choice", str(state_path), instructions, "--choices", ",".join(choices)]
        )
    return _record(
        kind="choice", label=label, state=state,
        payload={"instructions": instructions, "choices": choices},
        answer=answer, receipt_dir=receipt_dir,
    )


def healthcheck(model: str | None = None) -> dict[str, Any]:
    """Two known-answer probes; healthy only if both land on the right side.

    Returns {"healthy": bool, "yes_probe": float, "no_probe": float}.
    Observed 2026-09-19: service can degrade to ~0.3-0.5 on both probes
    (state apparently not read); batteries must not run in that state.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p_yes = Path(td) / "yes.json"
        p_yes.write_text(json.dumps({"fruit": "apple"}))
        p_no = Path(td) / "no.json"
        p_no.write_text(json.dumps({"fruit": "apple"}))
    flag = ["--model", model] if model else []
    a1 = _call_jev([*flag, "noul", str(p_yes), "Does fruit say apple?"])
    a2 = _call_jev([*flag, "noul", str(p_no), "Does fruit say banana?"])
    yes_probe = float(a1.get("noul", 0.0))
    no_probe = float(a2.get("noul", 1.0))
    return {
        "healthy": yes_probe >= 0.7 and no_probe <= 0.3,
        "yes_probe": yes_probe,
        "no_probe": no_probe,
        "checked_at": _utc_now(),
    }


def _demo() -> None:
    state = {
        "experiment": "E11 VerilogEval",
        "canonical_score": "129/312 = 41.35% pass@1",
        "code_completion": "67/156",
        "spec_to_rtl": "62/156",
    }
    r = run_ask(
        state,
        {
            "claims_consistent": {
                "type": "noul",
                "instructions": (
                    "Do the component counts 67/156 and 62/156 sum to the "
                    "reported total 129/312 as stated?"
                ),
            },
            "summary_faithful": {
                "type": "choice",
                "instructions": (
                    "Which summary best describes the evidence quality for the "
                    "canonical score claim in this state?"
                ),
                "choices": [
                    "component_breakdown_present",
                    "total_only_no_breakdown",
                    "contradictory_numbers",
                ],
            },
        },
        label="demo_selftest",
    )
    print(json.dumps(r["answer"], indent=1))
    print("receipt:", r["receipt_path"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        _demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "health":
        print(json.dumps(healthcheck(sys.argv[2] if len(sys.argv) > 2 else None), indent=1))
    else:
        print(__doc__)

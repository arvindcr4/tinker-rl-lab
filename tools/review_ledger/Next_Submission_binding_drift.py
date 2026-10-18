"""Check Next_Submission preregistration bindings against working tree and git history.

Usage: python3 tools/review_ledger/Next_Submission_binding_drift.py  (run from repo root)
Writes platform_hybrid/experiments/results/review_2026-10-18/Next_Submission/binding_drift.json
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TypedDict


class FileBinding(TypedDict):
    bound: str | None
    working_tree: str | None
    match: bool
    bound_version_commit: str | None


class BindingReport(TypedDict):
    preregistration_sha256: str
    files: dict[str, FileBinding]


root = Path(".")
prereg = json.loads((root / "zvf-program/next-submission/preregistration.json").read_text())
b = prereg["bindings"]
out: BindingReport = {
    "preregistration_sha256": hashlib.sha256(
        (root / "zvf-program/next-submission/preregistration.json").read_bytes()
    ).hexdigest(),
    "files": {},
}
for key, val in b.items():
    if not key.endswith("_path"):
        continue
    stem = key[: -len("_path")]
    want = b.get(stem + "_sha256")
    p = root / val
    have = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    hist = None
    if want and have != want:
        commits = subprocess.run(
            ["git", "log", "--format=%H", "-40", "--", val], capture_output=True, text=True
        ).stdout.split()
        for c in commits:
            blob = subprocess.run(["git", "show", f"{c}:{val}"], capture_output=True).stdout
            if hashlib.sha256(blob).hexdigest() == want:
                hist = c
                break
    out["files"][val] = {
        "bound": want,
        "working_tree": have,
        "match": have == want,
        "bound_version_commit": hist,
    }
dst = root / "platform_hybrid/experiments/results/review_2026-10-18/Next_Submission"
dst.mkdir(parents=True, exist_ok=True)
(dst / "binding_drift.json").write_text(json.dumps(out, indent=1))
for k, v in out["files"].items():
    if not v["match"]:
        print(
            "DRIFT",
            k,
            (v["bound"] or "")[:8],
            (v["working_tree"] or "")[:8],
            v["bound_version_commit"],
        )
print("prereg", out["preregistration_sha256"][:8])

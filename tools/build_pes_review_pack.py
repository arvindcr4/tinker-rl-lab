"""Assemble the explicitly selected PES review documents and offline demo."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "reports/project_finish_2026-10-11"


def record(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def validate_audit(audit, snapshot):
    """A PASS is usable only with v2 claim-level and byte-level bindings."""
    if (
        audit.get("schema_version") != "thesis-number-provenance-v2"
        or audit.get("status") != "PASS"
    ):
        raise ValueError("A structurally bound v2 thesis audit must pass before packaging")
    groups = audit.get("groups")
    bindings = audit.get("source_bindings")
    if not isinstance(groups, list) or not groups or not isinstance(bindings, list) or not bindings:
        raise ValueError("audit lacks source/claim bindings")
    bound = {}
    for row in bindings:
        path = row["path"]
        if path in bound or path not in snapshot:
            raise ValueError(f"duplicate or missing audit source: {path}")
        bound[path] = row["sha256"]
        if hashlib.sha256(snapshot[path]).hexdigest() != row["sha256"]:
            raise ValueError(f"stale audit source: {path}")
    for group in groups:
        chapter = str(Path(audit["thesis_directory"]) / group["chapter"])
        if chapter not in bound or group.get("status") != "PASS" or not group.get("tokens"):
            raise ValueError("audit lacks a passing chapter binding")
        for token in group["tokens"]:
            if (
                token.get("result") not in ("PASS", "CONTEXT")
                or token.get("result") == "PASS"
                and token.get("manuscript_binding") != "matched"
            ):
                raise ValueError("audit has an unbound manuscript claim")
        for artifact in group["artifacts"]:
            path = artifact["path"]
            if any(c in path for c in "*?["):
                import fnmatch

                matches = sorted(p for p in bound if fnmatch.fnmatch(p, path))
                digest = hashlib.sha256(
                    "".join(f"{bound[p]}  {p}\n" for p in matches).encode()
                ).hexdigest()
                if not matches or digest != artifact["sha256"]:
                    raise ValueError("audit artifact glob binding mismatch")
            elif bound.get(path) != artifact["sha256"]:
                raise ValueError("audit artifact binding mismatch")


def publish_archive(target, snapshot, modes=None):
    """Write and verify a stable byte snapshot before atomic publication."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".review-pack-", suffix=".zip", dir=target.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(snapshot.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 11, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (modes or {}).get(name, 0o100644) << 16
                archive.writestr(info, data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or any(
                archive.read(name) != data for name, data in snapshot.items()
            ):
                raise ValueError("archive byte verification failed")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def current_thesis_paths(pack: Path) -> tuple[Path, Path]:
    """Resolve the current PDF and its separate review receipt from one contract."""
    config = json.loads((pack / "thesis/current_artifact.json").read_text())
    return pack / "thesis" / config["pdf"], pack / config["pdf_check_receipt"]


def main() -> int:
    audit = json.loads((PACK / "thesis/provenance_audit.json").read_text())
    if audit.get("schema_version") != "thesis-number-provenance-v2" or audit["status"] != "PASS":
        raise ValueError(
            "The legacy PES pack requires a current, structurally bound v2 number audit; "
            "the preserved historical PASS does not authorize packaging this revision. "
            "See thesis/README.md for the current standalone build."
        )
    pdf, pdf_review = current_thesis_paths(PACK)
    quality = json.loads(pdf_review.read_text())
    if record(pdf)["sha256"] != quality["sha256"] or quality["visual_review"] != "PASS":
        raise ValueError("The PDF must match its completed scoped visual review")
    deck = PACK / "defense/TinkerRL_PES_Defense_2026-10-11.pptx"
    deck_quality = json.loads((PACK / "validation/deck_checks.json").read_text())
    if record(deck)["sha256"] != deck_quality["sha256"] or deck_quality["visual_review"] != "PASS":
        raise ValueError("The deck must match its completed scoped visual review")

    selected = set(PACK.glob("*.md")) | {
        PACK / "decisions.tsv",
        PACK / "verify_review_pack.py",
        PACK / "check_pdf.py",
    }
    selected.update((PACK / "defense").glob("*"))
    selected.update(
        path
        for path in (PACK / "colab_validation").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in (PACK / "thesis").rglob("*"):
        if (
            path.is_file()
            and "build" not in path.relative_to(PACK).parts
            and path.name != "thesis_master.tex"
        ):
            selected.add(path)
    selected.add(pdf_review)
    for name in ("SUMMARY.md", "deck_checks.json", "delivery_checks.json"):
        selected.add(PACK / "validation" / name)
    # same-stack v2 analysis (thesis §6.5.5, Table 6.2) and the per-run records it reads
    selected.update(path for path in (PACK / "validation/samestack_v2").iterdir() if path.is_file())
    selected.update(
        path
        for path in (ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s").iterdir()
        if path.is_file()
    )
    selected.update(
        {
            ROOT / "PROJECT_HISTORY.md",
            ROOT / "SUBMISSION.md",
            ROOT / "LICENSE",
            ROOT / "tools/build_pes_review_pack.py",
            ROOT / "tools/audit_thesis_numbers.py",
            ROOT / "platform_hybrid/registry/query.py",
            ROOT / "platform_hybrid/registry/schema.json",
            ROOT / "platform_hybrid/registry/entries/colab-open_dapo_e3.json",
            ROOT / "platform_hybrid/registry/entries/tinker_dapo_qwen3.5-4b_gsm8k.json",
        }
    )
    for name in (
        "README.md",
        "DEFENSE_RUNBOOK.md",
        "demo.sh",
        "run_demo.py",
        "fixtures/offline_demo.json",
        "tests/test_demo.py",
    ):
        selected.add(ROOT / "submission/demo" / name)
    for name in (
        "tinker_direct_eval.json",
        "samestack_gsm8k_cot.json",
        "samestack_gsm8k_cot_full.json",
    ):
        selected.add(ROOT / "platform_hybrid/experiments/results" / name)
    for name in (
        "platform_hybrid/sem 4 work/PROVENANCE.md",
        "research/public_results/README.md",
        "reports/public_revision_2026-10-03/README.md",
        "reports/public_revision_2026-10-04/thesis/README.md",
    ):
        selected.add(ROOT / name)

    for row in audit.get("source_bindings", []):
        path = ROOT / row["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("audit source escapes repository")
        selected.add(path)
    snapshot = {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in sorted(selected)
        if path.is_file()
    }
    modes = {
        path.relative_to(ROOT).as_posix(): 0o100755 if path.stat().st_mode & 0o111 else 0o100644
        for path in selected
        if path.is_file()
    }
    audit_name = (PACK / "thesis/provenance_audit.json").relative_to(ROOT).as_posix()
    validate_audit(json.loads(snapshot[audit_name]), snapshot)
    # Visual reviews are checked again against the bytes actually selected.
    for document, review in ((pdf, pdf_review), (deck, PACK / "validation/deck_checks.json")):
        quality = json.loads(snapshot[review.relative_to(ROOT).as_posix()])
        if (
            hashlib.sha256(snapshot[document.relative_to(ROOT).as_posix()]).hexdigest()
            != quality["sha256"]
            or quality["visual_review"] != "PASS"
        ):
            raise ValueError("document changed after visual review")
    rows = [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(snapshot.items())
    ]
    if len(rows) != len(selected):
        raise ValueError("A required review package input is missing")
    manifest_path = PACK / "REVIEW_MANIFEST.json"
    manifest = {
        "schema": "pes-evaluation-review-v2",
        "date": "2026-10-11",
        "scope": "Review documents, selected evidence, editable thesis sources and offline demo; not full experiment reproduction.",
        "self_excluded": manifest_path.relative_to(ROOT).as_posix(),
        "files": rows,
    }
    snapshot[manifest_path.relative_to(ROOT).as_posix()] = (
        json.dumps(manifest, indent=2) + "\n"
    ).encode()
    target = ROOT / "dist/TinkerRL_PES_Review_2026-10-11.zip"
    publish_archive(target, snapshot, modes)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n")
    print(json.dumps({"archive": str(target), "files": len(snapshot), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the explicitly selected PES review documents and offline demo."""

from __future__ import annotations

import hashlib
import json
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


def main() -> int:
    audit = json.loads((PACK / "thesis/provenance_audit.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("The current thesis number audit must pass before packaging")
    pdf = PACK / "thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf"
    quality = json.loads((PACK / "validation/pdf_checks.json").read_text())
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
    for name in ("SUMMARY.md", "pdf_checks.json", "deck_checks.json", "delivery_checks.json"):
        selected.add(PACK / "validation" / name)
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

    rows = [record(path) for path in sorted(selected) if path.is_file()]
    if len(rows) != len(selected):
        raise ValueError("A required review package input is missing")
    manifest_path = PACK / "REVIEW_MANIFEST.json"
    manifest = {
        "schema": "pes-evaluation-review-v1",
        "date": "2026-10-11",
        "scope": "Review documents, selected evidence, editable thesis sources and offline demo; not full experiment reproduction.",
        "self_excluded": manifest_path.relative_to(ROOT).as_posix(),
        "files": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    selected.add(manifest_path)
    target = ROOT / "dist/TinkerRL_PES_Review_2026-10-11.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            info = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 11, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o100755 if path.stat().st_mode & 0o111 else 0o100644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n")
    print(json.dumps({"archive": str(target), "files": len(selected), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

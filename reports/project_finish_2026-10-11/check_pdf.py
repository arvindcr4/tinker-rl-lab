"""Recheck the current PDF's text/bounds without overwriting historical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

FOLDER = Path(__file__).resolve().parent


def current_artifact(folder: Path = FOLDER) -> tuple[Path, Path, list[str]]:
    config = json.loads((folder / "thesis/current_artifact.json").read_text())
    return (
        folder / "thesis" / config["pdf"],
        folder / config["pdf_check_receipt"],
        config["required_text"],
    )


def check_pdf(pdf: Path, receipt_path: Path, required_text: list[str]) -> dict:
    out = receipt_path.parent
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-layout", str(pdf), str(out / "thesis_text.txt")], check=True)
    subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(out / "thesis_bbox.html")], check=True)
    text = (out / "thesis_text.txt").read_text()
    pages = text.split("\f")
    if not pages[-1].strip():
        pages.pop()
    raw = (out / "thesis_bbox.html").read_text()
    controls = r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
    tree = ET.fromstring(re.sub(controls, "", raw))  # noqa: S314 - local Poppler output
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    bounds = []
    for i, page in enumerate(tree.findall(".//x:page", ns), 1):
        width, height = float(page.get("width")), float(page.get("height"))
        for word in page.findall(".//x:word", ns):
            if (
                float(word.get("xMin")) < 0
                or float(word.get("yMin")) < 0
                or float(word.get("xMax")) > width + 1
                or float(word.get("yMax")) > height + 1
            ):
                bounds.append(i)
    normalized = " ".join(text.split())
    missing = [term for term in required_text if term not in normalized]
    receipt = {
        "schema": "pes-review-pdf-check-v2",
        "pdf": pdf.name,
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "bytes": pdf.stat().st_size,
        "pages": len(pages),
        "outside_page_bounds": sorted(set(bounds)),
        "replacement_character_pages": [i + 1 for i, page in enumerate(pages) if "\ufffd" in page],
        "missing_required_text": missing,
        "bbox_control_characters_removed_for_xml_parse": len(re.findall(controls, raw)),
        "visual_review": "pending",
        "scope": "Current PDF text and physical-page bounds only. Not a refreshed number audit, scientific authentication, accessibility certification or visual inspection.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    pdf, output, terms = current_artifact()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=pdf)
    parser.add_argument("--receipt", type=Path, default=output)
    args = parser.parse_args(argv)
    receipt = check_pdf(args.pdf, args.receipt, terms)
    print(json.dumps(receipt, indent=2))
    return int(bool(receipt["outside_page_bounds"] or receipt["missing_required_text"] or receipt["replacement_character_pages"]))


if __name__ == "__main__":
    raise SystemExit(main())

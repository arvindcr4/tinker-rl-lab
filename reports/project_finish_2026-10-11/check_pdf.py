"""Recheck the PDF's text and physical page bounds using Poppler."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

folder = Path(__file__).resolve().parent
pdf = folder / "thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf"
out = folder / "validation"
out.mkdir(exist_ok=True)
subprocess.run(["pdftotext", "-layout", str(pdf), str(out / "thesis_text.txt")], check=True)
subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(out / "thesis_bbox.html")], check=True)
text = (out / "thesis_text.txt").read_text()
pages = text.split("\f")
if not pages[-1].strip():
    pages.pop()
raw = (out / "thesis_bbox.html").read_text()
controls = r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
tree = ET.fromstring(re.sub(controls, "", raw))
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
audit_summary = json.loads((folder / "thesis/provenance_audit.json").read_text())["summary"]
terms = [
    "11 October 2026",
    f"{audit_summary['passed_checks']} checks pass",
    f"{audit_summary['context_only_not_independently_checked']} are context only",
    "136.4",
    "+0.32",
]
missing = [term for term in terms if term not in text]
receipt = {
    "schema": "pes-review-pdf-check-v1",
    "pdf": pdf.name,
    "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "bytes": pdf.stat().st_size,
    "pages": len(pages),
    "outside_page_bounds": sorted(set(bounds)),
    "replacement_character_pages": [i + 1 for i, page in enumerate(pages) if "\ufffd" in page],
    "missing_required_text": missing,
    "bbox_control_characters_removed_for_xml_parse": len(re.findall(controls, raw)),
    "visual_review": "pending",
    "scope": "Poppler text and physical-page bounds. This check does not authenticate scientific results, certify accessibility or replace visual inspection.",
}
(out / "pdf_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(int(bool(bounds or missing or receipt["replacement_character_pages"])))

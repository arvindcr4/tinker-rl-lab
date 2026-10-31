#!/usr/bin/env python3
"""Editorial/build checks; not a benchmark rerun or policy clearance."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

if not __debug__:
    raise RuntimeError("Run this audit without Python -O; assertions must remain enabled")


def check_printed_arithmetic(tex):
    """Read fractions/rounded answers from the manuscript, not copied literals."""
    matches = re.findall(r"(\d[\d,]*)/(\d[\d,]*)\s*=\s*(\d+\.\d+)(\\%)?", tex)
    assert matches, "No printed fraction/decimal comparisons found"
    for numerator, denominator, printed, percent in matches:
        value = int(numerator.replace(",", "")) / int(denominator.replace(",", ""))
        if percent:
            value *= 100
        precision = len(printed.split(".")[1])
        assert round(value, precision) == float(printed), (numerator, denominator, printed)
    conditional = re.search(
        r"(\d[\d,]*)/\((\d[\d,]*)(?:-|\$-\$)(\d[\d,]*)\)\s*=\s*(\d[\d,]*)/(\d[\d,]*)", tex
    )
    assert conditional, "Missing extraction-conditional denominator"
    numerator, total, failed, repeated, extracted = (
        int(v.replace(",", "")) for v in conditional.groups()
    )
    assert numerator == repeated
    assert total - failed == extracted
    return len(matches)


def count_primary_contrasts(inventory):
    """One top-level outcome, plus explicit nested primary outcome summaries."""

    def is_outcome(value):
        return isinstance(value, dict) and (
            {"trained_value", "base_value"} <= value.keys() or {"trained", "base"} <= value.keys()
        )

    return sum(
        int(is_outcome(row["summary"]))
        + sum(is_outcome(value) for value in row["summary"].values())
        for row in inventory
    )


def exact_sign_flip(differences):
    """Two-sided inclusive test of the absolute sum; Fraction avoids tie drift."""
    observed = abs(sum(differences))
    signed = [
        sum(sign * difference for sign, difference in zip(signs, differences, strict=True))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    extreme = sum(abs(value) >= observed for value in signed)
    return Fraction(extreme, len(signed))


def check_rendered_text(tex, body):
    """Require substantive extractable manuscript text and expected headings."""
    title = re.search(r"\\title\{([^}]+)\}", tex)[1]
    normalized_body = " ".join(body.split())
    assert title in normalized_body, "Rendered PDF lacks the manuscript title"
    for heading in ("Introduction", "Campaign Design", "References"):
        assert heading in normalized_body, f"Missing rendered section: {heading}"
    assert len(body.strip()) > 10000, "Insufficient extractable manuscript text"


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    tex = (root / "paper_camera.tex").read_text()
    bib = (root / "refs.bib").read_text()
    log = (root / "build/paper_camera.log").read_text()
    blg = (root / "build/paper_camera.blg").read_text()
    bbl = (root / "build/paper_camera.bbl").read_text()
    pdf = root / "build/paper_camera.pdf"

    def run(*args):
        return subprocess.check_output(args, text=True)

    info = run("pdfinfo", str(pdf))
    body = run("pdftotext", "-layout", str(pdf), "-")
    fonts = run("pdffonts", str(pdf))
    images = run("pdfimages", "-list", str(pdf))
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    check_rendered_text(tex, body)

    cited = {
        key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")
    }
    rendered = set(re.findall(r"\\bibitem\{([^}]+)\}", bbl))
    entries = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    abstract = tex.split(r"\begin{abstract}")[1].split(r"\keywords")[0]
    assert 10 <= pages <= 12, pages
    assert "(A4)" in info
    assert cited == rendered == entries, (cited, rendered, entries)
    assert len(cited) == 41, len(cited)
    assert 150 <= len(abstract.split()) <= 250, len(abstract.split())
    assert not re.search(r"Overfull|undefined|Fatal error|^!", log, re.M)
    assert "Warning" not in blg
    font_rows = fonts.splitlines()[2:]
    assert font_rows
    assert all(row.split()[-5] == "yes" and "Type 3" not in row for row in font_rows)
    image_rows = images.splitlines()[2:]
    assert not image_rows
    assert r"\author{Arvind C R$^{\star}$ \and Ramesh Prakash Guledgudd}" in tex
    assert r"\email{arvindcr4@gmail.com, rameshpg1988@gmail.com}\\$^{\star}$Corresponding author, +91 6379485347" in tex
    tables = tex.count(r"\begin{table}")
    figures = tex.count(r"\begin{figure}")
    equations = tex.count(r"\begin{equation}")
    assert (tables, figures, equations) == (5, 1, 2)
    ledger_table = tex.split(r"\label{tab:lanes}")[1].split(r"\end{table}")[0]
    later_table = tex.split(r"\label{tab:later}")[1].split(r"\end{table}")[0]
    originals = set(re.findall(r"(E\d+-O)\s*&", ledger_table))
    assert originals == {f"E{i}-O" for i in range(1, 15)}, originals
    replacements = set(re.findall(r"(E\d+-R)\s*&", ledger_table + later_table))
    state_fig = tex.split(r"\begin{figure}")[1].split(r"\end{figure}")[0]
    assert originals == set(re.findall(r"E\d+-O", state_fig))
    assert replacements == set(re.findall(r"E\d+-R", state_fig))

    inventory = json.loads((root / "verification/paired-outcomes-inventory.json").read_text())
    contrasts = count_primary_contrasts(inventory)
    paired_table = tex.split(r"\label{tab:paired}")[1].split(r"\end{tabularx}")[0]
    paired_rows = [line for line in paired_table.splitlines() if re.match(r"^(E\d+)?\s*&", line)]
    assert contrasts == len(paired_rows), (contrasts, len(paired_rows))
    assert {row["summary"]["lane"] for row in inventory} == set(
        re.findall(r"^(E\d+)\s*&", paired_table, re.M)
    )
    arithmetic_checks = check_printed_arithmetic(tex)
    signflip = json.loads((root / "verification/e12-signflip-arithmetic.json").read_text())
    differences = [
        Fraction(row["trained_passed"] - row["base_passed"], row["rubrics"])
        for row in signflip["apps"]
    ]
    paired_receipt = repo / signflip["source"]
    if paired_receipt.is_file():
        retained = json.loads(paired_receipt.read_text())
        for app in signflip["apps"]:
            rows = [
                value
                for key, value in retained["per_item"].items()
                if key.split(".")[0] == app["app"]
            ]
            assert len(rows) == app["rubrics"]
            assert sum(row["trained"] for row in rows) == app["trained_passed"]
            assert sum(row["base"] for row in rows) == app["base_passed"]
    signflip_p = exact_sign_flip(differences)
    assert float(signflip_p) == signflip["equal_app_pass_rate"]["two_sided_p"]
    e12_row = next(line for line in paired_rows if line.startswith("E12 "))
    printed_e12_p = float(re.search(r"\$p=([.\d]+)\$", e12_row)[1])
    assert printed_e12_p == float(signflip_p)
    assert "two-sided" in tex
    assert "equal-app pass-rate" in tex

    historical = repo / "outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json"
    historical_checked = historical.is_file()
    if historical_checked:
        assert (
            hashlib.sha256(historical.read_bytes()).hexdigest()
            == "7193cc30dfcdaacfaca736e246a971f83c1f7e06419e99337d8383e6bff3b3a2"
        )
        data = json.loads(historical.read_text())
        check = next(
            c for c in data["checks"] if c["check"] == "strict_completed_suites_are_E8_E10_E11"
        )
        assert check["status"] == "PASS"
        assert "replacement" not in check["detail"]
    assert "all-disposition" in tex
    assert "not a model score" in tex
    assert "exploratory" in tex
    assert "runtime contrast" in tex
    assert "proposed" in (root / "REVIEW_RESPONSE_AND_CHECKLIST.md").read_text().lower()
    years = [int(value) for value in re.findall(r"^\s*year\s*=\s*\{(\d+)\}", bib, re.M)]
    checked_at = datetime.now(timezone.utc)
    report = {
        "checked_at": checked_at.isoformat(),
        "checked_date": checked_at.date().isoformat(),
        "scope": "Editorial/build checks, source-parsed arithmetic, exact recount of retained app summaries and optional historical receipt hash; not new benchmark execution or venue-policy certification",
        "pages_including_references": pages,
        "paper_size": "A4",
        "extractable_text_characters": len(body.strip()),
        "rendered_title_verified": True,
        "cited_references": len(cited),
        "references_dated_2024_or_2025": sum(year in (2024, 2025) for year in years),
        "references_dated_2026": years.count(2026),
        "abstract_whitespace_word_count": len(abstract.split()),
        "unresolved_citations": len(cited - rendered),
        "overfull_boxes": len(re.findall("Overfull", log)),
        "bibtex_warnings": len(re.findall("Warning", blg)),
        "all_fonts_embedded_scalable": True,
        "raster_images": len(image_rows),
        "editable_tables": tables,
        "numbered_equations": equations,
        "vector_scope_state_figure": figures,
        "original_contract_rows": len(originals),
        "reported_replacement_identities": len(replacements),
        "primary_paired_outcome_contrasts": contrasts,
        "source_parsed_arithmetic_checks": arithmetic_checks,
        "e12_equal_app_pass_rate_two_sided_p": float(signflip_p),
        "e12_counts_checked_against_original_receipt": paired_receipt.is_file(),
        "historical_ledger_check_status": "checked existing repository record"
        if historical_checked
        else "not run: original ledger not included in standalone manuscript package",
        "historical_ledger_sha256_unchanged": True if historical_checked else None,
        "known_nonfatal_diagnostics": [
            "LLNCS/amsmath: Unable to redefine math accent vec",
            "Underfull bibliography lines; inspect render separately",
        ],
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((root / "paper_camera.tex").read_bytes()).hexdigest(),
        "bibliography_sha256": hashlib.sha256((root / "refs.bib").read_bytes()).hexdigest(),
    }
    (root / "verification/quality-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

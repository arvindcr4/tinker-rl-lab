"""The current thesis build, checker and packager share a dated artifact contract."""

import importlib.util
import json
from pathlib import Path

import pytest

from tools import build_pes_review_pack as packager

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "reports/project_finish_2026-10-11"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_and_both_consumers_resolve_same_current_pdf():
    build = load("current_thesis_build", PACK / "thesis/build_thesis.py")
    check = load("current_thesis_check", PACK / "check_pdf.py")
    pdf, receipt, terms = check.current_artifact()
    assert Path(build.PDF) == pdf
    assert packager.current_thesis_paths(PACK) == (pdf, receipt)
    assert pdf.name == "Thesis_Report_ArvindCR_2026-10-18_REVISED_REVIEW.pdf"
    assert receipt != PACK / "validation/pdf_checks.json"
    assert "18 October 2026" in terms
    assert not any("checks pass" in term for term in terms)


def test_current_contract_can_change_without_another_consumer_filename_edit(tmp_path):
    (tmp_path / "thesis").mkdir()
    (tmp_path / "thesis/current_artifact.json").write_text(
        json.dumps(
            {
                "pdf": "next-review.pdf",
                "pdf_check_receipt": "validation/next/check.json",
                "required_text": ["next revision"],
            }
        )
    )
    check = load("contract_thesis_check", PACK / "check_pdf.py")
    pdf, receipt, terms = check.current_artifact(tmp_path)
    assert packager.current_thesis_paths(tmp_path) == (pdf, receipt)
    assert pdf == tmp_path / "thesis/next-review.pdf"
    assert terms == ["next revision"]


def test_checker_preserves_old_receipt_and_requires_new_visual_review(tmp_path, monkeypatch):
    check = load("safe_thesis_check", PACK / "check_pdf.py")
    old = tmp_path / "validation/pdf_checks.json"
    old.parent.mkdir()
    old.write_text('{"visual_review":"PASS","historical":true}\n')
    before = old.read_bytes()
    pdf = tmp_path / "current.pdf"
    pdf.write_bytes(b"fixture PDF bytes")
    target = tmp_path / "validation/current/pdf_checks.json"

    def fake_poppler(cmd, **kwargs):
        if "-layout" in cmd:
            Path(cmd[-1]).write_text("18 October\n2026\nprovisional\f")
        else:
            Path(cmd[-1]).write_text(
                '<html xmlns="http://www.w3.org/1999/xhtml"><body><page width="100" '
                'height="100"><word xMin="1" yMin="1" xMax="20" yMax="20">'
                "provisional</word></page></body></html>"
            )

    monkeypatch.setattr(check.subprocess, "run", fake_poppler)
    result = check.check_pdf(pdf, target, ["18 October 2026", "provisional"])
    assert result["pages"] == 1
    assert result["missing_required_text"] == []
    assert result["outside_page_bounds"] == []
    assert result["visual_review"] == "pending"
    assert old.read_bytes() == before
    assert json.loads(target.read_text()) == result


def test_historical_numeric_pass_is_not_current_packaging_authority():
    historic = json.loads((PACK / "thesis/provenance_audit.json").read_text())
    assert historic["schema_version"] == "thesis-number-provenance-v1"
    with pytest.raises(ValueError, match="structurally bound v2"):
        packager.validate_audit(historic, {})


@pytest.mark.parametrize(
    ("selected", "available", "message"),
    [
        ("invalid", {"latexmk", "pdflatex"}, "must be tectonic or latexmk"),
        ("tectonic", {"latexmk", "pdflatex"}, "missing required tools: tectonic"),
        ("latexmk", {"tectonic"}, "missing required tools: latexmk, pdflatex"),
        ("latexmk", {"latexmk"}, "missing required tools: pdflatex"),
    ],
)
def test_figure_engine_override_fails_before_workers(monkeypatch, selected, available, message):
    figures = load("preflight_figures", PACK / "thesis/compile_figures.py")
    monkeypatch.setenv("THESIS_TEX_ENGINE", selected)
    monkeypatch.setattr(figures.shutil, "which", lambda name: name if name in available else None)
    with pytest.raises(ValueError, match=message):
        figures.select_engine()


def test_figure_engine_auto_selection(monkeypatch):
    figures = load("auto_figures", PACK / "thesis/compile_figures.py")
    monkeypatch.delenv("THESIS_TEX_ENGINE", raising=False)
    monkeypatch.setattr(
        figures.shutil, "which", lambda name: name if name in {"latexmk", "pdflatex"} else None
    )
    assert figures.select_engine() == "latexmk"


@pytest.mark.parametrize("missing", ["pdflatex", "bibtex", "kpsewhich"])
def test_thesis_preflight_requires_latexmk_dependencies(monkeypatch, capsys, missing):
    build = load("preflight_thesis", PACK / "thesis/build_thesis.py")
    monkeypatch.setenv("THESIS_TEX_ENGINE", "latexmk")
    monkeypatch.setattr(build.shutil, "which", lambda name: None if name == missing else name)
    assert build.main() == 1
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize(
    ("available", "explicit", "expected"),
    [
        ({"IEEEtran"}, None, "IEEEtran"),
        ({"ieeetr"}, None, "ieeetr"),
        ({"IEEEtran", "ieeetr"}, "ieeetr", "ieeetr"),
    ],
)
def test_direct_build_bibliography_detection(monkeypatch, available, explicit, expected):
    from types import SimpleNamespace

    build = load("style_thesis", PACK / "thesis/build_thesis.py")
    monkeypatch.delenv("THESIS_BIB_STYLE", raising=False)
    if explicit:
        monkeypatch.setenv("THESIS_BIB_STYLE", explicit)
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            returncode=0 if args[1][:-4] in available else 1,
            stdout="/tex/style.bst" if args[1][:-4] in available else "",
        ),
    )
    assert build.bibliography_style("latexmk") == expected


@pytest.mark.parametrize("explicit", [None, "missing"])
def test_no_silent_bibliography_substitution(monkeypatch, explicit):
    from types import SimpleNamespace

    build = load("missing_style_thesis", PACK / "thesis/build_thesis.py")
    monkeypatch.delenv("THESIS_BIB_STYLE", raising=False)
    if explicit:
        monkeypatch.setenv("THESIS_BIB_STYLE", explicit)
    monkeypatch.setattr(
        build.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="")
    )
    with pytest.raises(ValueError, match="missing bibliography style"):
        build.bibliography_style("latexmk")


def test_current_length_figure_caption_matches_evidence_boundary_source():
    build = load("caption_thesis", PACK / "thesis/build_thesis.py")
    rows = [row for rows in build.FIGURES.values() for row in rows if row[0] == "fig_length_bias"]
    assert len(rows) == 1
    assert "Evidence boundaries" in rows[0][1]
    assert "panels" not in rows[0][1]
    assert "peak-then-decay" not in rows[0][1]


def test_zvf_example_is_self_contained():
    import re

    text = (PACK / "thesis/ch04_methodology.md").read_text()
    snippet = next(
        code
        for code in re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
        if "def zvf" in code
    )
    namespace = {}
    exec(compile(snippet, "thesis-zvf-example", "exec"), namespace)  # noqa: S102
    assert namespace["zvf"]([[0, 0], [0, 1]])["zvf_exact"] == 0.5

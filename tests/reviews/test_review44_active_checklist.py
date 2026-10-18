"""Source-level guard for the active D&B checklist evidence boundary."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ROOT / "platform_hybrid/paper/neurips_2026_variants"


def test_dnb_includes_corrected_checklist():
    entry = (VARIANTS / "paper_P9_dnb.tex").read_text()
    assert r"\input{sections/checklist_anon}" in entry
    checklist = (VARIANTS / "sections/checklist_anon.tex").read_text()
    for boundary in (
        "Theory, not an empirical evidence grade",
        "expected exact-tie fraction",
        "not a reviewed-record held-out control",
        "executed-source identity",
        "cohort census",
        "one seed each and are Tier B",
        "step-level bootstrap intervals",
    ):
        assert boundary in checklist
    assert "only\n    unavailable information" not in checklist
    assert "A-grade, multi-seed inferential" not in checklist


def test_variant_summary_boundaries():
    dnb = (VARIANTS / "paper_P9_dnb.tex").read_text()
    assert "eligibility is unresolved for the historical cohort" in dnb
    assert "it is not\nan empirical inferential result" in dnb
    workshop = (VARIANTS / "paper_P8_workshop.tex").read_text()
    assert r"\input{sections/conclusion_workshop}" in workshop
    conclusion = (VARIANTS / "sections/conclusion_workshop.tex").read_text()
    assert "not a reviewed-record held-out control" in conclusion

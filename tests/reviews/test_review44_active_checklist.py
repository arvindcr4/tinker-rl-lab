"""Guard active evidence boundaries, including the later P8/P9 deep-review revisions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ROOT / "platform_hybrid/paper/neurips_2026_variants"


def _text(relative_path):
    """Ignore TeX source line wrapping when checking scientific caveats."""
    return " ".join((VARIANTS / relative_path).read_text().split())


def test_dnb_includes_corrected_checklist():
    entry = _text("paper_P9_dnb.tex")
    assert r"\input{sections/checklist_anon}" in entry
    checklist = _text("sections/checklist_anon.tex")
    for boundary in (
        "Theory, not an empirical evidence grade",
        "expected exact-tie fraction",
        r"conditionally i.i.d.\ binary rewards at fixed $p$",
        "not an identity for an observed finite batch",
        "outside the historical A/B/C tiers",
        "score treated as fixed",
        "not evidence of a gain",
        "executed-source identity",
        "cohort census",
        "one seed each and are Tier B",
        "step-level bootstrap intervals",
        "cannot establish seed-level uncertainty",
    ):
        assert boundary in checklist
    assert "only unavailable information" not in checklist
    assert "A-grade, multi-seed inferential" not in checklist


def test_variant_summary_boundaries():
    dnb = _text("paper_P9_dnb.tex")
    assert "eligibility is unresolved for the historical cohort" in dnb
    assert "it is not an empirical inferential result" in dnb
    assert "outside the A/B/C tiers and is not evidence of a gain" in dnb
    workshop = _text("paper_P8_workshop.tex")
    assert r"\input{sections/conclusion_workshop}" in workshop
    # The P8 deep review includes the later held-out comparison, whose records
    # are outside the historical manifest. Keep its null-result and fixed-base
    # boundaries rather than requiring the obsolete "future-paper" wording.
    assert "These runs are not manifest rows" in workshop
    assert "one-sample" in workshop
    assert "against the single deterministic base score" in workshop
    conclusion = _text("sections/conclusion_workshop.tex")
    for boundary in (
        "separate five-seed Qwen3-8B held-out comparison",
        "non-significant difference",
        "single base-model score treated as fixed",
        "outside the historical manifest",
        "do not establish a held-out gain",
    ):
        assert boundary in conclusion

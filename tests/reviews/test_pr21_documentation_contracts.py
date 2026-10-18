"""Static regression checks for PR21 documentation and publication boundaries."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_capstone_included_sections_are_peers():
    chapter = ROOT / "capstone-literature-survey/chapter2_foundation.tex"
    for name in re.findall(r"\\input\{([^}]+)\}", chapter.read_text()):
        source = (chapter.parent / f"{name}.tex").read_text()
        assert re.search(r"^\\section\{", source, re.M), name


def test_capstone_equations_do_not_contain_paragraph_breaks():
    source = (ROOT / "capstone-literature-survey/section_2_5_grpo.tex").read_text()
    for equation in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", source, re.S):
        assert not re.search(r"\n[ \t]*\n", equation)


def test_predictive_results_are_quarantined_until_recomputed():
    source = (ROOT / "platform_hybrid/experiments/zvf_predictive_validation.md").read_text()
    assert "Minimum eligible run length: 15 steps" in source
    assert "Quarantined historical results (not recomputed)" in source
    assert "Collapsed runs: 2 / 22" in source  # Preserve historical numbers.


def test_counterfactual_threshold_is_not_named_exact_zvf():
    source = (
        ROOT
        / "platform_hybrid/paper/neurips_2026_variants/sections/zvf_counterfactual_appendix.tex"
    ).read_text()
    assert "Near-constant fraction (last 10)" in source
    assert "This is not exact ZVF" in source
    assert "ZVF is the fraction of GRPO groups" not in source


def test_five_seed_instruct_row_is_descriptive():
    source = (
        ROOT / "platform_hybrid/paper/neurips_2026_variants/sections/base_vs_instruct_paired.tex"
    ).read_text()
    assert "& +0.0130 & 5 (d.o.)" in source


def test_optional_quickstarts_have_install_and_activation_steps():
    source = (ROOT / "README.md").read_text()
    assert "source .venv/bin/activate" in source
    assert "uv pip install --python .venv/bin/python tinker-cookbook" in source
    assert "uv pip install --python .venv/bin/python -e ./platform_tinker/atropos" in source

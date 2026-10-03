"""Build regressions use fake native tools; no thesis PDF is rebuilt here."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

THESIS_DIR = (
    Path(__file__).resolve().parents[1] / "outputs/PES_Phase2_Third_Review_2026-09-24/thesis"
)
OLD_PDF = b"previously published PDF"
NEW_PDF = b"newly compiled PDF"
FAILED_COMPILE_CASES = [
    (1, None),
    (1, b"partial PDF"),
    (0, None),
    (0, b""),
    ("timeout", b"partial PDF"),
]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, THESIS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def thesis(tmp_path, monkeypatch):
    tool = load_tool("build_thesis")
    paths = {
        "HERE": tmp_path,
        "BUILD": tmp_path / "build",
        "FIGDIR": tmp_path / "figures",
        "MASTER": tmp_path / "thesis_master.tex",
        "PDF": tmp_path / "Thesis_Report_ArvindCR.pdf",
        "EVIDENCE_MAP": tmp_path / "build/ch_back_evidence_map.md",
    }
    for name, path in paths.items():
        monkeypatch.setattr(tool, name, str(path))
    paths["BUILD"].mkdir()
    paths["FIGDIR"].mkdir()
    monkeypatch.setattr(tool, "CHAPTERS", [("chapter.md", None)])
    monkeypatch.setattr(tool, "APPENDICES", [("appendix.md", None)])
    monkeypatch.setattr(
        tool,
        "FIGURES",
        {
            "chapter.md": [("fig_required", "Caption.", "fig:required", ["Test"])],
            # A figure belonging to an unused chapter must not become required.
            "unused.md": [("fig_unused", "Unused.", "fig:unused", ["Unused"])],
        },
    )
    for name in ("chapter.md", "appendix.md"):
        (tmp_path / name).write_text(
            "# Test\n\n" + "Evidence-backed text. " * 30 + "(source: evidence/receipt.json)\n",
            encoding="utf-8",
        )
    for name in (
        "preamble.tex",
        "frontmatter.tex",
        "references.bib",
        "figures/pes_logo.png",
        "figures/signature_arvind.png",
        "figures/fig_required.pdf",
    ):
        (tmp_path / name).write_text("required input", encoding="utf-8")
    paths["MASTER"].write_text("previous master", encoding="utf-8")
    paths["PDF"].write_bytes(OLD_PDF)
    (paths["BUILD"] / "thesis_master.pdf").write_bytes(b"stale build PDF")
    monkeypatch.setattr(tool.shutil, "which", lambda name: f"/fake/{name}")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "pandoc":
            Path(cmd[cmd.index("-o") + 1]).write_text(
                "\\chapter{Test}\n\nComplete chapter.\n",
                encoding="utf-8",
            )
        else:
            assert cmd[0] == "tectonic"
            assert kwargs["cwd"] == tool.HERE
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            assert outdir != paths["BUILD"]
            assert not list(outdir.iterdir())
            assert paths["PDF"].read_bytes() == OLD_PDF
            (outdir / "thesis_master.pdf").write_bytes(NEW_PDF)
            (outdir / "thesis_master.log").write_text("fresh log", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(tool.subprocess, "run", Mock(side_effect=fake_run))
    return tool


@pytest.mark.parametrize(
    "missing",
    [
        "chapter.md",
        "appendix.md",
        "preamble.tex",
        "frontmatter.tex",
        "references.bib",
        "figures/pes_logo.png",
        "figures/signature_arvind.png",
        "figures/fig_required.pdf",
    ],
)
def test_missing_input_stops_before_native_tools(thesis, missing):
    (Path(thesis.HERE) / missing).unlink()
    assert thesis.main() == 1
    thesis.subprocess.run.assert_not_called()
    assert Path(thesis.PDF).read_bytes() == OLD_PDF
    assert Path(thesis.MASTER).read_text() == "previous master"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("chapter.md", b"# Incomplete"),
        ("appendix.md", b""),
        ("figures/fig_required.pdf", b""),
    ],
)
def test_incomplete_input_is_rejected(thesis, name, content):
    (Path(thesis.HERE) / name).write_bytes(content)
    assert thesis.main() == 1
    thesis.subprocess.run.assert_not_called()
    assert Path(thesis.PDF).read_bytes() == OLD_PDF


@pytest.mark.parametrize("missing", ["pandoc", "tectonic"])
def test_missing_native_tool_is_rejected(thesis, monkeypatch, missing):
    monkeypatch.setattr(thesis.shutil, "which", lambda name: None if name == missing else name)
    assert thesis.main() == 1
    thesis.subprocess.run.assert_not_called()
    assert Path(thesis.PDF).read_bytes() == OLD_PDF


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_failed_pandoc_cleans_temporary_markdown(thesis, failure):
    def fail(cmd, **kwargs):
        assert cmd[0] == "pandoc", "tectonic must not run after a failed chapter conversion"
        assert Path(cmd[1]).is_file()
        Path(cmd[cmd.index("-o") + 1]).write_text("partial conversion", encoding="utf-8")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 120)
        raise subprocess.CalledProcessError(1, cmd, stderr=b"conversion failed")

    thesis.subprocess.run.side_effect = fail
    assert thesis.main() == 1
    assert not list(Path(thesis.HERE).rglob("*.numbered.md"))
    assert Path(thesis.PDF).read_bytes() == OLD_PDF
    assert Path(thesis.MASTER).read_text() == "previous master"


@pytest.mark.parametrize(("exit_code", "artifact"), FAILED_COMPILE_CASES)
def test_failed_compile_cannot_publish_stale_or_partial_pdf(thesis, exit_code, artifact, capsys):
    convert = thesis.subprocess.run.side_effect

    def fail(cmd, **kwargs):
        if cmd[0] == "pandoc":
            return convert(cmd, **kwargs)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        assert outdir != Path(thesis.BUILD)
        assert not list(outdir.iterdir())
        if artifact is not None:
            (outdir / "thesis_master.pdf").write_bytes(artifact)
        if exit_code == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1200)
        return subprocess.CompletedProcess(cmd, exit_code, stdout=b"", stderr=b"failed")

    thesis.subprocess.run.side_effect = fail
    assert thesis.main() == 1
    assert any(call.args[0][0] == "tectonic" for call in thesis.subprocess.run.call_args_list)
    assert Path(thesis.PDF).read_bytes() == OLD_PDF
    assert (Path(thesis.BUILD) / "thesis_master.pdf").read_bytes() == b"stale build PDF"
    assert not list(Path(thesis.HERE).glob(".thesis-*"))
    assert "OK ->" not in capsys.readouterr().out


def test_success_publishes_new_pdf_with_atomic_replace(thesis, monkeypatch):
    replace = Mock(wraps=os.replace)
    monkeypatch.setattr(thesis.os, "replace", replace)
    assert thesis.main() == 0
    assert Path(thesis.PDF).read_bytes() == NEW_PDF
    replace.assert_called_once()
    source, destination = replace.call_args.args
    assert Path(source).parent.parent == Path(thesis.PDF).parent
    assert destination == thesis.PDF
    assert (Path(thesis.BUILD) / "thesis_master.log").read_text() == "fresh log"
    assert not list(Path(thesis.HERE).glob(".thesis-*"))
    assert not list(Path(thesis.HERE).rglob("*.numbered.md"))


def test_publication_error_preserves_existing_pdf(thesis, monkeypatch):
    monkeypatch.setattr(thesis.os, "replace", Mock(side_effect=OSError("publication failed")))
    assert thesis.main() == 1
    assert Path(thesis.PDF).read_bytes() == OLD_PDF
    assert not list(Path(thesis.HERE).glob(".thesis-*"))


@pytest.fixture
def figures(tmp_path, monkeypatch):
    tool = load_tool("compile_figures")
    monkeypatch.setattr(tool, "FIGDIR", str(tmp_path))
    (tmp_path / "fig_demo.tex").write_text("figure source", encoding="utf-8")
    (tmp_path / "fig_demo.pdf").write_bytes(OLD_PDF)
    os.utime(tmp_path / "fig_demo.tex", (10, 10))
    os.utime(tmp_path / "fig_demo.pdf", (20, 20))
    monkeypatch.setattr(tool.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(tool.subprocess, "run", Mock())
    return tool


def test_force_bypasses_figure_cache_and_publishes_new_pdf(figures, monkeypatch):
    assert figures.compile_one("fig_demo") == ("fig_demo", True, "cached")
    figures.subprocess.run.assert_not_called()

    def compile_pdf(cmd, **kwargs):
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        assert outdir != Path(figures.FIGDIR)
        assert not list(outdir.iterdir())
        assert (Path(figures.FIGDIR) / "fig_demo.pdf").read_bytes() == OLD_PDF
        (outdir / "fig_demo.pdf").write_bytes(NEW_PDF)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    figures.subprocess.run.side_effect = compile_pdf
    monkeypatch.setattr(figures.sys, "argv", ["compile_figures.py", "--force", "-j", "1"])
    assert figures.main() == 0
    figures.subprocess.run.assert_called_once()
    assert (Path(figures.FIGDIR) / "fig_demo.pdf").read_bytes() == NEW_PDF
    assert not list(Path(figures.FIGDIR).glob(".fig_demo-*"))


@pytest.mark.parametrize(("exit_code", "artifact"), FAILED_COMPILE_CASES)
def test_failed_figure_compile_keeps_published_pdf(figures, exit_code, artifact):
    def fail(cmd, **kwargs):
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        if artifact is not None:
            (outdir / "fig_demo.pdf").write_bytes(artifact)
        if exit_code == "timeout":
            raise subprocess.TimeoutExpired(cmd, 600)
        return subprocess.CompletedProcess(cmd, exit_code, stdout=b"", stderr=b"failed")

    figures.subprocess.run.side_effect = fail
    assert figures.compile_one("fig_demo", force=True)[1] is False
    assert (Path(figures.FIGDIR) / "fig_demo.pdf").read_bytes() == OLD_PDF
    assert not list(Path(figures.FIGDIR).glob(".fig_demo-*"))


def test_figure_cli_rejects_missing_selected_source(figures, monkeypatch):
    monkeypatch.setattr(figures.sys, "argv", ["compile_figures.py", "--only", "fig_demo,missing"])
    assert figures.main() == 1
    figures.subprocess.run.assert_not_called()


def test_figure_cli_requires_tectonic(figures, monkeypatch):
    monkeypatch.setattr(figures.sys, "argv", ["compile_figures.py"])
    monkeypatch.setattr(figures.shutil, "which", lambda name: None)
    assert figures.main() == 1
    figures.subprocess.run.assert_not_called()

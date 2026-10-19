"""Offline regressions for review-ledger parsing; never replay research scripts."""

import importlib
from pathlib import Path
import subprocess
import sys
import urllib.parse
from unittest.mock import Mock
from xml.parsers import expat

import pytest

arxiv = importlib.import_module("tools.review_ledger.P9_dnb_arxiv_check")

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2601.12345v2</id>
    <title> A title\n with &amp; whitespace </title>
    <author><name>A. Author</name></author>
    <author><name>B. Author</name></author>
    <summary> An abstract\n on two lines. </summary>
  </entry>
</feed>"""


def test_arxiv_formats_atom_text_and_optional_abstract():
    expected = ["2601.12345v2 | A title with & whitespace | A. Author, B. Author"]
    assert arxiv.format_entries(FEED) == expected
    assert arxiv.format_entries(FEED, include_abstract=True) == [
        *expected,
        "An abstract on two lines.",
    ]


@pytest.mark.parametrize(
    "declaration",
    [
        '<!DOCTYPE feed [<!ENTITY unsafe "expanded">]>',
        '<!DOCTYPE feed SYSTEM "file:///should-never-be-read">',
        '<!DOCTYPE feed [<!ENTITY unsafe SYSTEM "https://example.invalid/secret">]>',
        '<!DOCTYPE feed [<!ENTITY % unsafe SYSTEM "file:///never-read">%unsafe;]>',
    ],
)
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_arxiv_rejects_dtds_before_entity_expansion(declaration, encoding):
    xml = f'<?xml version="1.0" encoding="{encoding}"?>{declaration}<feed />'
    with pytest.raises(ValueError, match="DTD and entity declarations"):
        arxiv.parse_feed(xml.encode(encoding))


def test_arxiv_missing_required_field_has_explicit_error():
    with pytest.raises(ValueError, match="missing required text: a:title"):
        arxiv.format_entries(FEED.replace(b"<title>", b"<other>").replace(b"</title>", b"</other>"))


def test_arxiv_missing_abstract_only_fails_if_requested():
    feed = FEED.replace(b"<summary>", b"<other>").replace(b"</summary>", b"</other>")
    assert len(arxiv.format_entries(feed)) == 1
    with pytest.raises(ValueError, match="missing required text: a:summary"):
        arxiv.format_entries(feed, include_abstract=True)


def test_arxiv_malformed_xml_fails_closed():
    with pytest.raises(expat.ExpatError):
        arxiv.parse_feed(b"<feed>")


def test_arxiv_response_size_is_bounded(monkeypatch):
    monkeypatch.setattr(arxiv, "MAX_FEED_BYTES", 10)
    with pytest.raises(ValueError, match="response size limit"):
        arxiv.parse_feed(b"<feed>too large</feed>")


def test_arxiv_cli_uses_https_encoded_query_and_bounded_read(monkeypatch, capsys):
    response = Mock()
    response.read.return_value = FEED
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock(return_value=response)
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", opener)

    arxiv.main(["2601.12345,2602.54321&extra=bad", "abstract"])

    url = opener.call_args.args[0]
    assert url.startswith("https://export.arxiv.org/api/query?")
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query) == {
        "max_results": ["40"],
        "id_list": ["2601.12345,2602.54321&extra=bad"],
    }
    response.read.assert_called_once_with(arxiv.MAX_FEED_BYTES + 1)
    assert "An abstract on two lines." in capsys.readouterr().out


def test_arxiv_cli_requires_ids():
    with pytest.raises(SystemExit, match="usage:"):
        arxiv.main([])


def test_mixed_numeric_tsv_columns_keep_precise_runtime_values(tmp_path):
    from platform_modal.scripts._analysis_common import load_iter108_perrun

    path = tmp_path / "perrun.tsv"
    path.write_text(
        "model\twindow\tseed\tn_total\tn_in_window\tphi_L\tphi_R\tbwd\tfwd\tbwd_signed\tfwd_signed\n"
        "model-a\t4\t42\t12\t4\t0.1\t0.2\t0.3\t0.4\t-0.5\t0.6\n"
    )
    rows = load_iter108_perrun(str(path))
    assert rows == [
        {
            "model": "model-a",
            "window": 4,
            "seed": 42,
            "n_total": 12,
            "n_in_window": 4,
            "phi_L": 0.1,
            "phi_R": 0.2,
            "bwd": 0.3,
            "fwd": 0.4,
            "bwd_signed": -0.5,
            "fwd_signed": 0.6,
        }
    ]


def test_three_segment_fit_requires_sufficient_points_without_importing_runner():
    import ast
    from pathlib import Path

    import numpy as np

    # Compile only the pure helper: importing its report runner would create output dirs.
    path = Path(__file__).resolve().parents[1] / "tools/review_ledger/P1_scaling_rawtrace_refit.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "hinge3"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    fit = namespace["hinge3"]
    with pytest.raises(ValueError, match="At least six trace points"):
        fit(np.arange(5), np.arange(5))
    result = fit(np.arange(6), np.arange(6))
    assert (result["i"], result["j"]) == (2, 4)
    assert [result["m1"], result["m2"], result["m3"]] == pytest.approx([1, 1, 1])


def test_spectral_replay_imports_without_executing_report(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
def reject_write(*args, **kwargs):
    raise AssertionError('import must not write a replay artifact')
Path.write_text = reject_write
from tools.review_ledger import Spectral_Entropy_recompute as replay
from pilot import objective, benchmark_spectral_harness
assert replay.generate_benchmark_fixture is benchmark_spectral_harness.generate_benchmark_fixture
assert replay.condition_loss is objective.condition_loss
assert replay.entropic_givens_grpo is objective.entropic_givens_grpo
assert replay.spectral_legendre_grpo is objective.spectral_legendre_grpo
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

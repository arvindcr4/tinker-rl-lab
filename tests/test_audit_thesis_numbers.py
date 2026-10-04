from __future__ import annotations

import pytest

from tools import audit_thesis_numbers as audit


def test_wilson_matches_published_intervals():
    assert [round(x, 3) for x in audit.wilson(88, 97)] == [0.833, 0.950]
    assert [round(x, 2) for x in audit.wilson(11, 64, pct=True)] == [9.88, 28.21]
    assert audit.wilson(7, 7)[1] == 1.0


def test_signflip_and_holm():
    assert audit.signflip_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert audit.signflip_p([0.5, -0.5]) == 1.0
    assert audit.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_numbers_agree_at_printed_precision():
    assert audit.numbers_agree("[0.600, 0.670]", "[0.599500, 0.670223]")
    assert audit.numbers_agree("r = +0.21", "r = +0.205000")
    assert not audit.numbers_agree("+0.33", "+0.324600")
    assert not audit.numbers_agree("1/190 = 0.005", "1/190")


def test_in_source_ignores_tex_decoration_but_not_digits():
    tex = r"$\mathbf{90.10}$ & $[89.40,\,90.77]$ & $2{,}560$ & 64.76\%"
    for token in ("90.10", "89.40", "90.77", "2,560", "64.76%"):
        assert audit.in_source(token, tex)
    assert audit.in_source("89.4", tex)  # trailing zeros are the same number
    assert not audit.in_source("89.41", tex)
    assert not audit.in_source("0.77", tex)


def test_registry_tokens_are_verbatim_and_current():
    missing = [
        (group, ch)
        for group, ch, *_rest in audit.claims()
        for path in _rest[2]
        if not ((audit.ROOT / path).is_file() or audit.expand(path))
    ]
    if missing:
        pytest.skip(f"untracked campaign artifacts absent from this checkout: {len(missing)}")
    groups = audit.evaluate()
    failed = [(g["group"], r) for g in groups for r in g["tokens"] if r["result"] == "FAIL"]
    assert not failed
    assert all(r["in_chapter"] for g in groups for r in g["tokens"])

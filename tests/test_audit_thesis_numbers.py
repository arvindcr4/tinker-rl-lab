from __future__ import annotations

import json

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
    failed = [
        (g["group"], r)
        for g in groups
        for r in g["tokens"]
        if r["result"] in ("FAIL", "UNBOUND", "ARTIFACT_MISSING")
    ]
    assert not failed
    assert all(r["in_chapter"] for g in groups for r in g["tokens"])


def synthetic_group(method, *results):
    """A registered check fixture, not empirical thesis evidence."""
    return {
        "group": "§4.1 synthetic fixture",
        "chapter": audit.CHAPTERS["ch04"],
        "method": method,
        "artifacts": [],
        "tokens": [{"token": "0.5", "in_chapter": True, "result": result} for result in results],
        "status": next(
            (result for result in results if result in ("FAIL", "UNBOUND", "ARTIFACT_MISSING")),
            "NOT_CHECKED" if method == "W" else "PASS",
        ),
    }


@pytest.mark.parametrize("result", ["UNBOUND", "FAIL", "ARTIFACT_MISSING"])
def test_check_cli_rejects_incomplete_or_failed_bindings(monkeypatch, tmp_path, result):
    monkeypatch.setattr(audit, "evaluate", lambda: [synthetic_group("P", result)])
    report = tmp_path / "audit.json"
    appendix = tmp_path / "appendix.md"
    assert audit.main(["--check", "--json", str(report), "--appendix", str(appendix)]) == 1
    assert json.loads(report.read_text())["status"] == "FAIL"
    assert "0 checks pass" in appendix.read_text()


def test_report_separates_context_arithmetic_and_withheld(monkeypatch, tmp_path):
    groups = [
        synthetic_group("R", "PASS"),
        synthetic_group("A", "PASS", "CONTEXT"),
        synthetic_group("P", "CONTEXT"),
        synthetic_group("W", "NOT_CHECKED"),
    ]
    monkeypatch.setattr(audit, "evaluate", lambda: groups)
    report = tmp_path / "audit.json"
    appendix = tmp_path / "appendix.md"
    assert audit.main(["--check", "--json", str(report), "--appendix", str(appendix)]) == 0
    record = json.loads(report.read_text())
    assert record["status"] == "PASS"
    assert record["summary"] == {
        "registered_token_occurrences": 5,
        "passed_checks": 2,
        "context_only_not_independently_checked": 2,
        "printed_count_arithmetic_passed": 1,
        "withheld_not_checked": 1,
        "failed_checks": 0,
        "unbound_values": 0,
        "missing_artifact_values": 0,
        "passed_by_method": {"A": 1, "R": 1},
    }
    rendered = appendix.read_text()
    assert "2 checks pass and 2 are context only, not independently checked" in rendered
    assert "include 1 calculations from printed counts only" in rendered
    assert "1 occurrences rest on withheld sources and are not checked" in rendered
    assert "2 pass; 2 context; 1 not checked" in rendered
    assert "no value is left unchecked" not in rendered
    assert "5 checked values agree" not in rendered


def test_check_cli_uses_token_outcomes_when_group_status_is_stale(monkeypatch, tmp_path):
    group = synthetic_group("P", "UNBOUND")
    group["status"] = "PASS"
    monkeypatch.setattr(audit, "evaluate", lambda: [group])
    report = tmp_path / "audit.json"
    appendix = tmp_path / "appendix.md"
    assert audit.main(["--check", "--json", str(report), "--appendix", str(appendix)]) == 1
    record = json.loads(report.read_text())
    assert record["status"] == "FAIL"
    assert record["summary"]["unbound_values"] == 1


def test_alternate_thesis_directory_reads_fresh_sources_and_keeps_default(monkeypatch, tmp_path):
    original_thesis = audit.THESIS
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    selected = tmp_path / "reports" / "derived" / "thesis"
    selected.mkdir(parents=True)
    chapter = selected / audit.CHAPTERS["ch04"]
    chapter.write_text("old contents")
    rel = chapter.relative_to(tmp_path).as_posix()
    audit.text.cache_clear()
    assert audit.text(rel) == "old contents"
    chapter.write_text("## 4.1 Fresh fixture\n\n0.5")

    def evaluate_selected():
        assert audit.THESIS == "reports/derived/thesis"
        assert "Fresh fixture" in audit.text(rel)
        return [synthetic_group("R", "PASS")]

    monkeypatch.setattr(audit, "evaluate", evaluate_selected)
    assert audit.main(["--check", "--thesis-dir", "reports/derived/thesis"]) == 0
    report = json.loads((selected / "provenance_audit.json").read_text())
    assert report["thesis_directory"] == "reports/derived/thesis"
    assert (selected / "ch_back_provenance.md").is_file()
    assert audit.THESIS == original_thesis
    assert not (tmp_path / original_thesis / "provenance_audit.json").exists()
    audit.text.cache_clear()


@pytest.mark.parametrize("directory", ["../outside", "/", "missing-thesis"])
def test_alternate_thesis_directory_must_exist_inside_repository(monkeypatch, tmp_path, directory):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        audit.main(["--thesis-dir", directory])
    assert error.value.code == 2


def test_mechanism_statistics_recompute_seed_paired_intervals_and_historical_ratios():
    values = audit.mechanism_statistics()
    arithmetic, gsm8k = audit.MECHANISM_TASK_SEEDS
    assert values[arithmetic, "spearman_L_R"] == pytest.approx(
        (0.102480, -0.019246, 0.224206, 0.079598, 0.125), abs=1e-6
    )
    assert values[gsm8k, "ols_dL_dR"][:4] == pytest.approx(
        (-4.409500, -37.745556, 28.926556, 0.626662), abs=1e-6
    )
    assert audit.historical_mediation_percentages() == pytest.approx(
        [42.338, 54.986, 27.740, 12.240]
    )


def test_mechanism_pairing_uses_seed_keys_not_tsv_order(monkeypatch):
    expected = audit.mechanism_statistics()
    original = audit.tsv
    monkeypatch.setattr(audit, "tsv", lambda path: list(reversed(original(path))))
    assert audit.mechanism_statistics() == expected


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite"])
def test_mechanism_panel_rejects_missing_duplicate_or_nonfinite_values(monkeypatch, defect):
    rows = [dict(row) for row in audit.tsv(audit.MECHANISM_FILES[0])]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["ols_dL_dR"] = "nan"
    monkeypatch.setattr(audit, "tsv", lambda _path: rows)
    with pytest.raises(ValueError, match="mechanism (row|value)"):
        audit.mechanism_statistics()


def test_mechanism_prose_numbers_bind_only_after_successful_recomputation(monkeypatch):
    tokens = ["0.080", "−0.27", "−4.41", "−37.7", "+28.9", "55%", "28%", "0.081"]
    monkeypatch.setattr(audit, "claims", lambda: iter(()))
    monkeypatch.setattr(audit, "claims_ext", audit.mechanism_claims)
    monkeypatch.setattr(
        audit,
        "prose_claims",
        lambda _ch: iter(
            [("mechanism prose test", "ch06", tokens, "P", audit.MECHANISM_FILES, None)]
        ),
    )
    rows = audit.evaluate()[-1]["tokens"]
    assert [row["result"] for row in rows[:7]] == ["PASS"] * 7
    # An unregistered altered value is never admitted by the new bindings.
    assert rows[-1]["result"] in ("FAIL", "UNBOUND")


def test_changed_mechanism_source_fails_registered_contrast(monkeypatch):
    original = audit.tsv

    def changed_rows(path):
        rows = [dict(row) for row in original(path)]
        if path == audit.MECHANISM_FILES[0]:
            rows[0]["spearman_L_R"] = "0.9"
        return rows

    monkeypatch.setattr(audit, "tsv", changed_rows)
    monkeypatch.setattr(audit, "claims", lambda: iter(()))
    monkeypatch.setattr(audit, "claims_ext", audit.mechanism_claims)
    monkeypatch.setattr(audit, "prose_claims", lambda _ch: iter(()))
    groups = audit.evaluate()
    assert groups[0]["tokens"][2]["result"] == "FAIL"
    assert groups[0]["status"] == "FAIL"

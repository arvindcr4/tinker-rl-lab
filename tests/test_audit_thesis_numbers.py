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
    groups = audit.evaluate()
    # Always check available evidence, even when private campaign artifacts are absent.
    failed = [(g["group"], r) for g in groups for r in g["tokens"] if r["result"] == "FAIL"]
    assert not failed
    assert all(r["in_chapter"] for g in groups for r in g["tokens"])
    assert any(g["method"] == "R" and g["status"] == "PASS" for g in groups)
    for group in groups:
        if any(a["sha256"] is None for a in group["artifacts"]):
            assert group["status"] in {"ARTIFACT_MISSING", "NOT_CHECKED"}
    if any(g["status"] != "PASS" for g in groups):
        assert audit.overall_status(groups) == "INCOMPLETE"
    assert all(a["sha256"] for g in groups if "C1" in g["group"] for a in g["artifacts"])
    appendix = audit.appendix(groups, "audit.json")
    if any(g["status"] == "ARTIFACT_MISSING" for g in groups):
        assert "evidence unavailable" in appendix
    assert "no value is left unchecked" not in appendix


@pytest.mark.parametrize(
    ("printed", "computed", "expected"),
    [
        ("1e-5", "1.4e-5", True),
        ("1e-5", "1.6e-5", False),
        ("0.00000001", "0.00000002", False),
        ("−0.20", "-0.205", True),
        ("−0.20", "-0.20500001", False),
        ("1,234.00", "1234", True),
        ("none", "none", False),
    ],
)
def test_precision_boundaries(printed, computed, expected):
    assert audit.numbers_agree(printed, computed) is expected


@pytest.mark.parametrize(
    ("token", "source", "expected"),
    [
        ("0.2", "-0.2", False),
        ("0.2", "0.21", False),
        ("1", "1e-5", False),
        ("1e-5", "0.00001", True),
        ("0.984", "0.971--0.984", True),
        ("0.20", "0.2", True),
        ("absent", "other prose", False),
        ("present", "present text", True),
    ],
)
def test_source_numeric_boundaries(token, source, expected):
    assert audit.in_source(token, source) is expected


def test_norm_and_helper_boundaries():
    assert audit.norm_num("−1,234.00%") == (-1234, 2)
    assert audit.norm_num("1.2e-5") == (0.000012, 6)
    with pytest.raises(ValueError, match="finite"):
        audit.norm_num("nan")
    assert audit.mcnemar([(True, True), (False, False)]) == (0, 0, 1)
    assert audit.mcnemar([(True, False)] * 5) == (5, 0, 0.0625)
    assert audit.holm([]) == []
    assert audit.signflip_p([0, 0]) == 1
    assert audit.fisher_ci(0, 10)[0] == pytest.approx(-audit.fisher_ci(0, 10)[1])
    assert audit.frac(1, 2) == "1/2"
    assert audit.comma_frac(1000, 2000) == "1,000/2,000"
    assert audit.pct(1, 4) == 25
    with pytest.raises(ValueError, match="pattern not found"):
        audit.rx("missing", "text")


def mini_registry(monkeypatch, tmp_path, entries, chapter="0.25 0.05 0.9 absent"):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", "thesis")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    (tmp_path / "thesis").mkdir()
    (tmp_path / "thesis/chapter.md").write_text(chapter)
    (tmp_path / "source.txt").write_text("0.25")
    monkeypatch.setattr(audit, "claims", lambda: iter(entries))
    monkeypatch.setattr(audit, "claims_ext", lambda: iter(()))
    monkeypatch.setattr(audit, "prose_claims", lambda ch: iter(()))


def test_evaluator_methods_and_incomplete_status(monkeypatch, tmp_path):
    entries = [
        ("recompute", "ch06", ["0.25"], "R", ["source.txt"], lambda: [0.25]),
        ("transcribe", "ch06", ["0.25"], "T", ["source.txt"], None),
        ("code", "ch06", ["0.25"], "C", ["source.txt"], lambda: [[r"0\.25"]]),
        ("context", "ch06", ["0.05"], "A", [], lambda: [None]),
        ("verified prose", "ch06", ["0.25"], "P", [], None),
        ("threshold", "ch06", ["0.05"], "P", [], None),
        ("unbound", "ch06", ["0.9"], "P", [], None),
        ("missing", "ch06", ["0.25"], "R", ["absent.json"], lambda: [0.25]),
        ("withheld", "ch06", ["0.25"], "W", ["private.json"], None),
    ]
    mini_registry(monkeypatch, tmp_path, entries)
    groups = audit.evaluate()
    assert [g["tokens"][0]["result"] for g in groups] == [
        "PASS",
        "PASS",
        "PASS",
        "CONTEXT",
        "PASS",
        "CONTEXT",
        "UNBOUND",
        "ARTIFACT_MISSING",
        "NOT_CHECKED",
    ]
    assert audit.overall_status(groups) == "INCOMPLETE"
    assert audit.overall_status([]) == "INCOMPLETE"
    assert audit.overall_status(groups[:1]) == "PASS"
    assert "not checked" in audit.appendix(groups, "audit.json")
    # Readers must not retain a prior audit's file bytes.
    (tmp_path / "source.txt").write_text("0.9")
    assert audit.evaluate()[1]["status"] == "FAIL"


@pytest.mark.parametrize("compute", [list, lambda: [1, 2], lambda: 1, lambda: 1 / 0])
def test_bad_computation_is_reported(monkeypatch, tmp_path, compute):
    mini_registry(monkeypatch, tmp_path, [("bad", "ch06", ["0.25"], "R", [], compute)])
    groups = audit.evaluate()
    assert groups[0]["error"]
    assert audit.overall_status(groups) == "FAIL"


def test_missing_chapter_token_fails_even_withheld(monkeypatch, tmp_path):
    mini_registry(monkeypatch, tmp_path, [("bad", "ch06", ["not printed"], "W", [], None)])
    assert audit.evaluate()[0]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("method", "compute"),
    [
        ("R", lambda: ["0.9"]),
        ("C", lambda: [["no such pattern"]]),
    ],
)
def test_disagreement(monkeypatch, tmp_path, method, compute):
    mini_registry(
        monkeypatch, tmp_path, [("bad", "ch06", ["0.25"], method, ["source.txt"], compute)]
    )
    assert audit.evaluate()[0]["status"] == "FAIL"


def test_cli_success_incomplete_and_errors(monkeypatch, tmp_path, capsys):
    import json

    mini_registry(monkeypatch, tmp_path, [("ok", "ch06", ["0.25"], "R", [], lambda: [0.25])])
    args = [
        "--check",
        "--json",
        str(tmp_path / "report.json"),
        "--appendix",
        str(tmp_path / "appendix.md"),
    ]
    assert audit.main(args) == 0
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "PASS"
    monkeypatch.setattr(
        audit, "claims", lambda: iter([("missing", "ch06", ["0.25"], "W", [], None)])
    )
    assert audit.main(args) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "INCOMPLETE"
    assert report["token_counts"] == {"NOT_CHECKED": 1}
    assert audit.main(args[1:]) == 0  # writing a truthful incomplete report is supported
    (tmp_path / "thesis/chapter.md").unlink()
    assert audit.main(args) == 2
    assert "audit error:" in capsys.readouterr().err


def test_cli_output_error(monkeypatch, tmp_path, capsys):
    mini_registry(monkeypatch, tmp_path, [("ok", "ch06", ["0.25"], "R", [], lambda: [0.25])])
    assert audit.main(["--json", str(tmp_path / "absent/report.json")]) == 2
    assert "audit output error" in capsys.readouterr().err


def test_file_helpers_and_digest(monkeypatch, tmp_path):
    import hashlib

    mini_registry(monkeypatch, tmp_path, [])
    audit.evaluate()
    (tmp_path / "rows.tsv").write_text("# comment\na\tb\n1\t2\n")
    (tmp_path / "rows.jsonl").write_text('{"a":1}\n\n{"a":2}\n')
    assert audit.tsv("rows.tsv") == [{"a": "1", "b": "2"}]
    assert audit.jsonl("rows.jsonl") == [{"a": 1}, {"a": 2}]
    assert audit.expand("*.txt") == ["source.txt"]
    h = audit.sha256("source.txt")
    assert audit.sha256("*.txt") == hashlib.sha256(f"{h}  source.txt\n".encode()).hexdigest()
    assert audit.digest([]) == "—"
    assert audit.digest([{"path": "missing", "sha256": None}]) == "—"
    assert audit.digest([{"path": "source.txt", "sha256": h}]) == h[:12]
    assert audit.digest([{"path": "a", "sha256": h}, {"path": "b", "sha256": h}]).startswith("Σ")
    assert audit.short_path(["a/x", "b/x"]) == "2 × x"
    assert audit.short_path(["a/x", "b/y"]) == "2 files"
    assert audit.location("Table A.4 detail") == "Table A.4"
    assert audit.location("§6 rest") == "§6 recomputations"
    assert audit.location("§6.1 details") == "§6.1"


@pytest.mark.parametrize(("k", "n"), [(0, 0), (-1, 3), (4, 3), (float("nan"), 3)])
def test_wilson_rejects_invalid_counts(k, n):
    with pytest.raises(ValueError, match="Wilson requires"):
        audit.wilson(k, n)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_signflip_rejects_invalid_data(values):
    with pytest.raises(ValueError, match="nonempty finite"):
        audit.signflip_p(values)


@pytest.mark.parametrize("values", [[-0.1], [1.1], [float("nan")]])
def test_holm_rejects_invalid_probabilities(values):
    with pytest.raises(ValueError, match="probabilities"):
        audit.holm(values)


@pytest.mark.parametrize(("r", "n"), [(1.1, 10), (-1.1, 10), (0, 3), (float("nan"), 10)])
def test_fisher_rejects_undefined_interval(r, n):
    with pytest.raises(ValueError, match="Fisher interval"):
        audit.fisher_ci(r, n)


@pytest.mark.parametrize(("diff", "expected"), [(0, 0), (0.1, 0.5), (0.2, 1), (-0.1, 0.5)])
def test_tost_constant_data_limit(diff, expected):
    assert audit.tost_p([diff] * 4, 0.1) == expected


@pytest.mark.parametrize(("diffs", "margin"), [([1], 0.1), ([float("nan"), 0], 0.1), ([0, 1], 0)])
def test_tost_invalid_inputs(diffs, margin):
    with pytest.raises(ValueError, match="TOST"):
        audit.tost_p(diffs, margin)


def test_finish_reasons_formats(monkeypatch, tmp_path):
    import base64
    import json

    mini_registry(monkeypatch, tmp_path, [])
    audit.evaluate()
    body = {"choices": [{"finish_reason": "length"}, {"finish_reason": "stop"}]}
    (tmp_path / "bodies.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"raw_body_base64": base64.b64encode(json.dumps(body).encode()).decode()}
                ),
                "",
                json.dumps({"response": {"choices": [{"finish_reason": "stop"}]}}),
            ]
        )
    )
    (tmp_path / "dispatch_result.json").write_text(json.dumps({"responses": [body]}))
    assert audit.finish_reasons("bodies.jsonl") == ["length", "stop", "stop"]
    assert audit.finish_reasons("dispatch_result.json") == ["length", "stop"]


def test_e14_accepted_denominator_and_token_cap(monkeypatch, tmp_path):
    import json

    mini_registry(monkeypatch, tmp_path, [])
    audit.evaluate()
    monkeypatch.setattr(audit, "E14_DISP", "dispositions.jsonl")
    rows = [
        {"native_accepted": True, "correct": True, "actor_usage": {"completion_tokens": 2048}},
        {"native_accepted": True, "correct": False, "actor_usage": {"completion_tokens": 2047}},
        {"native_accepted": False, "correct": True, "actor_usage": {"completion_tokens": 100}},
    ]
    (tmp_path / "dispositions.jsonl").write_text("\n".join(map(json.dumps, rows)))
    assert audit.e14() == {
        "n": 3,
        "accepted": 2,
        "correct": 1,
        "capped": 1,
        "capped_ok": 1,
        "uncapped": 2,
        "uncapped_ok": 0,
    }


def test_bootstrap_reproducible_and_hash_bound(monkeypatch, tmp_path):
    import json

    mini_registry(monkeypatch, tmp_path, [])
    audit.evaluate()
    monkeypatch.setattr(audit, "E11_BOOT", "bootstrap.json")
    monkeypatch.setattr(audit, "E11_RECEIPT", "source.txt")
    (tmp_path / "bootstrap.json").write_text(
        json.dumps(
            {
                "paired_verdicts": {"a": [1, 0], "b": [1, 1]},
                "seed": 42,
                "replicates": 100,
                "ci95": [0.5, 1.0],
                "source_sha256": audit.sha256("source.txt"),
            }
        )
    )
    expected = {
        "cc": 2,
        "spec": 1,
        "n": 2,
        "lo": 0.5,
        "hi": 1.0,
        "stored": [0.5, 1.0],
        "source_sha_ok": True,
    }
    assert audit.e11_boot() == expected
    audit.e11_boot.cache_clear()
    assert audit.e11_boot() == expected
    (tmp_path / "source.txt").write_text("changed source")
    audit.sha256.cache_clear()
    audit.e11_boot.cache_clear()
    assert audit.e11_boot()["source_sha_ok"] is False


def test_prose_keeps_unavailable_citations(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "THESIS", ".")
    monkeypatch.setattr(audit, "CHAPTERS", {"ch06": "chapter.md"})
    audit.text.cache_clear()
    (tmp_path / "chapter.md").write_text(
        "## 6.1 Results\n\nScore 0.25 (source: missing/data.json)."
    )
    groups = list(audit.prose_claims("ch06"))
    assert groups[0][2:5] == (["0.25"], "P", ["missing/data.json"])


def test_signflip_scale_invariant():
    assert audit.signflip_p([1e-15] * 5) == audit.signflip_p([1] * 5) == 2 / 32


def test_fisher_perfect_correlation_limits():
    assert audit.fisher_ci(1, 10) == (1, 1)
    assert audit.fisher_ci(-1, 10) == (-1, -1)


@pytest.mark.parametrize("values", [[], [1], [float("nan"), 1], [float("inf"), 1]])
def test_paired_t_rejects_invalid_inputs(values):
    with pytest.raises(ValueError, match="paired t interval"):
        audit.paired_t(values)


def test_paired_t_constant_and_variable():
    assert audit.paired_t([0.25, 0.25]) == (0.25, 0.25, 0.25)
    mean, lo, hi = audit.paired_t([1, 2, 3])
    assert mean == 2
    assert lo < mean < hi
    assert mean - lo == pytest.approx(hi - mean)


@pytest.mark.parametrize("method", ["R", "S", "T", "C"])
def test_prose_fallback_does_not_match_substrings_or_claim_recomputation(
    monkeypatch, tmp_path, method
):
    compute = (lambda: [[r"0\.25"]]) if method == "C" else (lambda: [10.9])
    mini_registry(
        monkeypatch,
        tmp_path,
        [
            ("earlier", "ch06", ["10.9"], method, ["source.txt"], compute),
            ("prose", "ch06", ["0.9"], "P", [], None),
            ("same prose", "ch06", ["10.9"], "P", [], None),
        ],
        chapter="10.9 0.9",
    )
    (tmp_path / "source.txt").write_text("10.9 0.25")
    groups = audit.evaluate()
    assert groups[0]["status"] == "PASS"
    assert groups[1]["status"] == "UNBOUND"
    assert groups[2]["status"] == ("PASS" if method == "R" else "UNBOUND")


def test_precise_supplementary_claim_calculations(monkeypatch):
    monkeypatch.setattr(
        audit,
        "text",
        lambda _: (
            "library/framework ($eta = 0.546$), training algorithm ($eta = 0.558$), family explains $eta = 0.471$; model scale explains $eta = 0.322$."
        ),
    )
    assert audit.marginal_eta_sum() == ["These four values sum to 1.897000"]
    monkeypatch.setattr(
        audit,
        "tsv",
        lambda _: [
            {"G": "2", "snr_advantage_variance": "1.4628"},
            {"G": "16", "snr_advantage_variance": "2.1622"},
        ],
    )
    assert audit.numbers_agree("52%", audit.snr_ratios()[0])
    assert audit.numbers_agree("48%", audit.snr_ratios()[1])
    monkeypatch.setattr(
        audit,
        "tsv",
        lambda _: [
            {
                "claim_id": "P2-C2",
                "heldout_metric": "Spearman rho=0.27, bootstrap 95% CI [-0.37, 0.88], n=23 pooled cells",
            }
        ],
    )
    assert audit.registry_bootstrap_ci() == ["the registry's bootstrap 95% CI is [-0.37, 0.88]"]


def test_prose_bindings_keep_method_and_are_section_specific():
    claim = {
        "group": "§6 registry bootstrap CI",
        "chapter": "ch06_results_core.md",
        "status": "PASS",
        "method": "S",
        "tokens": [{"token": "CI [−0.37, 0.88]", "result": "PASS"}],
    }
    assert audit.bound_prose_evidence("§6.3.4 prose", "−0.37", claim["chapter"], [claim]) is claim
    assert audit.bound_prose_evidence("§6.3.5 prose", "−0.37", claim["chapter"], [claim]) is None
    assert audit.bound_prose_evidence("§6.3.4 prose", "0.27", claim["chapter"], [claim]) is None
    claim["status"] = "ARTIFACT_MISSING"
    assert audit.bound_prose_evidence("§6.3.4 prose", "−0.37", claim["chapter"], [claim]) is None


def test_registry_ci_requires_unique_target_row(monkeypatch):
    monkeypatch.setattr(audit, "tsv", lambda _: [])
    with pytest.raises(ValueError, match="exactly one P2-C2"):
        audit.registry_bootstrap_ci()

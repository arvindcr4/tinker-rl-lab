"""Offline regression reproductions for repository-hygiene review 15."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = "autoresearch/improve-260714-1806/"
REMOTE = "autoresearch/reason-260727-2155/"


def test_secret_exemptions_are_not_global_paths_or_digest_lines():
    config = tomllib.loads((ROOT / ".gitleaks.toml").read_text())
    for rule in config["allowlists"]:
        assert rule.get("targetRules") == ["generic-api-key"]
        assert "generation_response" not in json.dumps(rule)
        assert "{64}" not in json.dumps(rule)
        assert "identifier-shaped" not in rule["description"]


def test_sanitizer_nested_url_headers_formats(monkeypatch):
    m = load(REMOTE + "audit_sanitizer.py")
    secret = "synthetic-value-not-a-real-credential"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    value = {
        "model": {"source_api_token": secret},
        "git": "https://name:password@example.org/repo",
        "lines": [
            "error tml-" + "X" * 32,
            "checkpoint hf_" + "Y" * 32,
            "warning api_key=" + "a" * 40,
            "Authorization: Bearer examplecredential",
            secret,
        ],
    }
    result = json.dumps(m.sanitize(value))
    for raw in (secret, "name:password", "X" * 32, "Y" * 32, "a" * 40, "examplecredential"):
        assert raw not in result


def paper():
    return {
        "identity": {"title": "t", "one_sentence_thesis": "x", "intended_paper_type": "p"},
        "claim_status": [],
        "strongest_contribution": "x",
        "major_problems": [],
        "overlap_and_scope": {
            "duplicates": [],
            "unique_material": "x",
            "scope_sentence_to_add": "x",
        },
        "consistency_checks": [],
        "concrete_edits": [],
        "evidence_needed": [],
        "verdict": "keep",
    }


def test_cache_binds_inputs_prompts_and_model(tmp_path, monkeypatch):
    m = load(AUDIT + "audit_papers.py")
    calls = []
    monkeypatch.setattr(
        m, "call_with_retry", lambda *args: calls.append(args) or json.dumps(paper())
    )

    def review(inputs):
        return m.cached_review(None, tmp_path, "prompt", inputs)

    review({"pdf": "a", "text": "a"})
    review({"pdf": "a", "text": "a"})
    assert len(calls) == 1
    review({"pdf": "b", "text": "b"})
    monkeypatch.setattr(m, "MODEL", "new-model")
    review({"pdf": "b", "text": "b"})
    assert len(calls) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert not (tmp_path / "inventory.tsv").exists()


def test_invalid_review_never_cached(tmp_path, monkeypatch):
    m = load(AUDIT + "audit_papers.py")
    for raw in ("not json", '{"parse_error":"failed"}', "[]"):
        monkeypatch.setattr(m, "call_with_retry", lambda *args, raw=raw: raw)
        with pytest.raises(ValueError, match=".+"):
            m.cached_review(None, tmp_path, "prompt", {})
    assert not list(tmp_path.iterdir())


def test_hf_fail_closed_and_optional_sidecar():
    m = load(REMOTE + "huggingface_history_audit.py")
    units = [{"unit": "a/seed-1", "hf_repo": "o/r", "hf_commit": "abc"}]
    m.validate_roster(units)
    row = {
        "expected_commit": "abc",
        "resolved_commit": "abc",
        "checkpoint_steps": m.EXPECTED_STEPS,
        "required_files": {"manifest": True, "adapter": True},
        "optional_files": {"progress": False},
        "manifest_checks": {"valid": True},
        "expected_manifest_sha256": "sha",
        "manifest_sha256": "sha",
    }
    assert m.integrity_verdict(units, [row], []) == "PASS"
    for field, wrong in [
        ("resolved_commit", "wrong"),
        ("checkpoint_steps", []),
        ("manifest_checks", {"valid": False}),
        ("required_files", {"adapter": False}),
        ("manifest_sha256", "wrong"),
    ]:
        invalid = dict(row, **{field: wrong})
        assert m.integrity_verdict(units, [invalid], []) == "FAIL"
    assert m.integrity_verdict(units, [], []) == "INCOMPLETE"
    with pytest.raises(ValueError, match=".+"):
        m.validate_roster(units + units)
    with pytest.raises(ValueError, match=".+"):
        m.validate_roster(units + [dict(units[0], unit="b/seed-2")])


def test_final_gate_rejects_failure_stale_hash_and_critical():
    m = load("tools/research_review_gate.py")
    report = b"final"
    digest = hashlib.sha256(report).hexdigest()
    manifest = {
        "schema_version": 2,
        "run_id": "r",
        "query_sha256": "q",
        "corpus_sha256": "c",
        "tier": "full",
        "final_sha256": digest,
        "lint_status": "passed",
    }
    reviews = [
        dict(manifest, critic=c, draft_sha256=digest, status="completed", findings=[])
        for c in m.CRITICS
    ]
    assert m.validate(report, manifest, reviews)["review_status"] == "complete"
    for change in (
        {"status": "failed", "findings": None},
        {"draft_sha256": "old"},
        {"findings": [{"severity": "critical", "resolution": "pending"}]},
    ):
        invalid = copy.deepcopy(reviews)
        invalid[0].update(change)
        with pytest.raises(ValueError, match=".+"):
            m.validate(report, manifest, invalid)
    with pytest.raises(ValueError, match=".+"):
        m.validate(b"edited after review", manifest, reviews)
    with pytest.raises(ValueError, match=".+"):
        m.validate(report, manifest, reviews[:-1])


def test_provenance_manifest_missing_records_and_signatures(tmp_path, monkeypatch):
    m = load("platform_hybrid/registry/provenance/verify_inventory.py")
    (tmp_path / "required_records.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [{"path": "a.provenance.json", "signature_required": True}],
            }
        )
    )
    calls = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match=".+"):
        m.verify(tmp_path)
    (tmp_path / "a.provenance.json").write_text("{}")
    with pytest.raises(ValueError, match=".+"):
        m.verify(tmp_path)
    (tmp_path / "a.provenance.json.sig").write_text("signature")
    m.verify(tmp_path)
    assert len(calls) == 2
    (tmp_path / "unexpected.provenance.json").write_text("{}")
    with pytest.raises(ValueError, match=".+"):
        m.verify(tmp_path)


def test_response_contract_rejects_reversed_markers_and_identity_free_cells():
    m = load("autoresearch/orchestrator-260730-1818/reviewer_response_metric.py")
    start, end = "<!-- POSTABLE_REPLY_START -->", "<!-- POSTABLE_REPLY_END -->"
    assert not m._postable(end + "x" + start)
    assert not m._postable(start + start + "x" + end)
    assert m._postable(start + "x" + end) == "x"
    assert not m.valid_cells([{"status": "not_evaluated"}] * 20)


def test_per_root_bibliography_and_repeated_include(tmp_path):
    inventory = load(AUDIT + "inventory_papers.py")
    review = load(AUDIT + "self_review_corpus.py")
    (tmp_path / "a.tex").write_text(r"\input{shared}\input{shared}\bibliography{a}")
    (tmp_path / "b.tex").write_text(r"\bibliography{b}")
    (tmp_path / "shared.tex").write_text(r"\label{twice}\cite{other}")
    (tmp_path / "a.bib").write_text("@article{mine, title={Mine}}")
    (tmp_path / "b.bib").write_text("@article{other, title={Other}}")
    assert "other" not in review.bib_keys(tmp_path / "a.tex")
    assert "other" in review.bib_keys(tmp_path / "b.tex")
    assert inventory.include_occurrences(tmp_path / "a.tex").count(tmp_path / "shared.tex") == 2
    (tmp_path / "shared.tex").write_text(r"\input{a}")
    with pytest.raises(ValueError, match=".+"):
        inventory.include_occurrences(tmp_path / "a.tex")


def test_wandb_limit_bounds_iterator_and_filters_secret_config(monkeypatch, capsys):
    from types import SimpleNamespace

    scanned = []

    def runs(*args, **kwargs):
        for i in range(10):
            scanned.append(i)
            yield SimpleNamespace(
                id=str(i),
                name="run",
                state="finished",
                created_at="now",
                group=None,
                job_type=None,
                tags=[],
                summary={},
                config={
                    "model": "test",
                    "source_api_token": "hidden",
                    "source": {"token": "hidden"},
                },
            )

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=lambda **kw: SimpleNamespace(runs=runs))
    )
    m = load(REMOTE + "wandb_claim_audit.py")
    monkeypatch.setattr(sys, "argv", ["inspect", "project", "--limit", "2"])
    m.main()
    output = capsys.readouterr().out
    assert len(scanned) == 3  # one lookahead establishes truncation
    assert json.loads(output.splitlines()[0])["complete"] is False
    assert "hidden" not in output


def test_review_budget_prevents_dispatch_and_permanent_retries(monkeypatch):
    from types import SimpleNamespace

    m = load(AUDIT + "audit_papers.py")
    calls = []

    def permanent(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("model not found")
        error.status_code = 404
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=permanent)))
    with pytest.raises(ValueError, match="budget exhausted"):
        m.call_with_retry(client, [])
    assert not calls
    monkeypatch.setattr(m, "REQUEST_BUDGET", 4)
    with pytest.raises(RuntimeError, match="model not found"):
        m.call_with_retry(client, [])
    assert len(calls) == 1


def test_real_gitleaks_canaries(tmp_path):
    """Run real scanner when installed; never substitute regex simulation."""
    import shutil
    import subprocess

    binary = shutil.which("gitleaks")
    if not binary:
        pytest.skip("gitleaks binary unavailable")
    target = tmp_path / "outputs" / "canary" / "tasks" / "example"
    target.mkdir(parents=True)
    # Synthetic credentials are generated locally, never used for authentication.
    hex_value = hashlib.sha256(b"review15 synthetic generic credential").hexdigest()
    (target / "generation_response.txt").write_text('HF_TOKEN="hf_' + hex_value[:34] + '"\n')
    (target / "source_context.json").write_text(json.dumps({"api_key": hex_value}))
    (target / "inventory.json").write_text(
        json.dumps({"digest": "a" * 64, "api_key": hex_value[::-1]})
    )
    report = tmp_path / "scan.json"
    result = subprocess.run(
        [
            binary,
            "dir",
            "--config",
            str(ROOT / ".gitleaks.toml"),
            "--redact",
            "--report-format",
            "json",
            "--report-path",
            str(report),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    findings = json.loads(report.read_text())
    found = {Path(row["File"]).name for row in findings}
    assert {"generation_response.txt", "source_context.json", "inventory.json"} <= found

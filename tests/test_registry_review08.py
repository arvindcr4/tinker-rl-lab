"""Review08 reproductions: evidence gates must fail closed without rewriting history."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "platform_hybrid/registry"
sys.path.insert(0, str(REG))
import query as q
import generate_entries as g


def cell(value, status="code"):
    return {"value": value, "status": status, "source": "fixture.py:L1"}


def complete():
    rec = json.loads((REG / "entries/tinker_grpo_qwen3-8b_gsm8k.json").read_text())
    rec["run_config"] = {f: cell("v-" + f) for f in q.FIELDS}
    rec["variant_deltas_applied"] = []
    return rec


@pytest.fixture
def registry(tmp_path, monkeypatch):
    reg = tmp_path / "repo/platform_hybrid/registry"
    (reg / "entries").mkdir(parents=True)
    (reg / "schema.json").write_bytes((REG / "schema.json").read_bytes())
    target = reg.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(
        (ROOT / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py").read_bytes()
    )
    monkeypatch.setattr(q, "HERE", reg)
    return reg


def put(reg, record, name="one.json"):
    (reg / "entries" / name).write_text(json.dumps(record))


def audit_runner(reg, *, findings=(), version=None, digest=None, output=True, mutate=False):
    def run(argv, **kwargs):
        if output:
            path = Path(argv[argv.index("--output") + 1])
            manifest = q.registry_input_manifest(reg)
            path.write_text(
                json.dumps(
                    {
                        "validator_version": q.VALIDATOR_VERSION if version is None else version,
                        "validator_sha256": hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest(),
                        "input_manifest": manifest,
                        "input_sha256": digest or q.registry_manifest_sha256(manifest),
                        "findings": list(findings),
                        "n_stack_entries": 0,
                        "n_deltas": 0,
                    }
                )
            )
            if mutate:
                (reg / "entries/mutation.json").write_text("{}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_failed_validator_never_reuses_clean_audit(registry, monkeypatch, capsys):
    cache = registry.parent / "experiments/results/p5p8/p6_iter118_strict_audit.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"findings":[],"n_stack_entries":0,"n_deltas":0}')
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=2, stdout="", stderr="child crash"),
    )
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) != 0
    result = capsys.readouterr()
    assert "child crash" in result.err
    assert "CI GATE: PASS" not in result.out
    assert cache.read_text().startswith('{"findings"')


@pytest.mark.parametrize(
    "kwargs", [{"output": False}, {"version": "old"}, {"digest": "stale"}, {"mutate": True}]
)
def test_gate_rejects_success_without_fresh_bound_output(registry, monkeypatch, kwargs):
    monkeypatch.setattr(subprocess, "run", audit_runner(registry, **kwargs))
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 1


def test_warning_verdict_matches_exit(registry, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", audit_runner(registry, findings=[{"severity": "warn"}]))
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=True)) == 1
    assert "CI GATE: FAIL" in capsys.readouterr().out


def test_success_requires_current_validator_and_inputs(registry, monkeypatch):
    monkeypatch.setattr(subprocess, "run", audit_runner(registry))
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 0


@pytest.mark.parametrize(
    "mutation", ["record_type", "id", "bad_status", "delta_duplicate", "required_leaf"]
)
def test_ingestion_rejects_hidden_or_malformed_records(registry, mutation):
    rec = complete()
    if mutation in ("record_type", "id"):
        rec.pop(mutation)
    elif mutation == "bad_status":
        rec["run_config"]["seeds"]["status"] = "unverified"
    elif mutation == "delta_duplicate":
        rec["variant_deltas_applied"] = [
            {"delta_id": "delta_dapo", "component": "x", "status": "unknown"}
        ] * 2
    else:
        rec["min_report"]["decontamination"].pop("parser_robustness_probe")
    put(registry, rec)
    with pytest.raises(ValueError, match="invalid registry record one\\.json"):
        q.load()


def test_duplicate_ids_rejected(registry):
    put(registry, complete())
    put(registry, complete(), "two.json")
    with pytest.raises(ValueError, match="duplicate registry id"):
        q.load()


@pytest.mark.parametrize("text", ['{"id":1,"id":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_strict_json_rejects_ambiguous_nonfinite(text):
    with pytest.raises(
        ValueError,
        match="duplicate JSON property|nonfinite JSON number|only finite, unambiguous JSON values are supported",
    ):
        q.strict_json_loads(text)


def test_registered_sidecar_digest_cannot_hide_record(registry):
    side = {"amendment": "historical"}
    put(registry, side, "amendment.json")
    with pytest.raises(ValueError, match="invalid registry record amendment\\.json"):
        q.load()
    raw = (registry / "entries/amendment.json").read_bytes()
    (registry / "sidecars.json").write_text(
        json.dumps(
            {
                "version": "registry-sidecars-v1",
                "documents": {
                    "amendment.json": {
                        "record_type": "historical_amendment",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
            }
        )
    )
    assert q.load() == {}
    put(registry, {"amendment": "edited"}, "amendment.json")
    with pytest.raises(ValueError, match="unrecognized or modified sidecar: amendment\\.json"):
        q.load()


@pytest.mark.parametrize("status", [None, "unverified", "garbage", "unknown"])
def test_unsupported_status_never_known(status):
    assert not q.is_known({"value": 1, "status": status, "source": "test"})


def test_deletion_does_not_improve_badge():
    a = {"performed": False, "parser_robustness_probe": None}
    b = {"performed": False}
    assert q.item_score(a, "decontamination") == q.item_score(b, "decontamination") == 0.5
    assert q.item_score({"source": "citation"}, "telemetry") == 0


def test_exact_integer_and_tiny_float_identity():
    for field, a, b in [("seeds", [2**53], [2**53 + 1]), ("optimizer.eps", 1e-13, 0.0)]:
        x, y = complete(), complete()
        x["run_config"][field] = cell(a)
        y["run_config"][field] = cell(b)
        assert q.stackdiff(x, y)["level"] == q.FIELDS[field]


def test_adaptive_rules_produce_r3():
    a, b = complete(), complete()
    for rec, rule in [(a, "if zvf > .5 increase to 16"), (b, "if zvf > .5 decrease to 4")]:
        rec.pop("run_config")
        rec["min_report"]["group_size_schedule"] = {
            "initial_g": 8,
            "schedule": "adaptive",
            "adaptation_rule": rule,
        }
    assert (
        "batch.adaptation_rule",
        3,
        a["min_report"]["group_size_schedule"]["adaptation_rule"],
        b["min_report"]["group_size_schedule"]["adaptation_rule"],
    ) in q.stackdiff(a, b)["differs"]


def test_unilateral_delta_and_revision_unknown_block_comparability():
    a, b = complete(), complete()
    a["variant_deltas_applied"] = [
        {"delta_id": "delta_dapo", "component": "clip", "status": "implemented"}
    ]
    result = q.stackdiff(a, b)
    assert result["level"] == 0
    assert not result["comparable"]
    assert result["hidden_max"] == 4
    assert result["coverage"] < 1
    b = complete()
    b["run_config"]["checkpoint.revision"] = cell(None, "unknown")
    result = q.stackdiff(complete(), b)
    assert result["underspecified"]
    assert not result["comparable"]


def test_assumptions_are_conditional_and_semantic_changes_require_review():
    a, b = complete(), complete()
    b["run_config"]["optimizer.eps"] = cell("v-optimizer.eps", "default")
    result = q.stackdiff(a, b)
    assert result["level"] == 0
    assert not result["comparable"]
    assert result["conditional_fields"] == ["optimizer.eps"]
    assert q.evidence_coverage(b)["assumed"] > 0
    b = complete()
    b["run_config"]["trainer.version"] = cell("other")
    assert q.stackdiff(a, b)["semantic_review_required"] == ["trainer.version"]


def test_framework_filter_applies_in_aggregate_mode(monkeypatch, capsys):
    a, b = complete(), complete()
    a["framework"]["name"] = "A"
    b["framework"]["name"] = "B"
    for rec in (a, b):
        rec["variant_deltas_applied"] = [
            {"delta_id": "delta_dapo", "component": "clip", "status": "implemented"}
        ]
    monkeypatch.setattr(q, "load", lambda *a, **kw: {"a": a_record, "b": b_record})
    a_record, b_record = a, b
    q.cmd_implementations(SimpleNamespace(delta=None, framework="A", status=None))
    assert capsys.readouterr().out.rstrip().endswith("1")


def source(tmp_path, text):
    path = tmp_path / "source.py"
    path.write_text(text)
    return g.Src(str(path))


def test_numeric_capture_is_parsed_or_asserted(tmp_path):
    src = source(tmp_path, "x = T.AdamParams(beta1=0.1, beta2=0.2)\n")
    assert src.optimizer_betas()["value"] == [0.1, 0.2]
    with pytest.raises(SystemExit, match="mismatch"):
        src.cell(r"beta1=([\d.]+), beta2=([\d.]+)", [0.9, 0.95])
    src = source(tmp_path, 'parser.add_argument("--save-every", type=int, default=17)\n')
    with pytest.raises(ValueError, match="derive descriptions"):
        src.cell(r"default=(\d+)", "every 5 steps")
    assert src.cell(r"default=(\d+)", cast=int, status="default")["status"] == "unknown"


@pytest.mark.parametrize(
    "change", ["else -100.0", "parse(t) != gold", "2.0 if", "rewards = [v * 2 for v in rewards]"]
)
def test_reward_fingerprint_covers_branches_comparison_and_transforms(tmp_path, change):
    text = "def parse(t):\n    return int(t)\nrewards = [1.0 if parse(t) == gold else 0.0 for t in texts]\n"
    base = source(tmp_path, text).reward_fingerprint()["value"]
    if change.startswith("rewards ="):
        modified = text + change + "\n"
        assert source(tmp_path, modified).reward_fingerprint()["value"] != base
    else:
        modified = (
            text.replace("else 0.0", change)
            if change.startswith("else")
            else text.replace("parse(t) == gold", change)
            if change.startswith("parse")
            else text.replace("1.0 if", change)
        )
        assert source(tmp_path, modified).reward_fingerprint()["value"] != base


def spec():
    return {
        "run_config": {"loss.clip": cell("none")},
        "sources": [],
        "label": "grpo",
        "framework": {"name": "fixture", "openness": "open"},
        "model": None,
        "task": None,
        "seeds": None,
        "telemetry": (None, None, None),
        "outcomes": {},
        "notes": "fixture",
    }


def test_owned_projection_drops_retained_and_edited_outcomes():
    old = complete()
    old["outcomes"] = {"heldout_accuracy": 0.99}
    new = g.build("fixture", spec(), old)
    assert "heldout_accuracy" not in new["outcomes"]
    new["outcomes"]["heldout_accuracy"] = 0.999
    assert g.build("fixture", spec(), new) != new
    assert set(new["run_config"]) == set(q.FIELDS)


def test_dryrun_has_no_execution_fallback():
    sp = spec()
    sp["run_config"] = {"execution.mode": cell("dryrun", "logged")}
    sp["intended_config"] = {"loss.clip": cell("none")}
    old = complete()
    old["model"] = "old"
    old["seeds"] = [42]
    old["outcomes"] = {"reward": 1}
    new = g.build("fixture", sp, old)
    assert new["model"] is None
    assert new["seeds"] is None
    assert not q.is_known(q.field_view(new)["loss.clip"])
    assert "reward" not in new["outcomes"]
    assert new["intended_config"]["loss.clip"]["value"] == "none"


def test_disabled_clipping_and_unknown_decontamination_are_preserved():
    new = g.build("fixture", spec(), complete())
    lf = new["min_report"]["loss_form"]
    assert lf["clipping_applied"] is False
    assert lf["clip_eps_low"] is None
    assert new["min_report"]["decontamination"] == {
        "performed": None,
        "parser_robustness_probe": None,
    }
    legacy = copy.deepcopy(new)
    legacy.pop("run_config")
    assert q.field_view(legacy)["loss.clip"]["value"] == "none"
    assert (
        new["outcomes"]["coverage"]["reported_leaf_coverage_v3"]
        == sum(q.badge(new)[1].values()) / 7
    )
    q.validate_record(new, json.loads((REG / "schema.json").read_text()))


def test_unknown_checkpoint_variant_stays_unknown():
    assert g.variant_rule("Qwen/Qwen3-8B")["status"] == "unknown"
    assert g.variant_rule("foo/database")["status"] == "unknown"


def test_h2h_recomputes_from_distinct_run_artifacts():
    runs = [
        SimpleNamespace(
            json={
                "status": "completed",
                "seed": seed,
                "steps": 10,
                "reward_trace": [value] * 10,
                "step_log": [{"step": i, "reward": value, "zvf": value} for i in range(1, 11)],
            },
            artifact=f"run-{seed}-digest",
        )
        for seed, value in [(1, 0.1), (2, 0.3), (3, 0.5)]
    ]
    out, ev = g.h2h_outcomes(runs)
    assert out["mean_last10_train_reward"] == pytest.approx(0.3)
    assert len(ev["mean_zvf"]["input_artifacts"]) == 3
    runs[2].json["seed"] = 1
    with pytest.raises(ValueError, match="distinct"):
        g.h2h_outcomes(runs)


def test_cache_requires_current_input_and_output_binding(registry, tmp_path):
    target = registry.parents[1] / "producer.py"
    target.write_text("version=1")
    cache = registry / "cache.json"
    cache.write_text("{}")
    args = SimpleNamespace(refresh=False)
    with pytest.raises(ValueError, match="unverified/stale"):
        q.audit_text(cache, target, args)
    manifest = q.audit_input_manifest(target)
    receipt = {
        "version": "registry-report-cache-v1",
        "input_manifest": manifest,
        "input_sha256": q.registry_manifest_sha256(manifest),
        "output_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    cache.with_name("cache.json.receipt.json").write_text(json.dumps(receipt))
    assert q.audit_text(cache, target, args) == "{}"
    cache.write_text('{"edited":true}')
    with pytest.raises(ValueError, match="unverified/stale"):
        q.audit_text(cache, target, args)


def test_validator_cannot_start_fails_closed(registry, monkeypatch):
    def unavailable(*a, **kw):
        raise OSError("not executable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 1


def test_direct_comparison_rejects_duplicate_declarations():
    a = complete()
    a["variant_deltas_applied"] = [
        {"delta_id": "delta_dapo", "component": "clip", "status": "unknown"}
    ] * 2
    with pytest.raises(ValueError, match="duplicate"):
        q.stackdiff(a, complete())


def test_all_unknown_cli_gate_cannot_pass(monkeypatch):
    a = complete()
    a["run_config"] = {f: cell(None, "unknown") for f in q.FIELDS}
    monkeypatch.setattr(
        q, "load", lambda *a, **kw: {"a": copy.deepcopy(record), "b": copy.deepcopy(record)}
    )
    record = a
    assert q.cmd_stackdiff(SimpleNamespace(all=True, generated=False, exit_level=True)) == 6


def test_every_current_generated_projection_is_valid_and_idempotent():
    schema = json.loads((REG / "schema.json").read_text())
    specs = {**g.fc_family(), **g.h2h_family(), **g.e3_family()}
    assert len(specs) == 12
    for eid, sp in specs.items():
        old = json.loads((REG / "entries" / f"{eid}.json").read_text())
        new = g.build(eid, sp, old)
        q.validate_record(new, schema)
        assert g.build(eid, sp, new) == new
        assert new["provenance"]["execution_binding"].startswith("unknown")


@pytest.mark.parametrize("command", ["coverage", "measured-coverage", "antiherding"])
def test_report_refresh_does_not_mutate_stored_evidence(command):
    # These reporting scripts historically rewrote records as a side effect.
    # Refresh must confine those side effects to a disposable copied registry.
    watched = [
        REG / "schema.json",
        *sorted((REG / "entries").glob("*.json")),
        *sorted((ROOT / "platform_hybrid/experiments/results/p5p8").glob("p6_*audit*")),
        REG / "measured_block_audit.json",
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.is_file()}
    result = subprocess.run(
        [sys.executable, str(REG / "query.py"), command, "--refresh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshot:" in result.stdout
    assert "fresh isolated recomputation" in result.stdout
    assert before == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}


def test_unparsed_semantic_anchor_is_an_inference(tmp_path):
    src = source(tmp_path, "def reward_fn(x):\n    return x\n")
    result = src.cell(r"def reward_fn", "claimed semantics")
    assert result["status"] == "inferred"
    assert "not mechanically extracted" in result["note"]


def test_real_strict_validator_joins_delta_record_ids_not_filenames(tmp_path):
    import shutil

    reg = tmp_path / "platform_hybrid/registry"
    (reg / "entries").mkdir(parents=True)
    for filename in ["query.py", "schema.json", "json_protocol.py"]:
        shutil.copy2(REG / filename, reg / filename)
    target = tmp_path / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py", target)
    stack = json.loads((REG / "entries/tinker_grpo_qwen3-8b_gsm8k.json").read_text())
    stack["variant_deltas_applied"] = [
        {"delta_id": "delta_dapo", "component": "fixture", "status": "unknown"}
    ]
    (reg / "entries/stack.json").write_text(json.dumps(stack))
    delta = json.loads((REG / "entries/delta_dapo.json").read_text())
    delta["id"] = "delta_aero"
    for field in ["measured", "expected_effects", "claim_validation"]:
        delta[field] = []
    (reg / "entries/delta_dapo.json").write_text(json.dumps(delta))
    result = subprocess.run(
        [sys.executable, str(reg / "query.py"), "validate-strict"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "CI GATE: PASS" not in result.stdout
    assert "orphan_delta_id" in result.stdout


def test_producer_replacement_during_run_cannot_pass(registry, monkeypatch):
    def changed(argv, **kwargs):
        Path(argv[1]).write_text('raise RuntimeError("replacement producer")')
        return audit_runner(registry)(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", changed)
    assert q.cmd_validate_strict(SimpleNamespace(include_warn=False)) == 1


def test_reporting_manifest_rejects_external_source_even_when_missing(
    registry, monkeypatch, tmp_path
):
    external = tmp_path / "external.tsv"
    external.write_text("synthetic")
    monkeypatch.setattr(q, "load", lambda: {"r": {"measured": {"source": str(external)}}})
    target = registry.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    with pytest.raises(ValueError, match="external reporting source"):
        q.audit_input_manifest(target)
    external.unlink()
    with pytest.raises(ValueError, match="external reporting source"):
        q.audit_input_manifest(target)


def test_reporting_manifest_binds_copied_runtime_dependencies(registry, monkeypatch):
    monkeypatch.setattr(q, "load", dict)
    for filename in ["query.py", "json_protocol.py"]:
        (registry / filename).write_text("before")
    target = registry.parents[1] / "platform_modal/scripts/p5p8/p6_iter118_strict_validator.py"
    before = q.audit_input_manifest(target)
    (registry / "json_protocol.py").write_text("after")
    assert q.audit_input_manifest(target) != before

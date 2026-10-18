"""Synthetic CPU-only regression tests for the five visible review-18 P0s."""

import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts/p5p8"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def schema_fixture(m, tmp_path, monkeypatch):
    ent = tmp_path / "registry" / "entries"
    ent.mkdir(parents=True)
    schema = {
        "$defs": {
            "nullable_string": {"type": ["string", "null"]},
            "measured_delta": {},
            "variant_delta_record": {"properties": {"citation": {"properties": {}}}},
            "stack_record": {
                "properties": {
                    "variant_deltas_applied": {
                        "items": {"properties": {"status": {"enum": ["implemented"]}}}
                    }
                }
            },
        }
    }
    sch = ent.parent / "schema.json"
    sch.write_text(json.dumps(schema))
    entry = {
        "id": "example",
        "record_type": "stack",
        "variant_deltas_applied": [{"status": "single-seed; same-stack isolation not run"}],
    }
    path = ent / "example.json"
    path.write_text(json.dumps(entry))
    for key, value in dict(
        ROOT=tmp_path, ENT=ent, SCH_PATH=sch, SCH_BAK=ent.parent / "backup.json"
    ).items():
        monkeypatch.setattr(m, key, value)
    return path, sch


def test_schema_dry_run_is_pure_and_apply_reloads(tmp_path, monkeypatch):
    m = load("p6_iter198_schema_drift_bump")
    path, sch = schema_fixture(m, tmp_path, monkeypatch)
    old = (path.read_bytes(), sch.read_bytes())
    monkeypatch.setattr(sys, "argv", ["audit", "--out-dir", str(tmp_path / "dry")])
    assert m.main() == 0
    assert old == (path.read_bytes(), sch.read_bytes())
    summary = json.loads((tmp_path / "dry/p6_iter198_summary.json").read_text())
    assert summary["post_bump"]["valid"] == 1
    assert not summary["applied"]
    calls = []
    original = m.load_entries

    def reload():
        calls.append(True)
        return original()

    monkeypatch.setattr(m, "load_entries", reload)
    monkeypatch.setattr(sys, "argv", ["audit", "--apply", "--out-dir", str(tmp_path / "apply")])
    assert m.main() == 0
    assert len(calls) == 2
    assert (
        json.loads(path.read_text())["variant_deltas_applied"][0]["status"]
        == "single-seed-surrogate"
    )
    assert m.validate_entry(json.loads(path.read_text()), json.loads(sch.read_text())) == (True, [])


def test_missing_validator_is_not_pass(monkeypatch):
    m = load("p6_iter198_schema_drift_bump")
    monkeypatch.setattr(m, "HAVE_JS", False)
    assert m.validate_entry({}, {})[0] is None
    monkeypatch.setattr(sys, "argv", ["audit"])
    with pytest.raises(SystemExit, match="SKIPPED"):
        m.main()


def test_amendment_proposal_never_approves_failed_or_empty_evidence():
    m = load("p6_iter194_contradicts_deepdive")
    checks = dict(h1=True, h2b=True, h3=True, h4=True, h5=True)
    assert m.amendment_proposal(checks, True)["eligible_for_human_review"]
    for key in checks:
        result = m.amendment_proposal(dict(checks, **{key: False}), True)
        assert not result["eligible_for_human_review"]
        assert result["proposed_sign"] is None
    assert not m.amendment_proposal(checks, False)["eligible_for_human_review"]
    assert (
        'cv["verdict"] = "SUPPORTS"'
        not in (SCRIPTS / "p6_iter194_contradicts_deepdive.py").read_text()
    )


def test_146_pairs_ids_not_positions(monkeypatch):
    m = load("p6_iter146_provenance_recompute")
    a = [dict(step=i, x=float(i)) for i in range(30, 40)]
    b = [dict(step=i, x=float(i) + 2) for i in reversed(range(30, 40))]
    monkeypatch.setattr(m, "load_n2_metrics", lambda: dict(grpo=a, aero=b))
    assert m.recompute_n2_panel("n2", "aero", "x", 10)[0] == 2
    b.pop()
    with pytest.raises(ValueError, match="Incomplete declared N2 last\\-K window"):
        m.recompute_n2_panel("n2", "aero", "x", 10)
    seeds = dict(
        grpo=[dict(seed="0", x=1), dict(seed="1", x=2)],
        aero=[dict(seed="1", x=4), dict(seed="0", x=3)],
    )
    monkeypatch.setattr(m, "load_zvf130_risk_index", lambda: seeds)
    assert m.recompute_zvf130_risk_metric("x", "aero")[0] == 2
    seeds["aero"].append(dict(seed="1", x=5))
    with pytest.raises(ValueError, match="Duplicate observation identity"):
        m.recompute_zvf130_risk_metric("x", "aero")


def test_190_pairs_reordered_steps_and_rejects_duplicate_missing():
    m = load("p6_iter190_measured_vs_claimed")
    rows = [
        dict(method=method, step=i, x=i + offset)
        for method, offset in [("grpo", 0), ("aero", 2)]
        for i in range(30, 40)
    ]
    vals = m.n2_last10_by_method_metric(rows[::-1], "x")
    assert m.paired_step_bootstrap_pct(vals["grpo"], vals["aero"], n_boot=20) == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="Duplicate N2 step identity"):
        m.n2_last10_by_method_metric(rows + [rows[0]], "x")
    with pytest.raises(ValueError, match="Incomplete declared N2 step window"):
        m.n2_last10_by_method_metric(rows[1:], "x")
    with pytest.raises(ValueError, match="Paired comparisons require explicit identity mappings"):
        m.paired_step_bootstrap_pct([1, 2], [3, 4])


def test_metric_specific_sd_and_missing_n():
    m = load("p6_iter190_measured_vs_claimed")
    panel = dict(grpo=dict(mag_mean=1, zvf_risk_sd=99), aero=dict(mag_mean=3, zvf_risk_sd=99))
    assert m.zvf_risk_delta(panel, "grpo", "aero", "mag_mean") == (2.0, None, None, None)
    for row in panel.values():
        row.update(mag_sd=1, n_seeds=4)
    d, lo, hi, n = m.zvf_risk_delta(panel, "grpo", "aero", "mag_mean")
    assert d == 2
    assert hi - d == pytest.approx(1.96 * (0.5**0.5))
    assert n == 4
    panel["aero"]["mag_sd"] = -1
    with pytest.raises(ValueError, match="Invalid metric\\-specific uncertainty"):
        m.zvf_risk_delta(panel, "grpo", "aero", "mag_mean")


def test_190_descriptive_uncertainty_survives_output_pipeline(tmp_path, monkeypatch):
    m = load("p6_iter190_measured_vs_claimed")
    metrics = ["zvf", "reward_mean", "pcd", "mean_len", "cv_len", "frac_all_zero", "frac_all_one"]
    rows = [
        dict(method=method, step=step, **{metric: step + offset for metric in metrics})
        for method, offset in [("grpo", 0), ("aero", 2)]
        for step in range(30, 40)
    ]
    panel = {
        method: dict(
            method=method,
            zvf_risk_mean=offset,
            mag_mean=offset,
            csd_mean=offset,
            drift_mean=offset,
            zvf_risk_sd=99,
        )
        for method, offset in [("grpo", 0), ("aero", 2)]
    }
    ent = tmp_path / "entries"
    ent.mkdir()
    entry = {
        "id": "delta_aero",
        "name": "AERO",
        "expected_effects": [
            {"metric": "mag_mean", "panel": "zvf130_5seed", "predicted_sign": ">=0"}
        ],
    }
    fp = ent / "delta_aero.json"
    fp.write_text(json.dumps(entry))
    old = fp.read_bytes()
    monkeypatch.setattr(m, "load_n2", lambda: rows)
    monkeypatch.setattr(m, "zvfrisk_method_table", lambda: panel)
    monkeypatch.setattr(m, "ENT", ent)
    monkeypatch.setattr(m, "RES", tmp_path)
    m.main()
    import csv

    out = list(
        csv.DictReader((tmp_path / "p6_iter190_expected_vs_recomputed.tsv").open(), delimiter="\t")
    )
    assert out[0]["verdict"] == "DESCRIPTIVE_ONLY"
    assert out[0]["recomputed_ci_low"] == ""
    assert fp.read_bytes() == old

"""Synthetic-only regressions for review17 P5 protocol v2."""

import importlib
import json
import math
from pathlib import Path
import random
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "platform_modal/scripts/p5p8"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))


def module(name):
    return importlib.import_module(name)


def test_omega_and_cohens_d():
    m = module("p5_n2_unpacking")
    rows = [{"g": g, "y": v} for g, vs in [("a", [1, 2, 3]), ("b", [4, 5, 6])] for v in vs]
    assert m.axis_variance_fraction(rows, "g", "y")["omega2"] == pytest.approx(0.6756756757)
    assert m.axis_variance_fraction([{"g": "a", "y": 1}, {"g": "b", "y": 2}], "g", "y") is None
    m = module("p5_iter89_n2_unpacking_bootstrap")
    assert m.cohens_d([1, 2], [3, 4]) == pytest.approx(-2.8284271247)
    assert math.isnan(m.cohens_d([1, 1], [3, 3]))


def test_joint_ols_and_ratio_invalid():
    m = module("p5_review17_stats")
    rows = [{"a": a, "b": b} for a in [0, 1] for b in [0, 1]]
    assert m.additive_r2(rows, [r["a"] + r["b"] for r in rows], ["a", "b"]) == pytest.approx(1)
    assert m.additive_r2([{"a": 0, "b": 0}, {"a": 1, "b": 1}], [0, 1], ["a", "b"]) == pytest.approx(
        1
    )
    assert all(math.isnan(x) for x in m.ratio_interval([2, 3], 3))
    assert json.dumps(m.json_safe({"x": float("inf")}), allow_nan=False) == '{"x": null}'


def test_fraction_residual():
    m = module("p5_yield_residual_axes")
    assert m.delta_div({"mean_reward": 0.5, "G": 2, "zvf": 0.5}) == 0
    with pytest.raises(ValueError, match="mean_reward must be a finite fraction in \\[0,1\\]"):
        m.delta_div({"mean_reward": 50, "G": 2, "zvf": 0.5})


def test_degenerate_bootstrap_no_index_error():
    m = module("p5_stack_conditioning_eta2")
    rows = [
        dict(method="a", step=0, prompt=0, cell_mean=0),
        dict(method="b", step=0, prompt=1, cell_mean=1),
    ]
    result = m.boot_eta2(rows, "method", n_boot=100)
    assert result["n_invalid"] > 0
    assert result["status"] == "DEGENERATE"
    assert math.isnan(result["hi"])


def test_numeric_mve_rejects_nonfinite():
    m = module("p5_mve_empirical")
    manifest = {"cell_id": "x"}
    cells = {"x": {field: "nan" for field in m.MVE_FIELDS}}
    assert m.score_one(manifest, cells, True)["total"] == 0
    cells = {"x": {field: "inf" for field in m.MVE_FIELDS}}
    assert m.score_one(manifest, cells, True)["total"] == 0


def test_badges_share_schema():
    a = module("p5_mve_empirical")
    b = module("p5_manifest_outcome_coupling")
    manifest = dict(
        loss_form="grpo",
        ref_policy_kl="no-kl",
        sampler_backend_precision="vllm",
        per_step_zvf_path="x",
        group_size_schedule="fixed-G=8",
        heldout_split="test",
        decontamination_notes="checked",
    )
    assert a.score_one(manifest, {}, False)["total"] == b.badge_for(manifest)[0]
    assert math.isnan(b.permutation_p(1, [1, 2], [3, 4]))


def test_expected_cells_never_disappear(tmp_path, monkeypatch):
    m = module("p5_iter185_v25_cross_corpus")
    monkeypatch.setattr(m, "MEGA", tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "cells.tsv").write_text(
        "cell_id\tzvf\tmean_reward\tn_groups\n" + "".join(f"c{i}\t1\t0\t2\n" for i in range(10))
    )
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"reward_vectors": [[0] * 8, [0] * 8]}))
    (tmp_path / "manifests/c0.json").write_text(
        json.dumps({"cell_id": "c0", "per_step_zvf_path": str(raw)})
    )
    checks = m.audit_value_correctness_mega()
    assert len(checks) == 50
    zvf = [row for row in checks if row[0] == "zvf"]
    assert len(zvf) == 10
    assert sum(row[5] for row in zvf) == 1


def test_perstep_identity_and_binary_validation(tmp_path, monkeypatch):
    m = module("p5p8_iter96_per_step_yield_axes")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setattr(m, "TENSORS_DIR", a)
    monkeypatch.setattr(m, "TENSORS_DIR_RESUME", b)
    row = dict(step=0, rewards=[[0] * 8, [1] * 8])
    for path in [a, b]:
        (path / "grpo_s0_tensors.jsonl").write_text(json.dumps(row) + "\n")
    assert len(m.load_step_records()) == 1
    row["rewards"][0][0] = 0.5
    (b / "grpo_s0_tensors.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Binary"):
        m.load_step_records()
    simulated = m.simulate_step(dict(k_per_prompt=[1, 2, 3, 4]), random.Random(2))
    assert set(simulated) >= {"item_14_step", "p_hat"}


def test_n2_channel_retains_length(tmp_path, monkeypatch):
    m = module("p5_iter201_task_stratified_ratio")
    file = tmp_path / "metrics.tsv"
    file.write_text("method\tzvf\treward_mean\tmean_len\na\t.5\t.2\t200\n")
    monkeypatch.setattr(m, "N2_METRICS", file)
    assert m.load_n2_metrics()[0]["mean_completion_len"] == 200


def test_identical_categorical_encoding():
    m = module("p5_manifest_self_sufficiency")
    rows = [{"G": g, "manifest": {"group_size_schedule": f"fixed-G={g}"}} for g in [2, 4, 8]]
    x, _ = m.encode(rows, ["G"])
    y, _ = m.encode(rows, ["group_size_schedule"], "manifest")
    assert m.fit_r2(x, [0, 1, 0]) == pytest.approx(m.fit_r2(y, [0, 1, 0]))


def test_iter197_synthetic_main(tmp_path, monkeypatch):
    m = module("p5_iter197_ratio_sensitivity")
    monkeypatch.setattr(m, "B", 20)
    monkeypatch.setattr(m, "OUT", str(tmp_path))
    n2 = tmp_path / "n2.tsv"
    mega = tmp_path / "mega.tsv"
    n2.write_text(
        "method\tzvf\treward_mean\tmean_len\n"
        + "".join(
            f"{a}\t{0.2 + 0.1 * i + 0.05 * (a == 'b')}\t{0.1 * i + 0.1 * (a == 'b')}\t{10 + i + 2 * (a == 'b')}\n"
            for a in ["a", "b"]
            for i in range(5)
        )
    )
    mega.write_text(
        "model_family\ttask_slice\tG\ttemperature\tseed\tzvf\tmean_reward\tmean_completion_len\n"
        + "".join(
            f"{a}\t{b}\t{2 + 2 * c}\t{0.5 + 0.5 * c}\t{i}\t{0.1 + 0.2 * a + 0.1 * b + 0.01 * i}\t{0.2 + 0.1 * a + 0.2 * b + 0.01 * i}\t{10 + a + 3 * b + i}\n"
            for a in [0, 1]
            for b in [0, 1]
            for c in [0, 1]
            for i in [0, 1]
        )
    )
    monkeypatch.setattr(m, "N2", str(n2))
    monkeypatch.setattr(m, "MEGA", str(mega))
    m.main()
    result = json.loads((tmp_path / "p5_iter197_summary.json").read_text())
    assert result["analysis_protocol"] == "review17-p5-v2"
    assert len(result["paired"]) == 15


def test_iter185_unknown_lengths_and_count_scope(tmp_path, monkeypatch):
    m = module("p5_iter185_v25_cross_corpus")
    monkeypatch.setattr(m, "N2", tmp_path)
    row = dict(
        reward_mean=0.5,
        zvf=0.5,
        pcd=0.5,
        mean_len=10,
        group_size=8,
        prompt_indices=[1, 2],
        rewards=[[0] * 8, [1] * 8],
    )
    (tmp_path / "grpo_s0_tensors.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    result = m.load_n2_v25_manifests()[0]
    assert result["n_groups"] == 4
    assert "sampled_tokens" not in result
    assert "std_completion_len" not in result
    row["completion_lengths"] = [[10] * 8, [10] * 8]
    (tmp_path / "grpo_s0_tensors.jsonl").write_text(json.dumps(row) + "\n")
    result = m.load_n2_v25_manifests()[0]
    assert result["sampled_tokens"] == 160
    assert result["std_completion_len"] == 0
    assert not m.valid_field({"mean_reward": "nan"}, "mean_reward")


def test_all_invalid_bootstrap_reports_counts():
    m = module("p5_stack_conditioning_eta2")
    rows = [dict(method="a", step=0, prompt=0, cell_mean=1)]
    result = m.boot_eta2(rows, "method", n_boot=20)
    assert result["n_boot"] == 0
    assert result["n_invalid"] == 20
    assert result["status"] == "DEGENERATE"

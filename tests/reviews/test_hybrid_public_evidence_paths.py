"""Disk-loader-to-report evidence regressions; fixtures are offline synthetic data.

Positive fixtures exercise protocol validation only, not empirical observations.
No historical results, services, model code or generated candidate code are used.
"""

import copy
import csv
import json
import math

import pytest

from platform_hybrid.experiments import base_instruct_paired as bi
from platform_hybrid.experiments import p0_gsm8k_paired as p0
from platform_hybrid.experiments import stratified_heldout as sh
from platform_hybrid.experiments import zvf_lagged_regression as zvf
from platform_hybrid.experiments.research_contracts import EvidenceUnavailable


def receipt(run_id, **fields):
    return {
        "execution_kind": "empirical", "measured": True, "claim_eligible": True,
        "status": "completed", "run_id": run_id, "receipt_id": f"fixture:{run_id}",
        **fields,
    }


def write(path, data):
    path.write_text(json.dumps(data))
    return path


INVALID_PROVENANCE = [
    {"execution_kind": "simulation"}, {"measured": False}, {"claim_eligible": False},
    {"status": "service_error"}, {"receipt_id": None}, {"run_id": None},
]


def setup_stratified(monkeypatch, tmp_path, evaluation, *, root=None, master_root=None):
    master = write(tmp_path / "master.json", {
        **(master_root or {}), "experiments": [{
            "model": "Qwen/Qwen3-8B-Base", "experiment_id": "experiment",
            "last10_avg": 0.5, "reward_trace": [0.1, 0.9],
        }],
    })
    heldout = write(tmp_path / "heldout.json", {**(root or {}), "results": [evaluation]})
    monkeypatch.setattr(sh, "MASTER", master)
    monkeypatch.setattr(sh, "HELDOUT", heldout)


def stratified_evaluation(**extra):
    return receipt("evaluation", experiment_id="experiment", checkpoint_id="checkpoint",
                   heldout_accuracy=0.99, **extra)


@pytest.mark.parametrize("invalid", INVALID_PROVENANCE + [{"legacy": True}])
def test_stratified_public_report_rejects_unverified_accuracy(monkeypatch, tmp_path, invalid):
    evaluation = stratified_evaluation()
    if invalid == {"legacy": True}:
        evaluation = {k: evaluation[k] for k in ("experiment_id", "checkpoint_id", "heldout_accuracy")}
    else:
        evaluation.update(invalid)
    setup_stratified(monkeypatch, tmp_path, evaluation)
    loaded = sh.load_heldout_map()["experiment"]
    assert loaded == evaluation
    assert "heldout_n" not in loaded
    rows = sh.run(["p1"], 42)
    row = rows[0]
    assert row["mean_acc"] == row["rank"] == ""
    assert row["n_measured"] == 0
    assert row["source"] == "no_eligible_heldout_measurements"
    sh.write_tsv(rows, tmp_path / "report.tsv")
    reported = list(csv.DictReader((tmp_path / "report.tsv").open(), delimiter="\t"))[0]
    assert reported["n_measured"] == "0" and reported["mean_acc"] == ""


@pytest.mark.parametrize("container", ["root", "master_root"])
@pytest.mark.parametrize("invalid", INVALID_PROVENANCE[:4])
def test_stratified_container_veto_cannot_be_lost(monkeypatch, tmp_path, container, invalid):
    setup_stratified(monkeypatch, tmp_path, stratified_evaluation(), **{container: invalid})
    assert sh.run(["p1"], 42)[0]["n_measured"] == 0


def test_stratified_receipt_preserves_checkpoint_and_unknown_count(monkeypatch, tmp_path):
    evaluation = stratified_evaluation()
    setup_stratified(monkeypatch, tmp_path, evaluation)
    loaded = sh.load_heldout_map()
    assert loaded["checkpoint"] == loaded["experiment"] == evaluation
    grouped = sh.group_checkpoints(sh.load_master()["experiments"], loaded)
    saved = next(iter(grouped.values()))[0]
    assert saved["checkpoint_id"] == "checkpoint"
    assert saved["_heldout_n"] is None
    assert saved["_evaluation_receipt"]["receipt_id"] == "fixture:evaluation"
    report = sh.run(["p1"], 42)[0]
    assert report["mean_acc"] == 0.99 and report["n_measured"] == report["rank"] == 1


def test_stratified_missing_checkpoint_identity_never_invented(monkeypatch, tmp_path):
    evaluation = stratified_evaluation()
    del evaluation["checkpoint_id"]
    setup_stratified(monkeypatch, tmp_path, evaluation)
    assert sh.run(["p1"], 42)[0]["n_measured"] == 0


def setup_base_instruct(monkeypatch, tmp_path, *, invalid=None, container=False):
    monkeypatch.setattr(bi, "REPO", tmp_path)
    monkeypatch.setattr(bi, "OUT_TSV", tmp_path / "report.tsv")
    def observation(run_id, **fields):
        result = receipt(run_id, config={"model": "Qwen/Qwen3-8B"}, **fields)
        if invalid and not container:
            result.update(invalid)
        return result
    root = invalid if container else {}
    monkeypatch.setattr(bi, "MASTER", write(tmp_path / "master.json", {
        **root, "experiments": [observation("training", model="Qwen/Qwen3-8B", seed=42,
                                           task="gsm8k", group_size=8, lr=1e-5,
                                           reward_trace=[0.1, 0.9])]}))
    monkeypatch.setattr(bi, "HELDOUT_TOP10", write(tmp_path / "top10.json", {
        **root, "results": [observation("base-training", model="Qwen/Qwen3-8B-Base",
                                      heldout_accuracy=0.9)]}))
    control = observation("control", summary={"accuracy": 0.1})
    if container:
        control.update(invalid)
    monkeypatch.setattr(bi, "BASE_CONTROL_200", write(tmp_path / "control.json", control))
    seeds = []
    for seed in range(5):
        result = observation(f"seed-{seed}", seed=seed, summary={"accuracy": 0.9})
        if container:
            result.update(invalid)
        seeds.append(write(tmp_path / f"seed-{seed}.json", result))
    monkeypatch.setattr(bi, "HELDOUT_5SEED_GLOB", seeds)


@pytest.mark.parametrize("invalid", INVALID_PROVENANCE)
def test_base_instruct_public_report_rejects_ineligible_control_and_seeds(monkeypatch, tmp_path, invalid):
    setup_base_instruct(monkeypatch, tmp_path, invalid=invalid)
    rows = bi.build_rows()
    for row in rows:
        assert row["n_seeds"] == 0
        assert row["heldout_post_rl"] is None and row["heldout_pre_rl"] is None
        assert row["delta_heldout"] is None and math.isnan(row["paired_p_bh"])
    bi.write_tsv(rows)
    assert "0.8000" not in bi.OUT_TSV.read_text()


@pytest.mark.parametrize("invalid", INVALID_PROVENANCE[:4])
def test_base_instruct_collection_root_veto_survives_public_loaders(monkeypatch, tmp_path, invalid):
    setup_base_instruct(monkeypatch, tmp_path, invalid=invalid, container=True)
    for field, value in invalid.items():
        assert bi.load_master()[0][field] == value
        assert bi.load_heldout_top10()[0][field] == value
    assert all(row["n_seeds"] == 0 for row in bi.build_rows())


def test_base_instruct_valid_receipts_keep_descriptive_difference_without_fake_pairs(monkeypatch, tmp_path):
    setup_base_instruct(monkeypatch, tmp_path)
    row = next(r for r in bi.build_rows() if r["family"] == "Qwen3-8B" and r["ckpt"] == "Inst")
    assert row["n_seeds"] == 5 and row["delta_heldout"] == pytest.approx(0.8)
    assert math.isnan(row["paired_p"]) and math.isnan(row["paired_p_bh"])


def setup_paired(monkeypatch, tmp_path, base, seeds):
    monkeypatch.setattr(p0, "BASE_FILE", write(tmp_path / "base.json", base))
    monkeypatch.setattr(p0, "SEED_FILES", [write(tmp_path / f"seed-{i}.json", seed) for i, seed in enumerate(seeds)])
    monkeypatch.setattr(p0, "OUT_DIR", tmp_path / "output")
    monkeypatch.setattr(p0, "OUT_JSON", tmp_path / "output/report.json")
    monkeypatch.setattr(p0, "OUT_MD", tmp_path / "output/report.md")


@pytest.mark.parametrize("invalid", INVALID_PROVENANCE + [{"legacy": True}])
@pytest.mark.parametrize("invalid_arm", ["control", "seed"])
def test_paired_public_main_rejects_ineligible_root_before_report(monkeypatch, tmp_path, invalid, invalid_arm):
    base = receipt("base", examples=[{"idx": 0, "status": "correct"}])
    seed = receipt("seed", examples=[{"idx": 0, "status": "incorrect"}])
    bad = base if invalid_arm == "control" else seed
    if invalid == {"legacy": True}:
        for key in list(bad):
            if key != "examples":
                del bad[key]
    else:
        bad.update(invalid)
    setup_paired(monkeypatch, tmp_path, base, [seed])
    with pytest.raises(EvidenceUnavailable):
        p0.main()
    assert not p0.OUT_JSON.exists() and not p0.OUT_DIR.exists()


def test_paired_public_main_preserves_failure_status_and_excludes_ids_across_every_arm(monkeypatch, tmp_path):
    base = receipt("base", examples=[
        {"idx": 0, "status": "correct"}, {"idx": 1, "status": "incorrect"},
        {"idx": 2, "status": "correct"}, {"idx": 3, "status": "correct"},
        {"idx": 4, "status": "correct"}, {"idx": 5, "status": "correct"},
    ])
    seed = receipt("seed", examples=[
        {"idx": 0, "status": "incorrect"}, {"idx": 1, "status": "correct"},
        {"idx": 2, "status": "timeout"}, {"idx": 3, "status": "service_error"},
        {"idx": 4, "status": "correct", "measured": False},
    ])
    other = receipt("other", examples=copy.deepcopy(base["examples"]))
    setup_paired(monkeypatch, tmp_path, base, [seed, other])
    observations = p0.load_passfail(p0.SEED_FILES[0])
    assert dict(observations)[1] == 1
    assert all(dict(observations)[i] is None for i in [2, 3, 4])
    assert observations.statuses[2] == "timeout"
    with pytest.raises(ValueError, match="Unmeasured"):
        p0.align(p0.load_passfail(p0.BASE_FILE), observations, [2])
    p0.main()
    report = json.loads(p0.OUT_JSON.read_text())
    assert report["prompt_order"] == [0, 1]
    assert report["excluded_base_prompt_ids"] == [2, 3, 4, 5]
    assert all(row["n"] == 2 for row in report["per_seed"])
    assert report["per_seed"][0]["base_acc"] == report["per_seed"][0]["grpo_acc"] == 0.5
    exclusions = report["measurement_exclusions"]
    first = exclusions[str(p0.SEED_FILES[0])]
    assert first["2"]["status"] == "timeout" and first["2"]["measured"] is False
    assert first["3"]["status"] == "service_error"
    assert first["4"]["status"] == "correct" and first["4"]["reason"] == "unmeasured_outcome"
    assert first["5"]["status"] == "missing_prompt"
    assert exclusions[str(p0.BASE_FILE)]["2"]["reason"] == "excluded_in_another_arm"
    assert "Excluded 4 prompt IDs" in p0.OUT_MD.read_text()


def test_paired_no_measured_intersection_never_reports_accuracy(monkeypatch, tmp_path):
    setup_paired(monkeypatch, tmp_path,
                 receipt("base", examples=[{"idx": 0, "status": "correct"}]),
                 [receipt("seed", examples=[{"idx": 0, "status": "timeout"}])])
    with pytest.raises(EvidenceUnavailable, match="No common measured"):
        p0.main()
    assert not p0.OUT_JSON.exists()


def test_lagged_loader_deduplicates_copies_but_keeps_seed_config_and_attempt(monkeypatch, tmp_path):
    record = {
        "model": "fixture", "task": "gsm8k", "seed": 42, "config": {"lr": 1e-5},
        "step_log": [{"step": i, "reward": 0.5, "zvf": 0.2, "gu": 0.8} for i in range(6)],
    }
    master = write(tmp_path / "master.json", {"experiments": [record, record]})
    consolidated = write(tmp_path / "consolidated.json", {"nested": {"copies": [record]}})
    monkeypatch.setattr(zvf, "SOURCE_FILES", [master, consolidated])
    runs = zvf.load_runs()
    assert len(runs) == 1
    stable_key = runs[0].key
    variants = [{**record, "seed": 137}, {**record, "config": {"lr": 2e-5}},
                {**record, "attempt_id": "different"}]
    write(consolidated, {"copies": [record, *variants]})
    assert len(zvf.load_runs()) == 4
    moved = write(tmp_path / "moved.json", [record])
    monkeypatch.setattr(zvf, "SOURCE_FILES", [moved])
    assert zvf.load_runs()[0].key == stable_key
    with pytest.raises(EvidenceUnavailable):
        zvf.main()


def test_training_receipt_cannot_supply_missing_evaluation_receipt(monkeypatch, tmp_path):
    master = write(tmp_path / "master.json", {"experiments": [receipt(
        "training", model="Qwen/Qwen3-8B-Base", experiment_id="experiment",
        checkpoints=[{"checkpoint_id": "checkpoint", "evaluation": {
            "checkpoint_id": "checkpoint", "heldout_accuracy": 0.99}}],
    )]})
    monkeypatch.setattr(sh, "MASTER", master)
    monkeypatch.setattr(sh, "HELDOUT", tmp_path / "absent.json")
    assert sh.run(["p1"], 42)[0]["n_measured"] == 0


@pytest.mark.parametrize("status", ["timeout", "missing", "service_error", "error", None, "unknown"])
def test_paired_failed_prompt_is_excluded_instead_of_graded_wrong(monkeypatch, tmp_path, status):
    setup_paired(monkeypatch, tmp_path,
                 receipt("base", examples=[{"idx": 0, "status": "correct"}, {"idx": 1, "status": "correct"}]),
                 [receipt("seed", examples=[{"idx": 0, "status": "correct"}, {"idx": 1, "status": status}])])
    p0.main()
    report = json.loads(p0.OUT_JSON.read_text())
    assert report["prompt_order"] == [0]
    assert report["per_seed"][0]["grpo_acc"] == 1.0
    assert report["excluded_base_prompt_ids"] == [1]
    assert report["measurement_exclusions"][str(p0.SEED_FILES[0])]["1"]["status"] == (status or "missing_status")

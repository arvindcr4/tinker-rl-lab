"""Full aggregation CLI fixtures bind configuration and planned denominators."""

import hashlib
import json
import sys

import pytest
from platform_hybrid.experiments.webarena import aggregate


def execute(
    tmp_path, monkeypatch, *, mutate=None, subset=False, missing_row=False, duplicate=False
):
    config = dict(model="actual-model", benchmark="webarena", tasks=["a", "b"], seed=42)
    eid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    coverage = [
        dict(
            evaluation_id=eid,
            env_id=task,
            seed=int.from_bytes(hashlib.sha256(f"42:{task}".encode()).digest()[:4], "big"),
        )
        for task in config["tasks"]
    ]
    if subset:
        coverage = coverage[:1]
    rows = [dict(**item, score=1, num_steps=1, valid_action_count=1) for item in coverage]
    if missing_row:
        rows = rows[:1]
    receipts = [
        dict(evaluation_id=eid, effective_config=config, coverage=[item]) for item in coverage
    ]
    if mutate:
        mutate(receipts)
    if duplicate:
        receipts.append(receipts[0])
    inputs = tmp_path / "rows"
    inputs.write_text("\n".join(map(json.dumps, rows)))
    manifest = tmp_path / "manifest"
    manifest.write_text(json.dumps(coverage))
    paths = []
    for i, receipt in enumerate(receipts):
        path = tmp_path / f"receipt{i}"
        path.write_text(json.dumps(receipt))
        paths.append(str(path))
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate",
            "--inputs",
            str(inputs),
            "--receipts",
            *paths,
            "--manifest",
            str(manifest),
            "--out",
            str(output),
        ],
    )
    aggregate.main()
    return json.loads(output.read_text())


def test_full_shards_merge_all_receipt_coverage(tmp_path, monkeypatch):
    result = execute(tmp_path, monkeypatch)
    assert result["completion_status"] == "complete"
    assert len(result["evaluation_receipt"]["coverage"]) == result["overall"]["n"] == 2


def test_changed_model_cannot_reuse_evaluation_id(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="hash/identity"):
        execute(
            tmp_path,
            monkeypatch,
            mutate=lambda rs: rs[0]["effective_config"].update(model="other-model"),
        )


@pytest.mark.parametrize("kwargs", [dict(subset=True), dict(missing_row=True)])
def test_omitted_planned_tasks_never_promote_complete(tmp_path, monkeypatch, kwargs):
    result = execute(tmp_path, monkeypatch, **kwargs)
    assert result["completion_status"] == "coverage_incomplete"


def test_duplicate_shard_coverage_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="duplicate"):
        execute(tmp_path, monkeypatch, duplicate=True)


def test_manifest_cannot_omit_receipt_coverage(tmp_path, monkeypatch):
    # Add a known planned task to a subset receipt, leaving subset manifest unchanged.
    def expand(rs):
        rs[0]["coverage"].append(
            dict(
                evaluation_id=rs[0]["evaluation_id"],
                env_id="b",
                seed=int.from_bytes(hashlib.sha256(b"42:b").digest()[:4], "big"),
            )
        )

    with pytest.raises(ValueError, match="manifest coverage"):
        execute(tmp_path, monkeypatch, subset=True, mutate=expand)


def test_partial_report_cannot_be_published_by_changing_status(tmp_path, monkeypatch):
    from platform_hybrid.experiments.webarena.push_model_card import _render_model_card

    result = execute(tmp_path, monkeypatch, subset=True)
    result["completion_status"] = "complete"
    with pytest.raises(ValueError, match="coverage"):
        _render_model_card(
            base_model="actual-model",
            benchmark="webarena",
            final=result,
            dataset_repo="fixture/data",
            run_id="fixture",
        )

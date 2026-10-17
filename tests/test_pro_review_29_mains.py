"""Synthetic production-entry-point checks. No repository/result-file writes."""

import csv
import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "platform_modal/scripts"))


class Buffer(io.StringIO):
    def close(self):
        pass


def make_trace(
    source="variance_mitigation",
    method="grpo",
    seed="0",
    *,
    group_size=None,
    zvfs=None,
    accuracies=None,
):
    zvfs = [0.0] * 6 if zvfs is None else zvfs
    accuracies = [0.8] * len(zvfs) if accuracies is None else accuracies
    return dict(
        source=source,
        method=method,
        seed=seed,
        group_size=group_size,
        n_steps=len(zvfs),
        steps=[
            dict(step=i, zvf=z, heldout_acc=a) for i, (z, a) in enumerate(zip(zvfs, accuracies))
        ],
    )


def produce(pool, groups=()):
    module = importlib.import_module("zvf_iter78")
    files = {}

    def writer(path, rows, fieldnames=None, **kwargs):
        b = Buffer()
        w = csv.DictWriter(
            b, fieldnames=fieldnames or list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)
        files[Path(path).name] = b

    def open_fake(path, mode="r", **kwargs):
        assert mode == "w"
        b = Buffer()
        files[Path(path).name] = b
        return b

    with (
        patch.object(module, "load_variance_mitigation", return_value=pool),
        patch.object(module, "load_groupsize_sweep", return_value=list(groups)),
        patch.object(module, "load_tinker_gsm8k", return_value=[]),
        patch.object(module, "load_tool_use_anchors", return_value=[]),
        patch.object(module, "write_tsv", writer),
        patch("builtins.open", open_fake),
        redirect_stdout(Buffer()),
    ):
        module.main()
    return {name: b.getvalue() for name, b in files.items()}


def consume(module_name, feature_text):
    module = importlib.import_module(module_name)
    files = {}

    def open_fake(path, mode="r", *args, **kwargs):
        if mode == "r":
            assert path.name == "zvf_iter78_per_step_features.tsv"
            return Buffer(feature_text)
        assert mode == "w"
        b = Buffer()
        files[path.name] = b
        return b

    with (
        patch.object(Path, "mkdir"),
        patch.object(Path, "open", open_fake),
        patch.dict(sys.modules, {"matplotlib": None}),
        redirect_stdout(Buffer()),
    ):
        module.main()
    return {name: b.getvalue() for name, b in files.items()}


def rows(text):
    return list(
        csv.DictReader(
            (line for line in text.splitlines() if not line.startswith("#")), delimiter="\t"
        )
    )


def test_producer_consumers_keep_same_seed_group_runs_distinct():
    files = produce(
        [make_trace()],
        [
            make_trace("groupsize_zvf_sweep", "groupsize", group_size=g, zvfs=[z] * 6)
            for g, z in [(2, 0.0), (4, 1.0)]
        ],
    )
    features = files["zvf_iter78_per_step_features.tsv"]
    assert len({r["trace_id"] for r in rows(features)}) == 3
    hazard = consume("zvf_iter82", features)
    trace_rows = rows(hazard["zvf_iter82_traces.tsv"])
    assert len(trace_rows) == 3
    assert sorted(
        float(r["mean_zvf"]) for r in trace_rows if r["source"] == "groupsize_zvf_sweep"
    ) == [0.0, 1.0]
    decision = consume("zvf_iter86", features)
    oop = rows(decision["zvf_iter86_oop_applied.tsv"])
    assert len(oop) == 2
    assert len({r["trace_id"] for r in oop}) == 2


def test_late_alarm_saves_remaining_steps_and_counts_as_missed():
    files = produce([make_trace(zvfs=[0.0] * 10 + [1.0] * 4, accuracies=[0.8] + [0.0] * 13)])
    out = consume("zvf_iter86", files["zvf_iter78_per_step_features.tsv"])
    savings = rows(out["zvf_iter86_compute_savings.tsv"])
    assert len(savings) == 1
    assert savings[0]["class"] == "late_alarm"
    assert savings[0]["t_stop"] == "10"
    assert savings[0]["t_fail"] == "8"
    assert savings[0]["compute_saved"] == "3"
    summary = {r["key"]: r["value"] for r in rows(out["zvf_iter86_summary.tsv"])}
    assert summary["n_missed_at_optimal"] == "1"
    assert summary["compute_saved_steps_at_optimal"] == "3"


def test_missing_endpoint_is_excluded_with_reason():
    files = produce(
        [make_trace()],
        [make_trace("groupsize_zvf_sweep", "groupsize", group_size=2, accuracies=[None] * 6)],
    )
    assert {r["source"] for r in rows(files["zvf_iter78_per_step_features.tsv"])} == {
        "variance_mitigation"
    }
    excluded = json.loads(files["zvf_iter78_meta.json"])["excluded_traces"]
    assert len(excluded) == 1
    assert "missing per-step heldout" in excluded[0]["reason"]


@pytest.mark.parametrize("name", ["zvf_iter82", "zvf_iter86"])
def test_empty_feature_table_rejected(name):
    with pytest.raises(ValueError, match="no eligible"):
        consume(name, "source\tmethod\tseed\tstep\tprotocol_id\ttrace_id\n")


@pytest.mark.parametrize("name", ["zvf_iter82", "zvf_iter86"])
def test_duplicate_feature_step_rejected(name):
    text = produce([make_trace()])["zvf_iter78_per_step_features.tsv"]
    lines = text.splitlines()
    duplicate = "\n".join([lines[0], lines[1], lines[1], *lines[2:]]) + "\n"
    with pytest.raises(ValueError, match="duplicate or nonconsecutive"):
        consume(name, duplicate)

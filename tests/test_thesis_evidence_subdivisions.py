"""Receipt totals must not conceal impossible individual subdivisions."""

import json
from collections import Counter
from pathlib import Path

import pytest

from tools import check_thesis_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]


def receipt(lane):
    return json.loads((ROOT / evidence.FINISH / lane / "result.json").read_text())


@pytest.mark.parametrize("lane", ["E5", "E6"])
def test_preserved_subdivision_receipts_pass(lane):
    assert evidence.check_replacement(lane, receipt(lane))


def test_e5_rejects_subset_successes_exceeding_its_denominator():
    data = receipt("E5")
    first, second = data["subsets"].values()
    # Preserve the headline total and both recorded subset rates. The first
    # subset now claims seven successes from one task, hidden by the second.
    second["n"] += first["n"] - 1
    first["n"] = 1
    for subset in data["subsets"].values():
        subset["pass1"] = subset["passes"] / subset["n"]
    with pytest.raises(ValueError, match="E5 subset successes exceed"):
        evidence.check_replacement("E5", data)


@pytest.mark.parametrize("first_n", [19, 33, 53])
def test_e6_rejects_impossible_split_outcome_partitions(first_n):
    data = receipt("E6")
    first, second, *_ = data["secondary"]["per_split"].values()
    # The split has 20 successes and 34 pending judgments. Test successes
    # alone, pending alone, and their disjoint union exceeding its denominator.
    # All aggregate n/success/pending totals remain unchanged.
    second["n"] += first["n"] - first_n
    first["n"] = first_n
    with pytest.raises(ValueError, match="E6 split successes and pending exceed"):
        evidence.check_replacement("E6", data)


def test_e6_accepts_split_with_no_remaining_graded_failures():
    data = receipt("E6")
    first, second, *_ = data["secondary"]["per_split"].values()
    first_n = first["success"] + first["judge_pending"]
    second["n"] += first["n"] - first_n
    first["n"] = first_n
    assert evidence.check_replacement("E6", data)


def test_e2_rejects_correct_outcome_on_ungraded_task():
    data = receipt("E2")
    rows = list(data["per_task"].values())
    next(row for row in rows if row["correct"])["status"] = "errored"
    data["n_graded"] -= 1
    data["outcome_breakdown"] = dict(Counter(f"{r['status']}/{r['end']}" for r in rows))
    with pytest.raises(ValueError, match="E2 correct task must be completed"):
        evidence.check_replacement("E2", data)


def test_e9_rejects_success_on_error_task():
    data = receipt("E9")
    next(row for row in data["per_task"] if row["success"])["rc"] = 1
    data["n_graded"] -= 1
    data["n_errors_or_timeouts"] += 1
    with pytest.raises(ValueError, match="E9 successful task has a nonzero return code"):
        evidence.check_replacement("E9", data)


def test_e1_rejects_prior_successes_exceeding_graded_count():
    data = receipt("E1")
    prior = data["prior_waves_separate_not_pooled"]
    prior["graded"] = 1
    prior["errors_patch_apply_or_eval"] = prior["attempted"] - prior["empty_patch"] - 1
    prior["resolved_over_graded"] = prior["resolved"] / prior["graded"]
    with pytest.raises(ValueError, match="E1 partition resolved exceeds graded"):
        evidence.check_replacement("E1", data)


def test_e6_rejects_errors_overlapping_successes():
    data = receipt("E6")
    data["n_errors"] = data["n_graded"]
    with pytest.raises(ValueError, match="E6 errors and successes exceed graded"):
        evidence.check_replacement("E6", data)


@pytest.mark.parametrize("field", ["b_trained_only", "c_base_only"])
def test_paired_discordance_aliases_cannot_contradict_checked_counts(field):
    record = {
        "trained_value": 0.5,
        "base_value": 0.5,
        "difference": 0.0,
        "discordant_b_trained_only": 1,
        "discordant_c_base_only": 1,
        "b_trained_only": 1,
        "c_base_only": 1,
        "mcnemar_exact_p": 1.0,
    }
    assert evidence.paired_metric(record, [1, 0], [0, 1], "difference") is None
    record[field] = 2
    with pytest.raises(ValueError, match="discordance alias"):
        evidence.paired_metric(record, [1, 0], [0, 1], "difference")

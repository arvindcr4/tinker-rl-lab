from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools import check_thesis_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_evidence(tmp_path):
    for relative in evidence.EVIDENCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_current_evidence_passes_with_explicit_scope():
    report = evidence.check_evidence(ROOT)
    assert report["status"] == "PASS", report["errors"]
    assert len(report["required_files"]) == 22
    assert len(report["checks"]) == 21
    assert "not source provenance" in report["scope"]
    assert report["not_checked"]


@pytest.mark.parametrize("change", ["raw", "summary", "replacement", "paired", "env_weighting"])
def test_tampered_arithmetic_fails(copied_evidence, change):
    if change == "raw":
        relative = evidence.M1B_FULL
    elif change == "summary":
        relative = evidence.M1B
    elif change == "replacement":
        relative = f"{evidence.FINISH}/E2/result.json"
    elif change == "paired":
        relative = f"{evidence.PAIRED}/E12/paired.json"
    else:
        relative = f"{evidence.FINISH}/E13/result.json"
    path = copied_evidence / relative
    data = json.loads(path.read_text())
    if change == "raw":
        data["runs"][0]["post_correct"][0] = 1 - data["runs"][0]["post_correct"][0]
    elif change == "summary":
        data["summary"]["grpo_g8"]["heldout_post_mean"] += 0.1
    elif change == "replacement":
        first = next(iter(data["per_task"].values()))
        first["correct"] = not first["correct"]
    elif change == "paired":
        data["mcnemar_exact_p"] = 0.001
    else:
        # Pooling 255 episodes must not replace the mean of six environment means.
        data["score"] = (
            sum(e["progression_pct"] * e["n_planned"] for e in data["per_env"].values())
            / data["n_planned"]
        )
    path.write_text(json.dumps(data))
    report = evidence.check_evidence(copied_evidence)
    assert report["status"] == "FAIL"
    assert report["errors"]


@pytest.mark.parametrize("damage", ["missing", "corrupt", "nan", "duplicate_key"])
def test_missing_or_corrupt_input_fails(copied_evidence, damage):
    path = copied_evidence / evidence.EVIDENCE_FILES[-1]
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(
            {
                "corrupt": "{broken",
                "nan": '{"score": NaN}',
                "duplicate_key": '{"lane": "E14", "lane": "E13"}',
            }[damage]
        )
    report = evidence.check_evidence(copied_evidence)
    assert report["status"] == "FAIL"
    assert evidence.EVIDENCE_FILES[-1] in report["errors"][0]
    assert not report["checks"]  # Never report partial coverage as a full pass.


def test_missing_raw_item_fails(copied_evidence):
    path = copied_evidence / evidence.M1B_FULL
    data = json.loads(path.read_text())
    data["runs"][0]["post_correct"].pop()
    path.write_text(json.dumps(data))
    report = evidence.check_evidence(copied_evidence)
    assert report["status"] == "FAIL"
    assert "200 evaluation items" in report["errors"][0]


def test_paired_balrog_uses_unrounded_items_and_equal_environment_weight():
    path = ROOT / evidence.PAIRED / "E13/paired.json"
    data = json.loads(path.read_text())
    assert evidence.check_paired("E13", data)
    # Rounded receipt is 1.875; per-item recomputation is 1.8752228163993.
    data["difference"] = 1.876
    with pytest.raises(ValueError, match="equal-env paired difference"):
        evidence.check_paired("E13", data)


def test_cli_exit_status_and_json_report(copied_evidence, tmp_path, capsys):
    output = tmp_path / "report.json"
    assert evidence.main(["--root", str(copied_evidence), "--json", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    (copied_evidence / evidence.M1B).unlink()
    assert evidence.main(["--root", str(copied_evidence)]) == 1
    assert '"status": "FAIL"' in capsys.readouterr().out


@pytest.mark.parametrize("location", ["mean_reward", "grpo_zvf"])
def test_nan_exception_is_limited_to_ppo_zvf(copied_evidence, location):
    path = copied_evidence / evidence.M1B_FULL
    data = json.loads(path.read_text())
    step = data["runs"][10 if location == "mean_reward" else 0]["step_log"][0]
    step["mean_reward" if location == "mean_reward" else "zvf"] = float("nan")
    path.write_text(json.dumps(data))
    report = evidence.check_evidence(copied_evidence)
    assert report["status"] == "FAIL"
    assert "non-finite" in report["errors"][0]


def test_omitted_paired_item_fails(copied_evidence):
    path = copied_evidence / evidence.PAIRED / "E14/paired.json"
    data = json.loads(path.read_text())
    data["per_item"].pop(next(iter(data["per_item"])))
    path.write_text(json.dumps(data))
    report = evidence.check_evidence(copied_evidence)
    assert report["status"] == "FAIL"
    assert any("E14" in error and "outcome count" in error for error in report["errors"])


def test_e12_documented_staged_rounding_preserves_raw_delta():
    data = json.loads((ROOT / evidence.PAIRED / "E12/paired.json").read_text())
    checked = evidence.check_paired("E12", data)
    assert checked["recorded_delta"] == -0.0332
    assert checked["raw_paired_delta"] == pytest.approx(-5 / 151)
    assert checked["rounding_rule"] == "round(round(mean(trained), 4) - round(mean(base), 4), 4)"
    assert checked["generator"].endswith("E3/code/paired_finalize.py")
    data["difference_trained_minus_base"] = -0.0331
    with pytest.raises(ValueError, match="staged-rounding paired difference"):
        evidence.check_paired("E12", data)

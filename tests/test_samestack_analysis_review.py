"""Regression tests for the Pro review's pure analysis and canonical cohort.

All run data below are synthetic. No training or historical audit rewrite.
"""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2"


def module(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def statistics():
    return module("stats_lib")


@pytest.fixture(scope="module")
def audit():
    return module("v2_analysis")


@pytest.fixture
def plans(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return module("analysis_plan")


@pytest.mark.parametrize("name", ["v1_recompute", "v2_analysis", "stats_lib", "analysis_plan"])
def test_import_has_no_cli_or_analysis_io(name):
    script = """import sys, pathlib, importlib.util, builtins
sys.path.insert(0, sys.argv[1])
path = pathlib.Path(sys.argv[1]) / (sys.argv[2] + '.py')
sys.argv = ['arbitrary-host', '--invalid-flag']
before = list(sys.path)
spec = importlib.util.spec_from_file_location('tested_module', path)
mod = importlib.util.module_from_spec(spec)
# Ordinary imports may read source; prohibit analysis data I/O entry points.
import json, gzip
json.load = lambda *a, **k: (_ for _ in ()).throw(AssertionError('import read JSON'))
gzip.open = lambda *a, **k: (_ for _ in ()).throw(AssertionError('import opened gzip'))
spec.loader.exec_module(mod)
assert sys.path == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(DIRECTORY), name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "invalid", [[], [1], [float("nan"), 0], [float("inf"), 0], [[1, 2], [3, 4]]]
)
@pytest.mark.parametrize("function", ["signflip", "boot_ci"])
def test_direct_paired_difference_helpers_validate(statistics, invalid, function):
    with pytest.raises(ValueError, match="paired differences"):
        getattr(statistics, function)(invalid)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ({}, {}),
        ({1: 1}, {1: 0}),
        ({1: 0, 2: 1}, {3: 0, 4: 1}),
        ({1: float("inf"), 2: 0}, {1: 0, 2: 0}),
    ],
)
def test_paired_requires_two_finite_shared_seeds(statistics, a, b):
    with pytest.raises(ValueError, match="paired"):
        statistics.paired(a, b)


@pytest.mark.parametrize("effect", [0, 0.02, -0.02, 0.125, -0.125])
def test_constant_variance_and_tost_boundaries(statistics, effect):
    result = statistics.paired(dict.fromkeys(range(3), effect), dict.fromkeys(range(3), 0))
    assert result["p_t"] == (1 if effect == 0 else 0)
    assert result["t_ci"] == result["tost_ci90"] == [effect, effect]
    assert result["tost_p"] == (0.5 if abs(effect) == 0.02 else 0 if abs(effect) < 0.02 else 1)
    assert result["within_margin_unadjusted"] == (abs(effect) < 0.02)
    json.dumps(result, allow_nan=False)


def test_tost_intervals_and_combined_p(statistics):
    d = np.array([-0.01, 0.01, 0.005, -0.004, 0.002])
    result = statistics.paired(dict(enumerate(d)), dict.fromkeys(range(len(d)), 0))
    se = np.std(d, ddof=1) / math.sqrt(len(d))
    assert result["tost_ci90"] == pytest.approx(stats.t.interval(0.9, 4, loc=d.mean(), scale=se))
    assert result["t_ci"] == pytest.approx(stats.t.interval(0.95, 4, loc=d.mean(), scale=se))
    assert result["tost_p"] == max(result["tost_lower_p"], result["tost_upper_p"])
    assert result["positive_5pp_test"]["null_hypothesis"] == "mean >= +0.05"


def test_exact_signflip_resolution_and_tiny_nonzero(statistics):
    assert statistics.signflip([1] * 5) == 0.0625
    assert statistics.signflip([1e-30] * 5) == 0.0625
    assert statistics.signflip([0] * 5) == 1


def test_mcnemar_zero_discordance(statistics):
    assert statistics.mcnemar_exact(0, 0) == 1
    assert statistics.mcnemar_exact(5, 0) == 0.0625
    with pytest.raises(ValueError, match="discordant counts"):
        statistics.mcnemar_exact(-1, 2)


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.inf, math.nan, None, True])
def test_holm_rejects_invalid_probabilities(statistics, bad):
    with pytest.raises(ValueError, match="probability"):
        statistics.holm({"a": 0.02, "bad": bad})


def test_holm_fixed_missing_slot(statistics):
    assert statistics.holm({"a": 0.01, "b": 0.02, "missing": 1}) == {
        "a": 0.03,
        "b": 0.04,
        "missing": 1,
    }


def test_power_includes_lower_tail_and_brackets_n2(statistics):
    sd, n, effect = 0.02, 2, 0.01
    critical = stats.t.ppf(0.975, n - 1)
    noncentral = effect / (sd / math.sqrt(n))
    expected = stats.nct.sf(critical, n - 1, noncentral) + stats.nct.cdf(
        -critical, n - 1, noncentral
    )
    assert statistics.paired_t_power(sd, n, effect) == pytest.approx(expected)
    mde = statistics.mde(0.02, 2)
    assert mde > 0.2  # The old fixed 10*SD bracket failed for n=2.
    assert statistics.paired_t_power(0.02, 2, mde) == pytest.approx(0.8)
    assert statistics.paired_t_power(0.02, 2, 0) == pytest.approx(0.05)
    required = statistics.n_needed(0.02, 0.01)
    assert statistics.paired_t_power(0.02, required, 0.01) >= 0.8
    assert statistics.paired_t_power(0.02, required - 1, 0.01) < 0.8


@pytest.mark.parametrize(
    ("sd", "n", "power", "alpha"),
    [
        (0, 2, 0.8, 0.05),
        (0.02, 1, 0.8, 0.05),
        (0.02, 2.2, 0.8, 0.05),
        (0.02, 2, 1, 0.05),
        (0.02, 2, 0.03, 0.05),
        (math.nan, 2, 0.8, 0.05),
        (0.02, 2, 0.8, 0),
    ],
)
def test_power_invalid_inputs_explicit(statistics, sd, n, power, alpha):
    with pytest.raises(ValueError, match="finite|integer|probability|exceed"):
        statistics.mde(sd, n, power, alpha)


def test_n_needed_failure_explicit(statistics):
    with pytest.raises(ValueError, match="exceeds"):
        statistics.n_needed(0.1, 0.001, max_n=3)
    with pytest.raises(ValueError, match="effect"):
        statistics.n_needed(0.1, 0)


def test_independent_grader_shared_fixtures(audit):
    fixture = json.loads((ROOT / "tests/fixtures/samestack_grading_v2.json").read_text())
    assert audit.GRADING_SPEC_ID == fixture["spec_id"]
    for case in fixture["cases"]:
        assert audit.grade(case["text"], case["gold"])[0] == case["reward"], case
        if case.get("status") == "unparsable":
            assert audit.grade(case["text"], case["gold"])[1] == "unparsable"
            malformed = audit.last_boxed(case["text"])
            assert audit.to_num(malformed) is None
            with pytest.raises(ValueError, match="gold answer"):
                audit.grade(r"\boxed{42}", malformed)


@pytest.mark.parametrize(
    "text", [r"\boxedX{42}", r"\boxed99 then text {42}", r"\boxed12 then {42}"]
)
def test_no_unrelated_brace_extraction(audit, text):
    assert audit.last_boxed(text) is None


def test_exact_number_precision_and_invalids(audit):
    assert audit.to_num("0.10000000000000001") != audit.to_num("0.1")
    assert audit.to_num(r"\frac{1}{2}") == audit.to_num(".5")
    assert audit.to_num("Infinity") is None
    assert audit.to_num("1/0") is None
    assert audit.to_num(r"\frac{1}{0}") is None
    assert audit.to_num("1e10001") is None
    assert audit.to_num("42%") == audit.to_num("42")
    assert audit.to_num("$42%") is None
    assert audit.to_num(r"42\text{seconds 99}") is None


def test_audit_rejects_zip_truncation(audit):
    with pytest.raises(ValueError, match="texts"):
        audit.validate_inputs(
            {
                "gold": ["1", "2"],
                "records": [{"n_eval": 2}],
                "completions": [{"pre": ["x"], "post": ["x"]}],
            }
        )


def test_strict_json_rejects_nan(audit, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="JSON|Non-finite"):
        audit.load_json(path)


def synthetic_cohort(plans, arms=None, seeds=(42, 123)):
    from samestack_fixtures import synthetic_run

    arms = list(plans.ARMS) if arms is None else arms
    records = [synthetic_run(arm=arm, seed=seed, smoke=False)[0] for arm in arms for seed in seeds]
    plan = plans.default_plan()
    plan["planned_seeds"] = list(seeds)
    for key in ("environment", "environment_hash", "protocol_hash", "ordered_test_item_hash"):
        plan[key] = records[0]["manifest"][key]
    plan["shared_stack_hash"] = plans.shared_stack_fingerprint(records[0])
    return records, plan


def test_canonical_fixed_family_missing_arm_and_no_substitution(plans):
    records, plan = synthetic_cohort(plans, arms=["grpo_g8", "grpo_g2", "ppo"])
    result = plans.analyze_runs(records, plan)
    assert list(result["primary_contrasts"]) == ["g8-ppo", "g8-ppo_8x8", "g8-g2"]
    assert list(result["exploratory_contrasts"]) == ["g2-ppo"]
    assert result["primary_contrasts"]["g8-ppo_8x8"]["p_t"] == 1
    assert result["primary_contrasts"]["g8-ppo_8x8"]["holm_tost_p"] == 1
    assert result["completion"] == {"planned": 8, "selected": 6, "missing": 2}
    assert result["status"] == "interim"
    assert result["final_eligible"] is False
    assert len(result["selected_runs"]) == 6
    assert all(
        set(("run_id", "environment", "protocol_hash", "ordered_test_item_hash", "seed", "arm"))
        <= set(run)
        for run in result["selected_runs"]
    )
    json.dumps(result, allow_nan=False)


def test_endpoint_only_complete_after_explicit_declaration(plans):
    records, plan = synthetic_cohort(plans)
    before = plans.analyze_runs(records, plan)
    assert before["status"] == "interim"
    plan["endpoint"]["declared"] = True
    after = plans.analyze_runs(records, plan)
    assert after["status"] == "endpoint_complete"
    assert after["final_eligible"] is False  # A retrospective repair is still not prospective.
    assert not any(c["equivalence_claim_eligible"] for c in after["primary_contrasts"].values())


def test_missing_pair_reports_seed_and_uses_missing_slot(plans):
    records, plan = synthetic_cohort(plans, arms=["grpo_g8", "ppo"])
    records = [r for r in records if not (r["arm"] == "ppo" and r["seed"] == 123)]
    result = plans.analyze_runs(records, plan)
    contrast = result["primary_contrasts"]["g8-ppo"]
    assert contrast["status"] == "unavailable"
    assert contrast["n"] == 1
    assert contrast["missing_seeds"] == [123]
    assert contrast["holm_p_t"] == 1
    assert {"arm": "ppo", "seed": 123} in result["missing_planned_runs"]


def test_duplicate_run_id_and_identity_rejected(plans):
    records, plan = synthetic_cohort(plans)
    with pytest.raises(ValueError, match="duplicate run_id"):
        plans.analyze_runs(records + [deepcopy(records[0])], plan)
    duplicate = deepcopy(records[0])
    duplicate["manifest"]["run_id"] = "other-run"
    with pytest.raises(ValueError, match="duplicate run identity"):
        plans.analyze_runs(records + [duplicate], plan)


@pytest.mark.parametrize(
    ("field", "value"),
    [("environment", "modal"), ("protocol_hash", "0" * 64), ("ordered_test_item_hash", "1" * 64)],
)
def test_incompatible_cohort_never_paired(plans, field, value):
    records, plan = synthetic_cohort(plans)
    records[0]["manifest"][field] = value
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 7
    assert result["primary_contrasts"]["g8-ppo"]["n"] == 1
    assert any(field in reason for reason in result["excluded_runs"][0]["reasons"])


def test_legacy_metadata_unknown_never_fabricated(plans):
    records, plan = synthetic_cohort(plans)
    del records[0]["manifest"]
    result = plans.analyze_runs(records, plan)
    assert result["excluded_runs"][0]["run_id"] is None
    assert result["completion"]["missing"] == 1
    assert result["status"] == "interim"


def test_wrong_plan_family_or_prospective_relabel_rejected(plans):
    plan = plans.default_plan()
    plan["primary_contrast_ids"][-1] = "g2-ppo"
    with pytest.raises(ValueError, match="frozen three"):
        plans.analyze_runs([], plan)
    plan = plans.default_plan()
    plan["prespecification_status"] = "prespecified"
    with pytest.raises(ValueError, match="retrospective"):
        plans.analyze_runs([], plan)


def test_plan_id_mismatch_excluded(plans):
    records, plan = synthetic_cohort(plans)
    plan["plan_id"] = "different-method"
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 0
    assert all(
        any("analysis_plan_id" in reason for reason in run["reasons"])
        for run in result["excluded_runs"]
    )


def test_bad_binary_and_length_rejected(plans):
    records, plan = synthetic_cohort(plans)
    records[0]["post_correct"].pop()
    with pytest.raises(ValueError, match="length"):
        plans.analyze_runs(records, plan)
    records, plan = synthetic_cohort(plans)
    records[0]["post_correct"][0] = math.nan
    with pytest.raises(ValueError, match="binary"):
        plans.analyze_runs(records, plan)


def test_full_audit_zero_discordance_is_strict_json(plans, audit, monkeypatch):
    from samestack_fixtures import synthetic_run

    records, plan = synthetic_cohort(plans)
    texts = [synthetic_run(arm=r["arm"], seed=r["seed"], smoke=False)[1] for r in records]
    result = audit.analyze(
        {
            "records": records,
            "completions": texts,
            "gold": ["1"] * 1319,
            "gold_item_ids": records[0]["manifest"]["test_item_ids"],
        },
        plan,
        bootstrap_samples=10,
    )
    assert result["item_level"]["g8-ppo"]["pooled_mcnemar_naive_p"] == 1
    assert result["item_level"]["g8-ppo"]["per_seed_mcnemar_p"] == [1, 1]
    assert result["actual_design_power"]["status"] == "unmet"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":1e999}', '{"x":Infinity}'])
def test_strict_json_rejects_duplicate_and_overflow(audit, tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="JSON|Non-finite"):
        audit.load_json(path)


def test_audit_rejects_ordered_gold_and_completion_identity_mismatch(plans, audit):
    from samestack_fixtures import synthetic_run

    record, texts = synthetic_run(smoke=False)
    inputs = {
        "records": [record],
        "completions": [texts],
        "gold": ["1"] * 1319,
        "gold_item_ids": list(reversed(record["manifest"]["test_item_ids"])),
    }
    with pytest.raises(ValueError, match="gold item IDs"):
        audit.validate_inputs(inputs)
    inputs["gold_item_ids"] = list(record["manifest"]["test_item_ids"])
    texts["test_item_ids"] = list(reversed(texts["test_item_ids"]))
    with pytest.raises(ValueError, match="completion item IDs"):
        audit.validate_inputs(inputs)


def test_canonical_holm_reserves_absent_family_members(plans):
    records, plan = synthetic_cohort(plans, arms=["grpo_g8", "ppo"])
    for record in records:
        if record["arm"] == "grpo_g8":
            correct = 10 if record["seed"] == 42 else 15
            record["post_correct"][:correct] = [1] * correct
            record["heldout_post_acc"] = correct / record["n_eval"]
            record["wrong_to_right"] = correct
    result = plans.analyze_runs(records, plan)
    value = result["primary_contrasts"]["g8-ppo"]
    assert value["status"] == "available"
    assert value["holm_p_t"] == pytest.approx(min(1, 3 * value["p_t"]))
    assert value["holm_tost_p"] == pytest.approx(min(1, 3 * value["tost_p"]))


def test_filename_identity_mismatch_excluded(plans):
    records, plan = synthetic_cohort(plans)
    records[0]["source_path"] = "ppo_s42.json"
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 7
    assert any("filename" in reason for reason in result["excluded_runs"][0]["reasons"])


def rehash_synthetic_identity(record):
    """Recompute declared synthetic hashes after a deliberately valid variation."""
    from platform_hybrid.experiments.samestack_contract import canonical_hash

    manifest = record["manifest"]
    manifest["config_hash"] = canonical_hash(manifest["effective_config"])
    manifest["environment_hash"] = canonical_hash(
        {
            "environment": manifest["environment"],
            "dependencies": manifest["dependencies"],
            "hardware": manifest["hardware"],
        }
    )
    manifest["run_id"] = canonical_hash(
        {
            "protocol_hash": manifest["protocol_hash"],
            "config_hash": manifest["config_hash"],
            "environment": manifest["environment"],
            "environment_hash": manifest["environment_hash"],
            "source_commit": manifest["source"]["commit"],
            "source_dirty_hash": manifest["source"]["dirty_hash"],
        }
    )


def test_shared_stack_excludes_intended_arm_seed_differences(plans):
    records, plan = synthetic_cohort(plans)
    assert len({plans.shared_stack_fingerprint(record) for record in records}) == 1
    assert plan["shared_stack_hash"] is not None
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 8
    assert all(
        run["environment_hash"] == plan["environment_hash"]
        and run["shared_stack_hash"] == plan["shared_stack_hash"]
        for run in result["selected_runs"]
    )


@pytest.mark.parametrize("variation", ["gpu", "dependency_build"])
def test_same_platform_label_different_actual_environment_excluded(plans, variation):
    from platform_hybrid.experiments.samestack_contract import validate_run

    records, plan = synthetic_cohort(plans)
    record = records[0]
    if variation == "gpu":
        record["manifest"]["hardware"]["gpu_names"] = ["SYNTHETIC DIFFERENT GPU"]
    else:
        record["manifest"]["dependencies"]["torch"] += "+other-build"
    rehash_synthetic_identity(record)
    assert validate_run(record, require_texts=False)["verified"]
    result = plans.analyze_runs(records, plan)
    assert record["manifest"]["environment"] == plan["environment"]
    assert result["completion"]["selected"] == 7
    assert any(
        "environment_hash does not match" in reason
        for reason in result["excluded_runs"][0]["reasons"]
    )


@pytest.mark.parametrize(
    "variation",
    [
        "source_files",
        "dirty_hash",
        "model_revision",
        "tokenizer_revision",
        "dataset_revision",
        "chat_template",
        "generation",
        "train_order",
    ],
)
def test_valid_same_protocol_different_effective_stack_excluded(plans, variation):
    from platform_hybrid.experiments.samestack_contract import canonical_hash, validate_run

    records, plan = synthetic_cohort(plans)
    record, manifest = records[0], records[0]["manifest"]
    if variation == "source_files":
        manifest["source"]["source_files"]["SYNTHETIC-NOT-REAL.py"] = "1" * 64
    elif variation == "dirty_hash":
        manifest["source"]["dirty_hash"] = "2" * 64
    elif variation.endswith("_revision"):
        component = variation.removesuffix("_revision")
        manifest["revisions"][component]["revision"] = "3" * 40
        manifest["effective_config"][variation] = "3" * 40
        record[variation] = "3" * 40
    elif variation == "chat_template":
        manifest["chat_template_hash"] = canonical_hash("SYNTHETIC changed template")
    elif variation == "generation":
        for mode in ("train", "eval"):
            manifest["generation"][mode]["eos_token_id"] = [3]
    elif variation == "train_order":
        manifest["train_item_ids"] = list(reversed(manifest["train_item_ids"]))
        for item_key, index_key in (
            ("policy_item_ids", "policy"),
            ("critic_warmup_item_ids", "critic_warmup"),
        ):
            manifest["schedules"][item_key] = [
                [manifest["train_item_ids"][i] for i in row]
                for row in manifest["schedules"][index_key]
            ]
    rehash_synthetic_identity(record)
    validation = validate_run(record, require_texts=False)
    assert validation["verified"], validation
    assert manifest["protocol_hash"] == plan["protocol_hash"]
    assert manifest["environment_hash"] == plan["environment_hash"]
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 7
    assert result["primary_contrasts"]["g8-ppo"]["n"] == 1
    assert any(
        "shared_stack_hash does not match" in reason
        for reason in result["excluded_runs"][0]["reasons"]
    )


def test_missing_shared_stack_component_stays_unverified(plans):
    records, plan = synthetic_cohort(plans)
    del records[0]["manifest"]["chat_template_hash"]
    assert plans.shared_stack_fingerprint(records[0]) is None
    result = plans.analyze_runs(records, plan)
    assert result["excluded_runs"][0]["shared_stack_hash"] is None
    assert result["completion"]["selected"] == 7
    assert any(
        "unverified shared_stack_hash" in reason for reason in result["excluded_runs"][0]["reasons"]
    )


def test_unbound_actual_environment_or_stack_keeps_all_runs_interim(plans):
    records, plan = synthetic_cohort(plans)
    plan["environment_hash"] = plan["shared_stack_hash"] = None
    result = plans.analyze_runs(records, plan)
    assert result["completion"]["selected"] == 0
    assert result["status"] == "interim"
    assert any("unbound provenance" in reason for reason in result["interim_reasons"])


def test_pure_api_distinguishes_caller_validated_bundles(plans):
    records, plan = synthetic_cohort(plans)
    checked = [records[0]["manifest"]["run_id"]]
    result = plans.analyze_runs(records, plan, completion_validated_run_ids=checked)
    assert sum(row["completion_bundle_validated"] for row in result["selected_runs"]) == 1
    assert result["artifact_validation"]["completion_validated_selected_run_ids"] == checked
    assert len(result["artifact_validation"]["unchecked_selected_run_ids"]) == 7
    exclusion = {
        "run_id": checked[0],
        "arm": records[0]["arm"],
        "seed": records[0]["seed"],
        "stage": "artifact_completion",
        "reasons": ["Missing completion marker"],
    }
    with pytest.raises(ValueError, match="selected and excluded"):
        plans.analyze_runs(records, plan, input_artifact_exclusions=[exclusion])
    result = plans.analyze_runs(records[1:], plan, input_artifact_exclusions=[exclusion])
    assert result["excluded_runs"][-1] == exclusion
    assert result["artifact_validation"]["input_artifact_exclusion_count"] == 1
    assert result["completion"]["missing"] == 1


def test_file_loader_rejects_uncompleted_manifested_run(audit, tmp_path):
    from samestack_fixtures import synthetic_run

    record, _ = synthetic_run(smoke=False)
    path = tmp_path / "grpo_g8_s42.json"
    path.write_text(json.dumps(record, allow_nan=False))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"test": [{"question": "Synthetic?", "answer": "#### 1"}]}))
    with pytest.raises(ValueError, match="no valid completion bundle"):
        audit.load_inputs(tmp_path, gold)


def test_file_loader_records_validated_bundle_and_rejects_corruption(audit, tmp_path):
    import gzip
    from samestack_fixtures import synthetic_run
    from platform_hybrid.experiments.samestack_contract import write_completion

    record, texts = synthetic_run(smoke=False)
    path = tmp_path / "grpo_g8_s42.json"
    path.write_text(json.dumps(record, allow_nan=False))
    text_path = tmp_path / "grpo_g8_s42_texts.json.gz"
    with gzip.open(text_path, "wt") as stream:
        json.dump(texts, stream, allow_nan=False)
    adapter = tmp_path / "adapters/grpo_g8_s42"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"SYNTHETIC-NOT-A-MODEL")
    for name in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / name).write_text('{"synthetic":true}')
    for mode in ("train", "eval"):
        (adapter / f"{mode}_generation_config.json").write_text(
            json.dumps(record["manifest"]["generation"][mode])
        )
    write_completion(path)
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"test": [{"question": "Synthetic?", "answer": "#### 1"}]}))
    inputs = audit.load_inputs(tmp_path, gold)
    assert inputs["completion_validated_run_ids"] == [record["manifest"]["run_id"]]
    assert len(inputs["records"]) == 1  # Completion marker is not a second result.
    texts["post"][0] = "CHANGED SYNTHETIC TEXT"
    with gzip.open(text_path, "wt") as stream:
        json.dump(texts, stream, allow_nan=False)
    with pytest.raises(ValueError, match="no valid completion bundle"):
        audit.load_inputs(tmp_path, gold)


def test_loader_discovers_nested_promoted_runs_and_ignores_receipts(audit, tmp_path):
    import gzip

    leaf = tmp_path / "grpo_g8_s42"
    leaf.mkdir()
    (leaf / "grpo_g8_s42.json").write_text(json.dumps({"arm": "grpo_g8", "seed": 42}))
    (leaf / "grpo_g8_s42.complete.json").write_text("{}")
    with gzip.open(leaf / "grpo_g8_s42_texts.json.gz", "wt") as stream:
        json.dump({"pre": ["1"], "post": ["1"]}, stream)
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"test": [{"question": "synthetic", "answer": "#### 1"}]}))
    inputs = audit.load_inputs(tmp_path, gold)
    assert len(inputs["records"]) == 1
    assert inputs["records"][0]["seed"] == 42
    assert inputs["completions"][0]["post"] == ["1"]


def test_loader_rejects_unrecognized_run_filename(audit, tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"test": [{"question": "synthetic", "answer": "#### 1"}]}))
    (tmp_path / "unexpected_s42.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected run filename"):
        audit.load_inputs(tmp_path, gold)

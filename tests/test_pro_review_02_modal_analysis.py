"""Concrete review-02 reproductions. Synthetic/local data only; no training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "platform_modal/scripts"
for folder in (SCRIPTS, SCRIPTS / "p5p8", SCRIPTS / "berkeley"):
    sys.path.insert(0, str(folder))

import _analysis_common as common
import _stats as stats_helpers
from _review02_contracts import (
    retention_noninferiority,
    paired_spearman_ci,
    cell_distance_spearman_ci,
    sampled_pairs,
    finite_json,
    profile_counts,
)
import _harmbench_protocol as hb
import _minreport_contract as mr
import adding_error_bars_to_evals as bars
import group_size_analysis as groups
import p5_iter205_manifest_minreport_subfield_audit as subfields
import p5_minreport_subfield_audit as legacy_subfields
import minreport_auditor as badge
import p5_iter113_minreport_v22_recovery as recovery
import p8_iter196_vstat_loo_ablation as loo
import p8_iter184_vstat_quartile_ablation as quartile
import p8_iter172_vstat_ensemble_precision as precision
import statistical_rigor_report as rigor
import multiact_population_eval as population
import eval_protocol_hardening as hardening
from _plot_provenance import archive_history, metric_spec
import generate_ramesh_status_slides_20260704 as deck


def test_retention_equal_accuracies_pass_noninferiority():
    r = retention_noninferiority([0.8] * 5, [0.8] * 5)
    assert r["noninferior"] is True
    assert r["estimate"] == pytest.approx(0.8 - 0.976 * 0.8)
    assert r["retention_ratio"] == pytest.approx(1)
    assert r["confidence_level"] == pytest.approx(0.9)


def test_retention_twenty_point_loss_is_inferior():
    r = retention_noninferiority([0.6] * 5, [0.8] * 5)
    assert r["noninferior"] is False
    assert r["status"] == "inferior"
    # Existing difference summaries cannot recover scaled paired variance.
    summary = bars.headline_h4_native_wu_paired([dict(diff=0.2, diff_se=0.01, cohens_d=2)])
    assert summary["wu_claim_holds_at_97.6pct"] is not True
    assert summary["result"]["status"] == "insufficient_data"


def test_relative_margin_depends_on_reference_accuracy():
    assert retention_noninferiority([0.19] * 5, [0.20] * 5)["noninferior"] is False
    assert retention_noninferiority([0.79] * 5, [0.80] * 5)["noninferior"] is True


def test_group_summary_intervals_are_not_synthetic_samples():
    rows = [
        dict(
            budget_tokens=1_000_000,
            G=g,
            heldout_acc_mean=0.8,
            heldout_acc_ci_low=0.79,
            heldout_acc_ci_high=0.81,
        )
        for g in [4, 32]
    ]
    result = groups.broader_g4_vs_g32(rows)["per_T"][0]
    assert result["retention"] == 1
    assert result["status"] == "insufficient_data"
    assert result["wu_claim_holds"] is None
    assert math.isnan(result["retention_ci_low"])


def test_group_real_seed_pairing_and_duplicate_validation():
    rows = [
        dict(
            budget_tokens=1_000_000,
            G=g,
            seed=s,
            heldout_acc=0.8,
            model_id="model",
            task_id="task",
            evaluation_id="eval",
            comparison_config_id="fixed-config",
        )
        for g in [4, 32]
        for s in [3, 1, 2]
    ]
    assert groups.broader_g4_vs_g32(rows)["per_T"][0]["wu_claim_holds"] is True
    with pytest.raises(ValueError, match="duplicate seed"):
        groups.broader_g4_vs_g32(rows + [rows[0]])


@pytest.mark.parametrize(
    ("lo", "hi", "radius", "expected"),
    [
        (-0.01, 0.01, 0.02, "EQUIVALENT"),
        (0.03, 0.1, 0.02, "SUPERIOR"),
        (-0.1, -0.03, 0.02, "INFERIOR"),
        (-0.1, 0.1, 0.02, "INCONCLUSIVE"),
        (float("nan"), 0.1, None, "INSUFFICIENT_DATA"),
    ],
)
def test_distinct_interval_verdicts(lo, hi, radius, expected):
    assert bars.ci_verdict(lo, hi, equiv_radius=radius) == expected


def test_harmbench_outage_aborts_without_fabricated_data():
    def unavailable(*args, **kwargs):
        raise OSError("dataset unavailable")

    with pytest.raises(RuntimeError, match="evaluation aborted"):
        hb.load_behaviors(unavailable, "a" * 40)


def test_harmbench_pinning_ids_context_subset_provenance():
    rows = [
        dict(BehaviorID=str(i), Behavior="test", FunctionalCategory="standard") for i in range(3)
    ]
    calls = []

    def loader(*args, **kwargs):
        calls.append(kwargs)
        return rows

    result, meta = hb.load_behaviors(loader, "a" * 40, limit=2)
    assert calls[0]["revision"] == "a" * 40
    assert meta["subset_evaluation"]
    assert meta["selected_example_ids"] == ["0", "1"]
    assert meta["category_composition"] == {"standard": 2}
    with pytest.raises(ValueError, match="revision"):
        hb.load_behaviors(loader, "main")
    with pytest.raises(ValueError, match="context"):
        hb.load_behaviors(
            lambda *a, **k: [
                dict(BehaviorID="x", Behavior="test", FunctionalCategory="contextual")
            ],
            "a" * 40,
        )


def test_keyword_in_compliant_completion_never_becomes_safety_score():
    summary, details = hb.keyword_diagnostic(
        [dict(BehaviorID="x", Behavior="synthetic", FunctionalCategory="standard")],
        ["I'm sorry, here are the requested instructions"],
    )
    assert summary["refusal_keyword_rate"] == 1
    assert summary["safety_claim_supported"] is False
    assert summary["benchmark_score_available"] is False
    assert "is_safe" not in details[0]
    assert "refusal_rate" not in summary


def test_shared_index_spearman_all_2000_pairs():
    x = np.arange(2000)
    lo, hi = paired_spearman_ci(x, x, 100, 42)
    assert lo == pytest.approx(1)
    assert hi == pytest.approx(1)


def test_cell_bootstrap_recomputes_both_distances():
    # Hamming between cumulative binary fingerprints equals outcome distance.
    f = [[int(j < i) for j in range(12)] for i in range(12)]
    lo, hi = cell_distance_spearman_ci(f, np.arange(12), 100, 42)
    assert lo == pytest.approx(1)
    assert hi == pytest.approx(1)


@pytest.mark.parametrize("n", [0, 1, 2, 10, 63, 98])
def test_pair_sampler_terminates_and_caps(n):
    pairs = sampled_pairs(n)
    assert len(pairs) == min(2000, n * (n - 1) // 2)
    assert len(set(pairs)) == len(pairs)
    assert all(0 <= i < j < n for i, j in pairs)


def test_legacy_text_never_proves_substantive_facts(tmp_path, monkeypatch):
    assert "disjoint_from_reward_env" not in subfields.parse_heldout_split(
        "same train ids as reward environment"
    )
    assert subfields.parse_decontamination_notes("not performed")["performed"] is False
    assert "per_step_zvf" not in subfields.parse_per_step_zvf_path(str(tmp_path / "missing.json"))
    assert "precision" not in subfields.parse_sampler_backend_precision("tinker-closed")
    assert legacy_subfields._covered_subfields("loss_form", "grpo") == []


def fully_specified_manifest():
    manifest = {"cell_id": "synthetic", "min_report": {}}
    for item, field, _ in subfields.MIN_REPORT_FIELDS:
        if field in mr.BOOL_FIELDS:
            value = False
        elif field in mr.NUMBER_FIELDS:
            value = 0.0
        elif field in mr.INTEGER_FIELDS:
            value = 2
        else:
            value = "bf16" if field == "precision" else "explicit_value"
        manifest["min_report"].setdefault(item, {})[field] = value
    manifest["min_report"]["group_size_schedule"]["adaptation_rule"] = {
        "applicable": False,
        "reason": "fixed group size",
    }
    return manifest


def test_complete_incomplete_and_false_zero_na_manifest_fixtures():
    full = fully_specified_manifest()
    assert all(
        subfields.classify_subfield(full, it, s, pk, {}) == "EXTRACTABLE_LITERAL"
        for it, s, pk in subfields.MIN_REPORT_FIELDS
    )
    assert badge.score_manifest(full)["badge"] == 100
    assert badge.score_manifest({"cell_id": "empty"})["badge"] == 0
    assert (
        subfields.classify_subfield({"performed": False}, "decontamination", "performed", None, {})
        == "EXTRACTABLE_LITERAL"
    )
    assert (
        subfields.classify_subfield({"kl_beta": 0.0}, "reference_kl", "kl_beta", None, {})
        == "EXTRACTABLE_LITERAL"
    )
    assert mr.field_report(False) == dict(
        reported=True, applicable=True, verified=False, value=False, status="reported"
    )
    assert mr.valid_literal("precision", "closed") is False


def test_split_verification_uses_ids():
    assert mr.verify_split(["a"], ["a"])["disjoint"] is False
    assert mr.verify_split(["a"], ["b"])["disjoint"] is True
    assert mr.verify_split([], [])["verified"] is False


@pytest.mark.parametrize("path_kind", ["empty", "directory", "missing", "invalid", "valid"])
def test_trajectory_requires_file_and_schema(tmp_path, path_kind):
    path = tmp_path / "trace.json"
    if path_kind == "invalid":
        path.write_text('{"hello":"world"}')
    if path_kind == "valid":
        path.write_text('[{"step":1,"zvf":0.2,"gu":0.8}]')
    value = "" if path_kind == "empty" else str(tmp_path) if path_kind == "directory" else str(path)
    r = mr.verify_trajectory(value, tmp_path)
    assert r["verified"] is (path_kind == "valid")


def test_manifest_denominator_comes_from_inventory(tmp_path):
    inv = tmp_path / "cells.tsv"
    inv.write_text("cell_id\na\nb\nc\n")
    folder = tmp_path / "manifests"
    folder.mkdir()
    (folder / "a.json").write_text("{}")
    (folder / "b.json").write_text("not json")
    records = mr.load_inventory_manifests(folder, inv)
    assert len(records) == 3
    counts = mr.inventory_counts(records)
    assert counts["observed"] == 1
    assert counts["invalid"] == 1
    assert counts["missing"] == 1
    # Deleting all report fields does not change eligibility.
    (folder / "a.json").write_text('{"loss_form":"grpo"}')
    assert len(mr.load_inventory_manifests(folder, inv)) == 3


def test_registry_missing_report_still_eligible():
    from p6_registry_minreport_audit import eligible_stack_entries

    records = [
        {"record_type": "stack", "_path": "stack.json"},
        {"record_type": "delta", "_path": "delta_x.json"},
    ]
    assert eligible_stack_entries(records) == [records[0]]


def test_tied_auc_and_spearman_reproductions():
    assert loo.auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert math.isnan(loo.auc(np.array([0.5, 0.5]), np.array([1, 1])))
    assert recovery.spearman(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(0)
    assert math.isnan(recovery.spearman(np.ones(4), np.arange(4)))


def test_h1_budget_selected_by_value_and_no_cross_budget_interval():
    rows = [
        dict(T=t, GU_ratio=r)
        for t, r in [(64_000_000, 1), (4_000_000, 1), (1_000_000, 5), (16_000_000, 1)]
    ]
    h = bars.headline_h1_gu_ratio({}, rows)
    assert h["point_GU_ratio"] == 5
    assert math.isnan(h["CI95_lo"])
    assert h["result"]["ci_low"] is None


def test_singular_ols_draws_use_retained_quantiles():
    diagnostics = {}
    b, lo, hi, n = bars.ols_slope_with_ci(
        [0, 1, 2], [1, 2, 3], B=1000, seed=0, diagnostics=diagnostics
    )
    assert (b, lo, hi) == pytest.approx((1, 1, 1))
    assert 0 < diagnostics["valid_resamples"] < diagnostics["requested_resamples"] == 1000


def test_exact_sign_test_not_bootstrap_tail():
    r = common.paired_bootstrap_delta(np.zeros(5), np.arange(1, 6), 1000, 0)
    assert r["p"] == pytest.approx(0.0625)
    assert r["bootstrap_tail_fraction_two_sided"] == 0


def test_undefined_data_remain_undefined():
    r = stats_helpers.bootstrap_ci_rng([0.8], 100, 0.05, random.Random(1))
    assert r.estimate == 0.8
    assert r.status == "insufficient_data"
    assert stats_helpers.wilson_ci(0, 0).status == "insufficient_data"
    assert quartile.auc_gap_verdict(float("nan")) == "INSUFFICIENT_DATA"
    assert population.hypothesis_h4({}) == ([], 0, 0)
    assert json.dumps(finite_json({"x": float("nan")}), allow_nan=False) == '{"x": null}'


def test_numeric_last_endpoint_and_explicit_pair_keys():
    rows = [
        dict(method=arm, seed=seed, step=step, heldout_acc=value, initialization_id="same")
        for arm in ["a", "b"]
        for seed in [1, 2]
        for step, value in [(100, 0.8), (2, 0.2), (10, 0.5)]
    ]
    records, excluded = rigor.endpoint_records(rows, "method", ["heldout_acc"], "demo")
    assert not excluded
    assert len(records) == 4
    assert all(r["step"] == 100 and r["accuracy"] == 0.8 for r in records)
    report = rigor.paired_comparison(records, "a", "b", "test")
    assert report["n_pairs"] == 2
    assert report["estimate"] == 0
    assert report["matched_init"] == "yes"
    assert report["meets_step_minimum"]
    assert report["power_status"].startswith("not_assessed")
    with pytest.raises(ValueError, match="duplicate"):
        rigor.endpoint_records(rows + [rows[0]], "method", ["heldout_acc"], "demo")


def test_group_budget_summary_not_treated_as_seeds():
    rows = [dict(G=4, heldout_acc_mean=0.8, budget_tokens=t, n_seeds=20) for t in [1, 2, 3, 4]]
    records, excluded = rigor.endpoint_records(rows, "G", ["heldout_acc_mean"], "group")
    assert records == []
    assert len(excluded) == 4


def test_loo_direction_ties_and_operating_denominators():
    assert loo.most_damaging_features(0.9, {"mean": 0.89, "std": 0.6, "max": 0.8}) == ["std"]
    assert loo.most_damaging_features(0.9, {"mean": 0.6, "std": 0.6}) == ["mean", "std"]
    s = np.array([0.9, 0.8, 0.1, 0.05])
    y = np.array([1, 0, 0, 0])
    rates = loo.operating_rates(s, y, 0.5)
    assert rates == pytest.approx(
        dict(flag_rate=0.5, fp_per_transaction=0.25, false_positive_rate=1 / 3)
    )
    assert loo.recall_at_flag_budget(s, y, 0.25) == 1
    th, _, flag = loo.cost_curve(s, y, 100)
    idx = np.where(np.isclose(th, 0.5))[0][0]
    assert flag[idx] == 0.5


def test_precision_denominator_includes_cells_that_never_reach_target():
    rows = [
        dict(
            seed=i, rate_pct=1, fset="full", tau=0.5, esc_prec=0.1 if i == 0 else 0, value_rate=0.9
        )
        for i in range(100)
    ]
    r = precision.precision_target_rate(rows)
    assert r["rate"] == 0.01
    assert r["conditional_rate"] == 1
    assert r["eligible"] == 100


def test_population_sd_sem_identity_not_success():
    rows, decisive, n = population.hypothesis_h3(
        {"scaling_law_three_phase": [{"peak": v} for v in [0, 1, 2, 3]]}, {}
    )
    assert decisive == 0
    assert n == 1
    assert rows[0]["verdict"] == "DESCRIPTIVE_SD_VS_SEM"
    assert "sd_across_heterogeneous_values" in rows[0]
    rows, decisive, n = population.hypothesis_h3({"scaling_law_three_phase": [{"peak": 1}] * 4}, {})
    assert decisive == 0
    assert rows[0]["verdict"] == "INSUFFICIENT_DATA"


def test_unequal_axis_contrasts_survive_json():
    from minreport_stratified_audit import contrast_json

    result = contrast_json({(1, "G"): 1.25, (1, "zvf"): 8.5}, [1], ["G", "zvf"])
    assert result == {"item1/G": 1.25, "item1/zvf": 8.5}


def test_emission_counts_actual_fields_and_partition():
    items = [("a", "loss_form", "live", "manifest"), ("b", "new", "schema_only", "schema")]
    r = recovery.item_emission_counts(
        items, {"x": {}, "y": {"loss_form": "grpo"}}, {"x": {}, "y": {}}, set()
    )
    assert r[0]["n_cells_emitted"] == 1
    assert r[1]["n_cells_emitted"] == 0


def test_stable_seed_ignores_python_hash_seed():
    code = "import sys;sys.path.insert(0,sys.argv[1]);from _review02_contracts import stable_seed;print(stable_seed(42,'contrast',3))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code, str(SCRIPTS)],
            env=dict(os.environ, PYTHONHASHSEED=seed),
            text=True,
        )
        for seed in ["1", "99"]
    ]
    assert outputs[0] == outputs[1]


def test_singletons_are_not_distinct_profiles_divided_by_cells():
    r = profile_counts([i for i in range(49) for _ in range(2)])
    assert r["n_distinct_profiles"] / r["n_cells"] == 0.5
    assert r["singleton_fraction"] == 0


def test_sample_pooled_d_distinct_from_explicit_legacy():
    a, b = [1, 2, 3], [0, 1, 2]
    assert stats_helpers.cohens_d_pstdev_pooled(a, b) / stats_helpers.cohens_d_sample_pooled(
        a, b
    ) == pytest.approx(math.sqrt(3 / 2))


def test_result_literal_cannot_execute(tmp_path):
    marker = tmp_path / "executed"
    with pytest.raises((ValueError, SyntaxError)):
        population.parse_group_table(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()
    assert population.parse_group_table('[{"G":2,"last10_reward_mean":0.8}]')[0]["G"] == 2


def test_empirical_oracle_searches_below_fixed_grid_and_endpoints():
    cost, threshold = loo.min_cost(np.array([0.009, 0.001]), np.array([1, 0]), 100)
    assert cost == 0
    assert 0.001 < threshold <= 0.009
    assert loo.min_cost(np.array([0.5, 0.5]), np.array([0, 0]), 100)[0] == 0


def test_reward_tensor_units_and_exact_redundancy():
    rv = np.array([[0, 0], [0, 1], [1, 1]])
    r = recovery.compute_items_from_tensor(rv)
    assert r["K_variance_residual"] == pytest.approx(
        r["G"] ** 2 * r["prompt_p_hat_var"] - r["G"] * r["p_hat"] * (1 - r["p_hat"])
    )
    for bad in [np.array([[0.5, 1]]), np.array([[float("nan"), 1]]), np.empty((0, 2))]:
        with pytest.raises(ValueError, match="binary reward"):
            recovery.compute_items_from_tensor(bad)


def test_fixed_campaign_entropy_not_bootstrapped_identifier_bias():
    import p5_placebo_replacement_analysis as diversity

    manifests = [
        (
            str(i),
            {k: str(i) if k == "per_step_zvf_path" else "fixed" for k in diversity.CURRENT_ITEMS},
        )
        for i in range(98)
    ]
    point, lo, hi = diversity.boot_total_h(manifests, n_boot=20)
    assert point == pytest.approx(math.log2(98))
    assert math.isnan(lo)
    assert math.isnan(hi)
    _, _, scientific = diversity.compute_v1_info_budget(manifests)
    assert scientific == 0


def test_within_pool_ties_are_sets_not_arbitrary_winners():
    values = {"a": [0.1, 0.1], "b": [0.1, 0.1], "c": [0.3, 0.3]}
    assert hardening.tied_top_set(values) == {"a", "b"}
    by_m = {
        m: [{"seed": i, "zvf_risk_max": v} for i, v in enumerate(vals)]
        for m, vals in values.items()
    }
    result = hardening.h1_mvsp(by_m, n_boot=20)
    assert result["tied_full_winners"] == ["a", "b"]
    assert result["rows"][-1]["self_comparison"] is True
    assert "within-pool" in result["interpretation"]
    assert "not clustering" in hardening.h4_clusters(by_m)["interpretation"]


def test_plot_exports_exact_unsampled_rows_and_units(tmp_path):
    class Run:
        id = "synthetic"
        name = "test"
        path = ["entity", "project", "synthetic"]

        def history(self, *args, **kwargs):
            raise AssertionError("sampled history must not be used")

        def scan_history(self):
            return iter(
                [
                    {"_step": 2, "train/accuracy": 0.8},
                    {"_step": 1, "train/accuracy": 0.4},
                    {"_step": 3, "other": 5},
                ]
            )

    rows, meta = archive_history(Run(), metric_spec("train/accuracy"), tmp_path / "plot.png", 2)
    assert rows == [
        {"step": 1, "value": 0.4, "plotted_value": 0.4},
        {"step": 2, "value": 0.8, "plotted_value": pytest.approx(0.6)},
    ]
    assert meta["omitted_missing_metric_or_step"] == 1
    assert meta["metric_spec"]["label"] == "Training accuracy"
    assert meta["input_sha256"]
    assert json.loads((tmp_path / "plot.png.synthetic.rows.json").read_text())[1]["value"] == 0.8


def test_zvf_out_of_range_rejected_instead_of_clipped(tmp_path):
    class Run:
        id = "bad"

        def scan_history(self):
            return iter([{"_step": 0, "zvf": 80}])

    with pytest.raises(ValueError, match="outside"):
        archive_history(Run(), metric_spec("zvf"), tmp_path / "plot.png")


def test_missing_log_is_unknown_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(deck, "ROOT", tmp_path)
    assert deck.n2_status().startswith("unknown:")
    assert deck.n10_status().startswith("unknown:")
    assert deck.mega_status().startswith("unknown:")


def test_four_perfect_ranks_exact_probability_is_two_over_24():
    from _review02_contracts import exact_small_spearman

    rho, p = exact_small_spearman([0, 1, 2, 3], [0, 1, 2, 3])
    assert rho == 1
    assert p == pytest.approx(2 / 24)


def test_delta_telemetry_requires_valid_observed_reward_units():
    from p5_delta_div_minreport import delta_div

    assert delta_div({"G": 2, "mean_reward": None, "zvf": 0.5}) is None
    assert delta_div({"G": 2, "mean_reward": 50, "zvf": 0.5}) == 0
    assert delta_div({"G": 2, "mean_reward": 0.5, "zvf": 0.5}, reward_unit="fraction") == 0
    with pytest.raises(ValueError, match="units"):
        delta_div({"G": 2, "mean_reward": 101, "zvf": 0.5})


def test_singular_design_records_insufficient_status_and_counts():
    result = bars.ols_slope_with_ci([1, 1, 1], [1, 2, 3], B=100)
    assert result.status == "insufficient_data"
    assert result.valid_resamples == 0
    assert result.requested_resamples == 100


def test_step_bootstrap_can_preserve_prespecified_time_blocks():
    rows = [{"step": i, "value": i} for i in range(4)]
    draws = stats_helpers.paired_step_bootstrap(
        rows, lambda sample: sum(r["value"] for r in sample), 10, 0, block_length=4
    )
    assert draws == [6] * 10


def test_structured_error_bar_json_tsv_and_narrative_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(bars, "OUT", tmp_path)
    monkeypatch.setattr(
        bars,
        "load_zvf_link",
        lambda: [
            dict(T=1_000_000, GU_ratio=5, retention=0.8),
            dict(T=4_000_000, GU_ratio=1, retention=0.7),
        ],
    )
    monkeypatch.setattr(bars, "load_iter123_snr", lambda: dict(slope=0.4, ci_lo=0.2, ci_hi=0.6))
    monkeypatch.setattr(
        bars, "load_iter135_nativewu", lambda: [dict(diff=0.2, diff_se=0.01, cohens_d=2)]
    )
    monkeypatch.setattr(bars, "load_iter137_offset_t80", list)
    monkeypatch.setattr(bars, "load_iter130_auroc", dict)
    monkeypatch.setattr(bars, "load_iter136_paired", list)
    bars.main()
    import csv

    data = json.loads((tmp_path / "adding_error_bars_summary.json").read_text())
    with (tmp_path / "adding_error_bars_audit.tsv").open() as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    assert len(table) == len(data["results"]) == 7
    for row, record, prose in zip(table, data["results"], data["key_findings"], strict=True):
        assert row["headline"] == record["headline"]
        assert row["estimand"] == record["estimand"]
        assert record["status"] in prose
    h1 = data["results"][0]
    assert h1["estimate"] == 5
    assert h1["ci_low"] is None
    assert data["results"][-1]["ci_low"] is None


def test_group_semilog_units_and_point_only_figure(tmp_path, monkeypatch):
    runs = [
        dict(
            group_size=g,
            seed=seed,
            heldout_acc=0.8,
            mean_zvf=z,
            step_log=[dict(mean_reward=0.8, zvf=z, entropy=1)] * 3,
        )
        for g, z in [(2, 0.838), (4, 0.76), (8, 0.70), (16, 0.6312)]
        for seed in range(2)
    ]
    a = groups.fit_reward_vs_G(runs)
    assert "semilog_zvf_slope_per_decade" in a
    assert "loglog_zvf_slope_per_decade" not in a
    b = groups.broader_g4_vs_g32(
        [
            dict(
                budget_tokens=t,
                G=g,
                heldout_acc_mean=0.8,
                heldout_acc_ci_low=0.7,
                heldout_acc_ci_high=0.9,
            )
            for t in [1_000_000, 4_000_000]
            for g in [4, 32]
        ]
    )
    monkeypatch.setattr(groups, "RES", tmp_path)
    output = groups.write_outputs(a, b).read_text()
    assert "20.68 percentage points" in output
    assert "GU=1-ZVF increases" in output
    assert "mean_zvf ~" in output
    assert "log10(mean_zvf)" not in output
    groups.write_figure(a, b, tmp_path / "synthetic.png")
    assert (tmp_path / "synthetic.png").is_file()


def test_stack_main_small_corpus_completes_and_records_cell_uncertainty(tmp_path, monkeypatch):
    import p5_stack_axis_minreport as stack

    cells = [
        dict(
            cell_id=str(i),
            G=2 + i,
            temperature=0.5 + i / 10,
            seed=i,
            task_slice="a" if i % 2 else "b",
            model_family="one" if i < 2 else "two",
            zvf=0.2 * i,
            mean_reward=0.2 * i,
            pcd=i,
            mean_completion_len=10 + i,
        )
        for i in range(4)
    ]
    manifests = {
        str(i): {key: str(i) if key == "per_step_zvf_path" else "fixed" for key in stack.V1_ITEMS}
        for i in range(4)
    }
    monkeypatch.setattr(stack, "load_cells", lambda: cells)
    monkeypatch.setattr(stack, "load_manifests", lambda: manifests)
    monkeypatch.setattr(stack, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stack, "N_BOOT", 10)
    stack.main()
    result = json.loads((tmp_path / "p5_stack_axis_minreport_summary.json").read_text())
    assert result["n_pairs"] == 6
    assert "cell-bootstrap" in result["correlation_uncertainty"]
    assert "not joint information" in result["entropy_estimand"]
    assert "does NOT" not in result["verdict_H2"]


def test_reported_disjoint_claim_can_be_verified_contradictory():
    manifest = fully_specified_manifest()
    manifest["min_report"]["heldout_split"].update(
        disjoint_from_reward_env=True, heldout_example_ids=["same"], reward_example_ids=["same"]
    )
    result = badge.score_manifest(manifest)
    assert result["badge"] == 100  # completeness is independent of scientific truth
    assert result["evidence"]["split"]["verified"] is True
    assert result["evidence"]["split"]["contradictory"] is True
    assert mr.field_report("unknown")["reported"] is False


def test_figure_threshold_is_increment_over_baseline():
    from _review02_contracts import profile_target_increment

    assert profile_target_increment(98, 15) == 34
    assert profile_target_increment(99, 15) == 35
    assert profile_target_increment(98, 60) == 0


def test_missing_report_is_not_consistency_success(tmp_path, monkeypatch):
    import delta_minreport_consistency as consistency

    registry = tmp_path / "registry"
    registry.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for identity, report in [
        ("missing", {}),
        ("match", {"loss_form": {"length_normalization": False}}),
    ]:
        (registry / f"{identity}.json").write_text(
            json.dumps(
                dict(
                    id=identity,
                    record_type="stack",
                    min_report=report,
                    variant_deltas_applied=[
                        dict(
                            delta_id="delta_drgrpo",
                            component="length_normalization",
                            status="implemented",
                        )
                    ],
                )
            )
        )
    monkeypatch.setattr(consistency, "REG", registry)
    monkeypatch.setattr(consistency, "OUT", out)
    assert consistency.main() == 0
    honesty = json.loads((out / "delta_minreport_consistency.json").read_text())[
        "implementation_honesty"
    ]
    assert honesty["n_missing_report"] == 1
    assert honesty["checked_count"] == 1
    assert honesty["match_rate"] == 0.5
    assert honesty["consistent_rate"] == 1


def test_registry_missing_and_malformed_reports_stay_in_denominator(tmp_path, monkeypatch):
    import p6_registry_minreport_audit as registry

    entries = [
        dict(_path="stack_a.json", record_type="stack"),
        dict(_path="stack_b.json", record_type="stack", min_report={"loss_form": "unknown"}),
    ]
    monkeypatch.setattr(registry, "collect_registry_entries", lambda: entries)
    monkeypatch.setattr(registry, "OUT", tmp_path)
    registry.main()
    result = json.loads((tmp_path / "p6_registry_minreport_summary.json").read_text())
    assert result["n_stack_entries"] == 2
    assert result["missing_min_report"] == 1
    assert all(not value for value in registry.per_entry_subfield_audit(entries[1]).values())


def test_parsed_fields_are_not_automatically_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setattr(badge, "OUT_DIR", tmp_path)
    monkeypatch.setattr(badge, "FIG_DIR", tmp_path / "fig")
    result = badge.write_outputs(
        [badge.score_manifest({"cell_id": "legacy", "decontamination_notes": "not performed"})]
    )
    assert result["inventory_counts"]["not_applicable"] == 0


@pytest.mark.parametrize(
    ("manifest", "item", "field", "parser", "cells"),
    [
        (
            {"group_size_schedule": "fixed-G=-1"},
            "group_size_schedule",
            "initial_g",
            "group_size_schedule",
            {},
        ),
        (
            {"sampler_backend_precision": "unknown"},
            "sampler_backend",
            "backend",
            "sampler_backend_precision",
            {},
        ),
        (
            {"cell_id": "x"},
            "sampler_backend",
            "temperature",
            "cells_tsv",
            {"x": {"temperature": float("nan")}},
        ),
        (
            {"cell_id": "x"},
            "sampler_backend",
            "temperature",
            "cells_tsv",
            {"x": {"temperature": -1}},
        ),
    ],
)
def test_parsed_and_inventory_values_obey_same_typed_validation(
    manifest, item, field, parser, cells
):
    assert subfields.classify_subfield(manifest, item, field, parser, cells) == "AMBIGUOUS"
    normalized = subfields.normalized_min_report(manifest, cells)
    assert normalized[item][field] is None


def test_valid_parsed_inventory_zero_and_literal_values_agree():
    parsed = subfields.normalized_min_report(
        {"cell_id": "x", "group_size_schedule": "fixed-G=2"}, {"x": {"temperature": 0.0}}
    )
    literal = subfields.normalized_min_report(
        {
            "min_report": {
                "group_size_schedule": {"initial_g": 2, "schedule": "fixed"},
                "sampler_backend": {"temperature": 0.0},
            }
        }
    )
    assert parsed == literal


def test_retention_rejects_seed_collision_across_model_task_eval():
    rows = [
        dict(
            budget_tokens=1_000_000,
            G=g,
            seed=seed,
            heldout_acc=0.8,
            model_id=f"model-{g}",
            task_id=f"task-{g}",
            evaluation_id=f"eval-{g}",
            comparison_config_id="configuration",
        )
        for g in (4, 32)
        for seed in (1, 2, 3)
    ]
    result = groups.broader_g4_vs_g32(rows)["per_T"][0]
    assert result["matching_status"] == "mismatched"
    assert result["wu_claim_holds"] is None
    assert result["n_pairs"] == 0
    assert result["n_candidate_seed_pairs"] == 3
    assert math.isnan(result["retention"])
    assert set(result["noninferiority"]["matching"]["mismatched_fields"]) == {
        "model",
        "task",
        "evaluation",
    }


@pytest.mark.parametrize("absent", ["all", "comparison_config_id", "evaluation_id"])
def test_retention_missing_identity_is_unknown_not_matched(absent):
    identities = dict(
        model_id="model", task_id="task", evaluation_id="eval", comparison_config_id="config"
    )
    if absent == "all":
        identities = {}
    else:
        identities.pop(absent)
    rows = [
        dict(budget_tokens=1_000_000, G=g, seed=seed, heldout_acc=0.8, **identities)
        for g in (4, 32)
        for seed in (1, 2, 3)
    ]
    result = groups.broader_g4_vs_g32(rows)["per_T"][0]
    assert result["matching_status"] == "unknown"
    assert result["wu_claim_holds"] is None
    assert result["n_pairs"] == 0
    assert math.isnan(result["retention_ci_low"])


def test_retention_rejects_mismatched_training_configuration():
    records = [
        dict(model_id="model", task_id="task", evaluation_id="eval", comparison_config_id=c)
        for c in ["one", "two"]
    ]
    assert groups.pairing_identity_status(records)["status"] == "mismatched"
    records = [
        dict(
            model_id="model",
            task_id="task",
            evaluation_id="eval",
            comparison_config_id="same",
            heldout_ids_sha256=checksum,
        )
        for checksum in ["a" * 64, "b" * 64]
    ]
    assert groups.pairing_identity_status(records)["mismatched_fields"] == ["heldout_ids_sha256"]


def test_constant_stack_correlation_and_interval_are_undefined(tmp_path, monkeypatch):
    import p5_stack_axis_minreport as stack

    assert math.isnan(stack.spearman([0, 0, 0], [1, 2, 3]))
    assert math.isnan(stack.spearman([1, 2, 3], [0, 0, 0]))
    # Even distinct one-hot cells have a constant observed Hamming vector.
    assert all(math.isnan(x) for x in cell_distance_spearman_ci(np.eye(4), [0, 1, 2, 3], 10))
    cells = [
        dict(
            cell_id=str(i),
            G=2,
            temperature=0.5,
            seed=0,
            task_slice="task",
            model_family="model",
            zvf=0.2 * i,
            mean_reward=0.2 * i,
            pcd=i,
            mean_completion_len=10 + i,
        )
        for i in range(4)
    ]
    monkeypatch.setattr(stack, "load_cells", lambda: cells)
    monkeypatch.setattr(
        stack, "load_manifests", lambda: {str(i): fully_specified_manifest() for i in range(4)}
    )
    monkeypatch.setattr(stack, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stack, "N_BOOT", 10)
    stack.main()
    result = json.loads((tmp_path / "p5_stack_axis_minreport_summary.json").read_text())
    assert result["spearman_v1"]["zvf"] is None
    assert result["boot_ci"]["rho_v1_zvf"] == [None, None, None]
    assert result["correlation_status"]["rho_v1_zvf"] == "insufficient_data"
    assert "insufficient_data" in result["verdict_H2"]
    assert "undefined" in result["verdict_H4_per_axis"]


def test_stack_canonical_nested_reports_receive_actual_scores_and_entropy(tmp_path, monkeypatch):
    import p5_stack_axis_minreport as stack

    cells = [
        dict(
            cell_id=str(i),
            G=2 + i,
            temperature=0.5,
            seed=i,
            task_slice="task",
            model_family="model",
            zvf=0.2 * i,
            mean_reward=0.2 * i,
            pcd=i,
            mean_completion_len=10 + i,
        )
        for i in range(4)
    ]
    manifests = {}
    for i in range(4):
        manifest = fully_specified_manifest()
        manifest["cell_id"] = str(i)
        manifest["min_report"]["group_size_schedule"]["initial_g"] = i + 2
        manifest["min_report"]["heldout_split"]["description"] = "split-" + str(i % 2)
        manifests[str(i)] = manifest
        assert badge.score_manifest(manifest)["badge"] == 100
    monkeypatch.setattr(stack, "load_cells", lambda: cells)
    monkeypatch.setattr(stack, "load_manifests", lambda: manifests)
    monkeypatch.setattr(stack, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stack, "N_BOOT", 10)
    stack.main()
    result = json.loads((tmp_path / "p5_stack_axis_minreport_summary.json").read_text())
    assert result["badge_v1_mean"] == 100
    assert result["v1_total_h_bits"] == pytest.approx(3)
    assert result["v1_varying_items"] == 2
    nested = manifests["0"]
    legacy = {mr.ITEM_ALIASES.get(k, k): v for k, v in nested["min_report"].items()}
    assert subfields.normalized_min_report(nested) == subfields.normalized_min_report(legacy)


def test_coverage_summary_prose_tracks_complete_typed_report(tmp_path, monkeypatch):
    import minreport_coverage as coverage

    manifest = fully_specified_manifest()
    trace = tmp_path / "trajectory.json"
    trace.write_text('[{"step": 0, "zvf": 0.25, "gu": 0.75}]')
    manifest["min_report"]["telemetry"].update(
        source=str(trace), per_step_zvf=True, per_step_gu=True
    )
    monkeypatch.setattr(coverage, "load_manifests", lambda: [manifest])
    monkeypatch.setattr(coverage, "OUT_DIR", tmp_path)
    assert coverage.main() == 0
    result = json.loads((tmp_path / "minreport_summary.json").read_text())
    assert result["fully_covered_cells"] == 1
    assert result["per_field_validated_pct"]["2:ref_policy_kl"] == 100
    assert result["ambiguity_flags"] == []


def test_coverage_summary_prose_reports_actual_missing_denominator(tmp_path, monkeypatch):
    import minreport_coverage as coverage

    monkeypatch.setattr(coverage, "load_manifests", lambda: [{"cell_id": "missing"}])
    monkeypatch.setattr(coverage, "OUT_DIR", tmp_path)
    assert coverage.main() == 0
    result = json.loads((tmp_path / "minreport_summary.json").read_text())
    assert len(result["ambiguity_flags"]) == 7
    assert all("1/1 missing" in flag for flag in result["ambiguity_flags"])
    assert "98" not in str(result["ambiguity_flags"])

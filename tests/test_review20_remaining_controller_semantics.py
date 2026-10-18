"""Offline synthetic regressions for the remaining Review20 P7 defects."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "platform_modal/scripts/p5p8"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def joint():
    return load("p7_joint_controller")


def record(step, k, ddiv=.05):
    return dict(method="synthetic", step=step, K=k, p_hat=k/8,
                zvf_actual=float(k in (0, 8)), step_delta_div=ddiv,
                contrast_prompt=0 < k < 8, boundary_prompt=k in (0, 8))


def step_record(step, ks, ddiv=.05):
    return dict(step=step, ddiv_step=ddiv, n_prompts=len(ks),
                contrast_count=sum(0 < k < 8 for k in ks),
                prompts=[dict(K=k, p_hat=k/8, zvf_actual=float(k in (0, 8)),
                              contrast=0 < k < 8, boundary=k in (0, 8)) for k in ks])


def test_joint_step_priority_counts_and_separate_units(joint):
    rows = [record(0, 0), record(0, 4), record(1, 4, 0)]
    result = joint.joint_controller(rows, .05)
    assert result["n_fired_steps"] == 1
    assert result["n_fired_prompts"] == 2
    assert result["n_boundary_escalated"] == 1
    assert result["predicted_restored_prompts"] == 0
    assert result["observed_restored_prompts"] is None
    assert result["g_total"] == 34
    assert result["rollout_saves"] == 6
    assert result["extra_rollouts"] == 16
    assert result["net_rollouts_saved"] == -10
    assert result["net_rollouts_saved"] == result["baseline_rollouts"]-result["g_total"]
    assert "net_saves" not in result
    assert "n_zvf_saved" not in result
    assert joint.ddiv_only(rows, .05)["predicted_restored_prompts"] == 0


def test_already_contrast_and_boundary_are_not_credited_as_restored(joint):
    for k in (0, 4, 8):
        result = joint.joint_controller([record(0, k)], .05)
        assert result["predicted_restored_prompts"] == 0
        assert result["observed_restored_prompts"] is None


def test_joint_and_ci_extension_share_exact_policy_and_resampling(joint):
    extension = load("p7_multitrigger_seed_robust")
    steps = [step_record(0, [0, 4]), step_record(1, [4], 0)]
    flat = extension.flatten_step_records(steps)
    expected = joint.joint_controller(flat, .05)
    assert extension.headline_for_method(steps, .05) == expected
    assert extension.joint_controller_one_step(steps[0], .05) == (0, 0, -16, 0, 1, 2)
    intervals = joint.bootstrap_controller_metrics(flat, .05, 200, 17)
    actual = extension.bootstrap_ci_metrics(steps, .05, 200, 17)
    for metric in joint.METRICS:
        for endpoint in ("lo", "hi"):
            assert actual[f"{metric}_ci_{endpoint}"] == intervals[metric][endpoint]
    # One step is the sampling unit. Identical clusters have exact intervals.
    single = joint.bootstrap_controller_metrics(flat[:2], .05, 20, 1)
    assert single["net_rollouts_saved"] == dict(pt=-16, lo=-16, hi=-16)


def test_joint_empty_and_stable_seed(joint):
    assert joint.joint_controller([], .05)["cost_ratio"] is None
    assert joint.bootstrap_controller_metrics([], .05, 10)["cost_ratio"]["lo"] is None
    assert joint.stable_seed("gift", .05) == joint.stable_seed("gift", .05)
    assert joint.stable_seed("gift", .05) != joint.stable_seed("gift", .07)


def test_n10_triage_high_tail_and_equality():
    module = load("p7_n10_replication")
    assert not module.triage_fires(.69, .7)
    assert module.triage_fires(.7, .7)
    assert module.triage_fires(.9, .7)


def test_n10_calibration_uncertainty_does_not_collapse():
    module = load("p7_n10_replication")
    trajectories = {1: [dict(zvf=.8)], 2: [dict(zvf=.8)]}
    result = module.bootstrap_restoration(trajectories,
        dict(per_seed_differences=[.01, .20]), lambda r: True, n_boot=400, seed=1)
    assert result["delta_zvf_mean"] == pytest.approx(.105)
    assert result["delta_zvf_lo"] < result["delta_zvf_mean"] < result["delta_zvf_hi"]
    assert result["n_valid_effect_boot"] == 400


def test_n10_trajectory_uncertainty_clipping_and_adverse_gain():
    module = load("p7_n10_replication")
    trajectories = {1: [dict(zvf=.02)], 2: [dict(zvf=.9)]}
    result = module.bootstrap_restoration(trajectories,
        dict(per_seed_differences=[.2, .2]), lambda r: True, n_boot=300, seed=5)
    assert result["delta_zvf_mean"] == pytest.approx(.11)
    assert result["delta_zvf_lo"] < result["delta_zvf_hi"]
    adverse = module.bootstrap_restoration({1: [dict(zvf=.9)]},
        dict(per_seed_differences=[-.2, -.2]), lambda r: True, n_boot=20)
    assert adverse["delta_zvf_mean"] == pytest.approx(-.1)
    assert adverse["restoration_per_step_mean"] == pytest.approx(-.1)


def test_n10_no_fire_outcome_is_undefined_but_total_zero():
    module = load("p7_n10_replication")
    result = module.bootstrap_restoration({1: [dict(zvf=.1)]},
        dict(per_seed_differences=[.1, .2]), lambda r: False, n_boot=20)
    assert result["delta_zvf_mean"] is None
    assert result["delta_zvf_lo"] is None
    assert result["restoration_total_per_seed_mean"] == 0
    assert result["n_valid_effect_boot"] == 0
    assert result["effect_interval_status"] == "unavailable_no_fires"


def test_retention_resamples_baseline_with_same_indices():
    module = load("p7_iter92_asymmetric_hysteresis")
    # Highly heterogeneous baseline cannot distort a true constant .5 retention.
    result = module.bootstrap_retention_from_yields(
        [1000, 1], [1001, 3], [1000.5, 2], B=300)
    assert result["yield_retention_point"] == .5
    assert result["ci_lo"] == result["ci_hi"] == .5
    assert result["n_undefined_ratio_boot"] == 0


def test_retention_zero_denominator_is_not_perfect_retention():
    module = load("p7_iter92_asymmetric_hysteresis")
    result = module.bootstrap_retention_from_yields([0, 4], [0, 4], [0, 4], B=40)
    assert result["yield_retention_point"] is None
    assert result["yield_retention_median"] is None
    assert result["ci_lo"] is None
    assert result["ci_excludes_1.0"] is None
    assert result["n_undefined_ratio_boot"] == 40
    assert result["gain_difference_ci_lo"] == result["gain_difference_ci_hi"] == 0
    partial = module.bootstrap_retention_from_yields([0, 0], [0, 1], [0, .5], B=200)
    assert 0 < partial["n_undefined_ratio_boot"] < 200
    assert partial["ratio_status"] == "conditional_on_positive_raw_gain"
    assert partial["ci_lo"] == partial["ci_hi"] == .5


def test_threshold_uses_step_fraction_and_actual_prompt_count():
    module = load("p7_iter99_seed_threshold_robustness")
    groups = [[0]*8, [1]*8, [0]*4+[1]*4, [0]*4+[1]*4]
    tensor = [dict(method="synthetic", step=17, rewards=groups, zvf=.99)]
    low = module.replay_n2(.3, tensor)[0]
    high = module.replay_n2(.8, tensor)[0]
    assert low["zvf"] == high["zvf"] == .5
    assert low["step"] == 17
    assert low["baseline_G_prompts"] == 32
    assert low["fires_des"] == 4
    assert high["fires_des"] == 0
    assert low["savings"] == .5
    assert high["savings"] == 0
    assert low["expected_contrast_prompts"] == pytest.approx(2*68/70)
    assert low["delta_expected_contrast_prompts"] < 0
    assert high["expected_contrast_prompts"] == 2
    n10 = dict(seed=1, step_log=[dict(zvf=.5)])
    assert module.replay_n10(.3, n10)["savings"] == low["savings"]
    assert module.replay_n10(.8, n10)["savings"] == high["savings"]


def test_threshold_all_equal_never_invents_contrast():
    module = load("p7_iter99_seed_threshold_robustness")
    for value in (0, 1):
        tensor = [dict(method="synthetic", rewards=[[value]*8]*16)]
        for tau in (.3, .8):
            result = module.replay_n2(tau, tensor)[0]
            assert result["expected_contrast_prompts"] == 0
            assert result["fires_des"] == result["headroom_bad"] == 0
            assert module.subsample_contrast_probability([value]*8, 4) == 0
    assert module.deescalates(.5, .5)
    assert not module.deescalates(.99, .5)


def test_n10_main_exports_high_tail_and_noncollapsed_ci(tmp_path, monkeypatch):
    module = load("p7_n10_replication")
    trajectories = {1: [dict(step=0, zvf=.8, reward=.1)],
                    2: [dict(step=0, zvf=.2, reward=.5)]}
    monkeypatch.setattr(module, "load_n10", lambda: trajectories)
    monkeypatch.setattr(module, "fit_contrast_regression", lambda: dict(
        delta_zvf_abs=.105, per_seed_differences=[.01, .2]))
    monkeypatch.setattr(module, "N10_SEEDS", [1, 2])
    monkeypatch.setattr(module, "BOOT_N", 100)
    monkeypatch.setattr(module, "OUT", tmp_path)
    module.main()
    with (tmp_path / "p7_n10_replication.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    selected = [r for r in rows if r["controller"] == "zvf_triage" and r["threshold"] == "0.7"]
    assert [int(r["fires"]) for r in selected] == [1, 0]
    summary = json.loads((tmp_path / "p7_n10_replication_summary.json").read_text())
    result = summary["controllers"]["zvf_triage_0.7"]
    assert result["delta_zvf_lo"] < result["delta_zvf_hi"]
    assert summary["controllers"]["zvf_triage_0.9"]["delta_zvf_mean"] is None


def test_joint_main_exports_only_dimensionally_valid_metrics(tmp_path, monkeypatch):
    module = load("p7_joint_controller")
    monkeypatch.setattr(module, "METHODS", ["synthetic"])
    monkeypatch.setattr(module, "DDIV_TAU_GRID", [.05])
    monkeypatch.setattr(module, "N_BOOT", 20)
    monkeypatch.setattr(module, "load_n2_with_step_ddiv", lambda: {
        "synthetic": [record(0, 0), record(0, 4)]})
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    module.main()
    data = json.loads((tmp_path / "p7_joint_controller_summary.json").read_text())
    assert data["tau_sweep"][0]["net_rollouts_saved"] == -16
    assert data["tau_sweep"][0]["observed_restored_prompts"] is None
    assert all("net_saves" not in r for r in data["comparisons"])


def test_multitrigger_main_exports_shared_rule(tmp_path, monkeypatch):
    module = load("p7_multitrigger_seed_robust")
    monkeypatch.setattr(module, "METHODS", ["synthetic"])
    monkeypatch.setattr(module, "load_n2_records", lambda: {
        "synthetic": [step_record(0, [0, 4]), step_record(1, [4], 0)]})
    monkeypatch.setattr(module, "load_n10_seeds", lambda: [dict(
        seed=1, mean_zvf=.5, heldout_acc=.1,
        step_log=[dict(zvf=.5, y_obs=.5, ddiv=.05)])])
    monkeypatch.setattr(module, "OUT", tmp_path)
    module.main()
    data = json.loads((tmp_path / "p7_multitrigger_seed_summary.json").read_text())
    assert data["observed_restored_prompts"] is None
    with (tmp_path / "p7_joint_controller_ci.tsv").open() as f:
        row = next(csv.DictReader(f, delimiter="\t"))
    assert row["rule_id"] == data["joint_rule_id"]
    assert float(row["net_rollouts_saved"]) == -10
    assert row["observed_restored_prompts"] == ""


def test_threshold_main_reports_no_feasible_threshold(tmp_path, monkeypatch):
    module = load("p7_iter99_seed_threshold_robustness")
    n2_dir, n10_dir, out = tmp_path/"n2", tmp_path/"n10", tmp_path/"out"
    n2_dir.mkdir()
    n10_dir.mkdir()
    for method in module.METHODS:
        (n2_dir/f"{method}_s0_tensors.jsonl").write_text(json.dumps(
            dict(method=method, step=0, rewards=[[0]*8]*2))+"\n")
    for seed in (1, 2):
        (n10_dir/f"n10_grpo_s{seed}.json").write_text(json.dumps(dict(
            seed=seed, step_log=[dict(zvf=1)], heldout_acc=.1)))
    monkeypatch.setattr(module, "SEEDS", [1, 2])
    monkeypatch.setattr(module, "N2_DIR", n2_dir)
    monkeypatch.setattr(module, "N10_DIR", n10_dir)
    monkeypatch.setattr(module, "OUT_DIR", out)
    monkeypatch.setattr(module, "TAU_GRID", [.3, .8])
    module.main()
    data = json.loads((out/"p7_iter99_seed_threshold_robustness_summary.json").read_text())
    assert data["calibration_status"] == "no_feasible_threshold"
    assert data["calibrated_tau"] is None
    assert data["calibrated_point"] is None


def test_hysteresis_main_serializes_undefined_retention(tmp_path, monkeypatch):
    module = load("p7_iter92_asymmetric_hysteresis")
    monkeypatch.setattr(module, "METHODS", ["synthetic"])
    monkeypatch.setattr(module, "N_STEPS", 2)
    monkeypatch.setattr(module, "N_PROMPTS", 1)
    monkeypatch.setattr(module, "load_step_zvf", lambda: {
        ("synthetic", 0): 1, ("synthetic", 1): 1})
    monkeypatch.setattr(module, "load_step_ks", lambda: {
        ("synthetic", 0): [0], ("synthetic", 1): [8]})
    monkeypatch.setattr(module, "OUT", tmp_path)
    module.main()
    with (tmp_path / "p7_iter92_asymm_boot.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert all(r["yield_retention_point"] == "" for r in rows)
    assert all(r["ratio_status"] == "unavailable_nonpositive_raw_gain" for r in rows)

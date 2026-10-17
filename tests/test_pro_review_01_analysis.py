"""Adversarial offline fixtures for review-01 scientific producer repairs."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, directory="berkeley"):
    path = ROOT / "platform_modal/scripts" / directory / (name+".py") if directory else ROOT / "platform_modal/scripts" / (name+".py")
    spec = importlib.util.spec_from_file_location("review01_"+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "directory"), [
    ("alphastar_league_grpo", "berkeley"), ("cfr_grpo_baseline", "berkeley"),
    ("dpo_iterative_rpo_vs_grpo", "berkeley"), ("mipro_pareto_grpo", "berkeley"),
    ("niah_training_signal", "berkeley"), ("stateflow_training_fsm", "berkeley"),
    ("unpacking_dpo_ppo_factorization", "berkeley"), ("hipporag_memory", "berkeley"),
    ("p6_drgrpo_measured_evidence", "p5p8"),
])
def test_paths_and_imports_have_no_file_writes(name, directory, monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "mkdir", Mock(side_effect=AssertionError("import wrote directory")))
        module = load(name, directory)
    root = getattr(module, "ROOT", getattr(module, "WORKTREE", None))
    assert root == ROOT
    assert "protocols/pro-review-01-v2" in str(module.OUT)
    if hasattr(module, "RES"):
        assert module.RES == ROOT / "platform_hybrid/experiments/results"
    if hasattr(module, "SRC"):
        assert module.SRC == ROOT / "platform_hybrid/experiments/results/groupsize_zvf_sweep.json"


@pytest.mark.parametrize("effects", [[.12, .16], [.10, .10], [.10]])
def test_p6_task_means_never_invent_aggregate_uncertainty(effects):
    module = load("p6_drgrpo_measured_evidence", "p5p8")
    mean, low, high, significant = module.welch_pooled_tstat(effects)
    assert mean == pytest.approx(sum(effects)/len(effects))
    assert low is None
    assert high is None
    assert significant is False


def test_p6_retains_task_intervals_and_does_not_infer_proxy_sign():
    module = load("p6_drgrpo_measured_evidence", "p5p8")
    row = dict(task="task", kind="elasticity", metric="neg_frac", mean_grpo=.1, mean_drgrpo=.2, mean_diff=.1, ci_lo=-.2, ci_hi=.4, n_pairs=5)
    result = module.summarize([row])["neg_frac"]
    assert result["per_task"][0]["ci_lo"] == -.2
    assert result["predicted_sign"] is None
    assert result["verdict"] == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="Duplicate"):
        module.summarize([row, row])


def state_run(rewards, advantage=1.):
    return {"seed": 0, "group_size": 8, "step_log": [
        {"step": i, "mean_reward": r, "entropy": 1., "advantage_variance": advantage}
        for i, r in enumerate(rewards)]}


def test_stateflow_does_not_credit_continued_training_suffix():
    module = load("stateflow_training_fsm")
    record = module.analyze_run(state_run([.8]+[1.]*39))
    assert record["heldout_checkpoint_retention"] is None
    assert record["stopping_policy_status"] == "unmeasured"
    assert record["low_advantage_variance_entry"] is None
    assert record["online_usable"] is False
    assert record["gradient_inactivity_established"] is False
    assert "retain_rew" not in record
    assert module.checkpoint_retention(None, 0, 39) is None
    assert module.checkpoint_retention({0: .8, 39: 1.}, 0, 39) == .8


def test_single_state_and_missing_transitions_do_not_pass():
    module = load("stateflow_training_fsm")
    ordering = module.state_ordering([2]*4, [1]*4, [1]*4)
    assert ordering["status"] == "INCOMPLETE"
    assert ordering["reward_increases"] is False
    assert ordering["entropy_decreases"] is False
    assert module.fsm_states([0]*4, 1.) == ([0]*4, None, None)


def test_niah_unrecovered_stays_censored_for_every_width():
    module = load("niah_training_signal")
    starts = {w: module.earliest_recovery([0.]*40, 1., w) for w in module.W_GRID}
    assert all(x is None for x in starts.values())
    order = module.recovery_order(starts)
    assert order == {"recovery_coverage": 0., "status": "INCOMPLETE", "monotone_nonincreasing": False}
    assert module.recovery_order({1: 30, 5: None})["monotone_nonincreasing"] is False


def test_niah_predictor_does_not_overlap_terminal_target():
    module = load("niah_training_signal")
    assert module.disjoint_window([0]*30+[1]*10, 21) == (0, 1)
    with pytest.raises(ValueError, match="completely observed"):
        module.window_mean([1], 0, 2)


def test_dpo_claims_scale_and_marks_irpo_unmeasured(monkeypatch):
    module = load("dpo_iterative_rpo_vs_grpo")
    monkeypatch.setattr(module, "load_optimal_g", lambda: {1000000: 8})
    row = module.irpo_optimal_g_vs_iter127()[0]
    assert row["Iterative_RPO_optimal_G"] is None
    assert row["status"] == "unmeasured_hypothesis"
    beta = .2
    difference = torch.tensor(1., requires_grad=True)
    frozen = difference.detach().clone()
    loss = -torch.nn.functional.logsigmoid(beta*(difference-frozen))
    loss.backward()
    assert difference.grad.item() == pytest.approx(-beta/2)
    assert module.dpo_pair_gradient_scale(beta, 0) == beta/2
    assert module.dpo_pair_gradient_scale(1e-8, 0) == 5e-9
    difference.grad.zero_()
    (-torch.nn.functional.logsigmoid(beta*(difference-difference))).backward()
    assert difference.grad.item() == 0
    monkeypatch.setattr(module, "load_zvflink", lambda: [{"T": 1, "GU_ratio": .5}])
    assert module.dpo_implicit_reward_equivalence()[0]["iterative_rpo_feasible_at_G4"] is None


def test_matched_effect_matches_test_sample_and_duplicates_rejected():
    module = load("unpacking_dpo_ppo_factorization")
    rows = [{"algo": "grpo", "seed": i, "heldout_acc": v} for i, v in enumerate([.5, .5, .9])]
    rows += [{"algo": "ppo", "seed": i, "heldout_acc": .6} for i in range(2)]
    matched, seeds = module.match_pairs(rows)
    assert seeds == [0, 1]
    assert sum(a-b for a, b in matched)/len(matched) == pytest.approx(-.1)
    with pytest.raises(ValueError, match="Duplicate"):
        module.match_pairs(rows+[rows[0]])
    assert module.cohens_d([0, 1], [1, 2]) == pytest.approx(-math.sqrt(2))


def test_small_permutation_is_enumerated_and_equivalence_uses_interval():
    module = load("unpacking_dpo_ppo_factorization")
    assert module.exact_paired_permutation([1]*5) == 2/32
    wide = module.paired_equivalence([-.4, -.2, 0, .2, .4], .005)
    assert wide["equivalent"] is False
    narrow = module.paired_equivalence([-.0002, -.0001, 0, .0001, .0002], .005)
    assert narrow["equivalent"] is True
    assert narrow["confidence"] == .9
    assert module.paired_equivalence([0]*5, .005)["equivalent"] is False


def test_terminal_window_sorted_and_completion_checked(tmp_path):
    module = load("unpacking_dpo_ppo_factorization")
    path = tmp_path / "trace.tsv"
    header = ["method", "seed", "step", "zvf", "reward_mean", "heldout_acc", "collapse"]
    def write(steps):
        with path.open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(header)
            for step in steps:
                writer.writerow(["grpo", 0, step, 0, 0, step/10, 0])
    write([6, 1, 4, 2, 5, 3])
    rows = module.extract_varmit_rows(path, expected_final_step=6)
    assert rows[0]["terminal_acc"] == .4
    with pytest.raises(ValueError, match="expected final"):
        module.extract_varmit_rows(path, expected_final_step=7)
    write([1, 2, 3, 4, 4, 5])
    with pytest.raises(ValueError, match="Duplicate"):
        module.extract_varmit_rows(path)


def test_constant_complementarity_and_missing_saturation_fail(monkeypatch):
    module = load("alphastar_league_grpo")
    monkeypatch.setattr(module, "_read_tsv", lambda path: [
        {"metric_key": "isoG_value_table", "headline": "values: {'4': 1, '8': 1, '16': 1}"},
        {"metric_key": "isoT_value_table", "headline": "values: {'1': 1, '4': 1}"},
    ])
    rows, rho_g, rho_t, amp, decisive = module.h4_complementarity()
    assert rho_g is None
    assert rho_t is None
    assert decisive is False
    bounded = [{"metric_key": f"T={t}", "headline": "delta=-0.1"} for t in [1000000, 4000000, 16000000, 64000000]]
    monkeypatch.setattr(module, "_read_tsv", lambda path: bounded if "bounded_cone" in str(path) else [])
    assert module.h2_bounded_cone()[-1] is False
    assert module.h2_bounded_cone()[2] is False


def test_empty_complementarity_never_passes(monkeypatch):
    module = load("alphastar_league_grpo")
    monkeypatch.setattr(module, "_read_tsv", lambda path: [])
    assert module.h4_complementarity()[-1] is False


def test_cfr_fits_actual_T_and_stated_delta_times_T_quantity(monkeypatch):
    module = load("cfr_grpo_baseline")
    def source(budgets):
        return [{"metric_key": f"row_G{g}_T{t}", "headline": f"acc_emp={value}+/-0.01"}
                for t in budgets for g, value in [(32, .2), (64, .2+.01*math.sqrt(t/1000000))]]
    monkeypatch.setattr(module, "_read_tsv", lambda path: source([1000000, 4000000, 16000000, 64000000]))
    _, slope, sublinear = module.h3_belief_state_compression()
    assert slope == pytest.approx(1.5)
    assert sublinear is False
    monkeypatch.setattr(module, "_read_tsv", lambda path: source([1000000, 16000000]))
    _, slope, sublinear = module.h3_belief_state_compression()
    assert slope == pytest.approx(1.5)
    assert sublinear is False


@pytest.mark.parametrize("iterations", [1, 2, 50])
def test_ppr_triangle_conserves_mass(iterations):
    module = load("hipporag_memory")
    kg = {"nodes": ["a", "b", "c"], "deg": {"a": 1, "b": 1, "c": 1}}
    mass = module.ppr(kg, [("a", "b"), ("b", "c"), ("c", "a")], ["a"], n_iters=iterations)
    assert sum(mass.values()) == pytest.approx(1.)
    assert all(v >= 0 for v in mass.values())


def test_ppr_fixed_restart_dangling_and_unknown_seeds():
    module = load("hipporag_memory")
    kg = {"nodes": ["a", "b", "isolated"]}
    mass = module.ppr(kg, [("a", "b")], ["a"])
    assert mass["a"] == pytest.approx(1/(1+.85), abs=1e-3)
    assert sum(module.ppr(kg, [("a", "b")], ["isolated"]).values()) == pytest.approx(1)
    assert sum(module.ppr(kg, [("a", "b")], ["unmatched"]).values()) == pytest.approx(1)


def test_random_rank_is_stable_across_hash_seeds(tmp_path):
    import os
    path = ROOT / "platform_modal/scripts/berkeley/hipporag_memory.py"
    snippet = f"import runpy; m=runpy.run_path({str(path)!r}); print([p['id'] for p,_ in m['rank_random']([{{'id':str(i)}} for i in range(20)], 'q')])"
    outputs = [subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True) for seed in [1, 2]]
    assert outputs[0] == outputs[1]


def test_mipro_small_budget_and_observation_identity():
    module = load("mipro_pareto_grpo")
    anchors = [f"a{i}" for i in range(12)]
    instructions = module.INSTRUCTIONS
    oracle = {(a, i): .1+.1*j for a in anchors for j, i in enumerate(instructions)}
    for function in [module.random_strategy, module.gridsearch_strategy, module.mipro_bo_strategy]:
        result = function(anchors, instructions, 1, oracle, random.Random(0))
        assert len(result["history"]) == len(result["observations"]) == 1
        assert result["evaluations"] == 1
        assert function(anchors, instructions, 0, oracle, random.Random(0))["evaluations"] == 0
    result = module.mipro_bo_strategy(anchors, instructions, 24, oracle, random.Random(0))
    assert len({(r["anchor"], r["instruction"]) for r in result["observations"]}) == 24
    early = module.early_discovery(oracle, result["observations"])
    assert early["n_hits"] <= 8
    assert early["threshold_met"] is False
    assert early["status"] == "infeasible_legacy_global_budget"
    assert early["thresholds"]["a0"] == pytest.approx(.325)  # true quantile, not .75*max=.3


def test_synthetic_landscape_seed_changes_scores():
    module = load("mipro_pareto_grpo")
    features = {"r_mean": .3, "r_var": .1, "frac_above_0p5": .3, "RQS": .4}
    a = module.latent_score(features, "i0_baseline", 0, 0, random.Random(0))
    b = module.latent_score(features, "i0_baseline", 0, 0, random.Random(1))
    assert a != b


def test_sweep_quotes_arbitrary_paths_and_is_bounded(tmp_path):
    module = load("hyperparameter_sweep_grpo", "")
    source = tmp_path / "base config.yaml"
    source.write_text("env: {}\ntinker: {}\n")
    output = tmp_path / "space 'quote;$(never)"
    plan = module.main(["--base-config", str(source), "--output-dir", str(output), "--backend", "tinker-atropos"])
    assert len(plan["planned_attempts"]) == 3
    assert plan["training_started"] is False
    script = output / "run_sweep.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    for entry in plan["planned_attempts"]:
        assert shlex.split(module.shell_command(entry["command"])) == entry["command"]
        assert entry["backend"] == "tinker-atropos"


def test_sweep_completion_and_failed_attempt_manifest(tmp_path, monkeypatch):
    module = load("_sweep_attempt", "")
    process = Mock(return_value=types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(module.subprocess, "run", process)
    path = tmp_path / "attempt.json"
    assert module.execute(["synthetic-command"], path, "hash", 7) == 0
    assert module.execute(["synthetic-command"], path, "hash", 7) == 0
    assert process.call_count == 1
    assert json.loads(path.read_text())["empirical_validation"] == "not_performed"
    process.return_value.returncode = 17
    failed = tmp_path / "failed.json"
    assert module.execute(["synthetic-command"], failed, "hash", 7) == 17
    assert json.loads(failed.read_text())["state"] == "failed"


def test_bm25_cutoff_25_is_really_evaluated():
    module = load("hipporag_memory")
    ranked = [({"row": {}, "passage": "gold" if i == 14 else "other"}, 1) for i in range(30)]
    assert module.first_gold(ranked, "gold", k=10) == (0, -1)
    assert module.first_gold(ranked, "gold", k=25) == (1, 15)


def test_synthetic_trajectories_are_rejected():
    module = load("stateflow_training_fsm")
    run = state_run([1]*40)
    run["data_kind"] = "synthetic_fixture"
    with pytest.raises(ValueError, match="Synthetic"):
        module.analyze_run(run)


@pytest.mark.parametrize("name", ["stateflow_training_fsm", "niah_training_signal"])
def test_trajectory_main_writes_only_synthetic_fixture_directory(name, tmp_path, monkeypatch):
    module = load(name)
    source = tmp_path / "input.json"
    # Synthetic test inputs intentionally imitate the legacy untyped source schema.
    source.write_text(json.dumps({"runs": [state_run([i/50 for i in range(40)])]}))
    output = tmp_path / "output"
    monkeypatch.setattr(module, "SRC", source)
    monkeypatch.setattr(module, "OUT", output)
    result = module.main()
    assert result["confirmatory"] is False
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert len(manifest["artifacts"]) == (6 if name == "stateflow_training_fsm" else 7)
    assert "NaN" not in next(p for p in output.glob("*summary.json")).read_text()


def test_p6_main_never_mutates_registry(tmp_path, monkeypatch):
    module = load("p6_drgrpo_measured_evidence", "p5p8")
    source = tmp_path / "input.tsv"
    source.write_text("task\tmetric\tkind\tn_pairs\tmean_grpo\tmean_drgrpo\tmean_diff\tci_lo\tci_hi\nA\tneg_frac\telasticity\t5\t0.1\t0.2\t0.1\t-0.2\t0.3\n")
    registry = tmp_path / "registry.json"
    registry.write_text('{"untouched":true}')
    monkeypatch.setattr(module, "LBIAS", source)
    monkeypatch.setattr(module, "OUT", tmp_path / "output")
    monkeypatch.setattr(module, "ENTRIES", tmp_path)
    result = module.main()
    assert result["registry_update"] is False
    assert registry.read_text() == '{"untouched":true}'


def test_synthetic_optimizer_main_uses_equal_nonexhaustive_budgets(tmp_path, monkeypatch):
    module = load("mipro_pareto_grpo")
    source = tmp_path / "eureka_rqs_per_anchor.tsv"
    source.write_text("model\tr_mean\tr_var\tfrac_above_0p5\tRQS\n" + "".join(f"a{i}\t.3\t.1\t.3\t.4\n" for i in range(12)))
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    result = module.main()
    assert result["evidence_type"] == "synthetic_benchmark"
    for run in result["runs"]:
        assert {strategy["evaluations"] for strategy in run["strategies"].values()} == {24}
        assert result["evaluation_budget_per_strategy"] < 48


def test_hipporag_main_records_real_cutoffs_and_manifest(tmp_path, monkeypatch):
    module = load("hipporag_memory")
    for name, _, _ in module.CORPUS_SPEC:
        (tmp_path/name).write_text("model\tmetric\nqwen3.5-4b\trqs\nqwen3-8b\trqs\n")
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path/"out")
    module.main()
    result = json.loads((tmp_path/"out/hipporag_summary.json").read_text())
    assert result["H3_ppr_entity_mass_vs_rqs_pearson"] is None
    assert result["H5_cost_measured"] is False
    assert result["confirmatory"] is False
    assert all("bm25_hit25" in row for row in result["results"])
    manifest = json.loads((tmp_path/"out/artifact_manifest.json").read_text())
    assert len(manifest["artifacts"]) == 2

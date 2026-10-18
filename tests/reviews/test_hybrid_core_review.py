"""Offline regressions for Pro slice 06. Never launches model/service jobs.

Notebook-style legacy modules have import-time GPU/login behavior. Their pure
functions are extracted from the checked-in AST, not imported, to test adapters
without invoking those side effects. Generated candidate code is never executed.
"""

import ast
import dataclasses
import importlib.util
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from platform_hybrid.experiments import research_contracts as c
from platform_hybrid.experiments import group_size_token_normalized as gs
from platform_hybrid.experiments import stratified_heldout as sh
from platform_hybrid.experiments.measured_analysis import analyze_measured_runs, MATCH_KEYS

ROOT = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments"


def functions(path, names, **extra):
    tree = ast.parse((ROOT / path).read_text())
    body = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in body} == set(names)
    for n in body:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    ns = {
        **vars(c),
        "np": np,
        "torch": torch,
        "F": F,
        "random": random,
        "re": re,
        "json": json,
        "math": math,
        "Path": Path,
        "Any": object,
        "Optional": __import__("typing").Optional,
        "Dict": dict,
        "List": list,
        "Tuple": tuple,
        "Sequence": __import__("typing").Sequence,
        "pd": SimpleNamespace(DataFrame=object),
        **extra,
    }
    exec(  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(ROOT / path),
            "exec",
        ),
        ns,
    )
    return SimpleNamespace(**ns)


def empirical(run, **kw):
    return dict(
        execution_kind="empirical",
        measured=True,
        claim_eligible=True,
        status="completed",
        run_id=run,
        receipt_id=f"receipt:{run}",
        **kw,
    )


@pytest.mark.parametrize(
    ("file", "fn"),
    [
        ("p8_proper_eval.py", "run_p8_eval"),
        ("p1_scaled_layer_freeze.py", "run_layer_freeze_colab"),
        ("p2p3_token_budget_curriculum.py", "token_budget_optimal_curriculum"),
        ("p4_length_bias_kl_mask.py", "run_p4_kl_surprise_mask"),
    ],
)
def test_c1_scaffolds_require_explicit_quarantined_simulation(file, fn, tmp_path):
    call = getattr(functions("implementations/" + file, [fn]), fn)
    with pytest.raises(c.EvidenceUnavailable):
        call()
    result = call(simulation=True, output_dir=tmp_path)
    assert result["execution_kind"] == "simulation"
    assert not result["measured"]
    assert list(tmp_path.glob("simulations/*.json"))
    with pytest.raises(c.EvidenceUnavailable):
        c.require_empirical(result)


def test_c1_real_variance_path_never_manufactures_results():
    module = functions("variance_mitigation_integration.py", ["run_real"], MethodConfig=object)
    with pytest.raises(c.EvidenceUnavailable):
        module.run_real(None, "missing", 0)


def test_c2_measured_099_not_replaced_by_trace_surrogates():
    exp = {
        "model": "Qwen/Qwen3-8B-Base",
        "experiment_id": "run",
        "reward_trace": [0.1, 0.9],
        "last10_avg": 0.5,
    }
    rows = sh.group_checkpoints(
        [exp],
        {"run": empirical("run", checkpoint_id="saved", heldout_accuracy=0.99, heldout_n=100)},
    )
    flat = sum(rows.values(), [])
    assert len(flat) == 1
    assert flat[0]["_heldout_acc"] == 0.99
    assert not sh.group_checkpoints([exp], {})


def test_c2_missing_saved_checkpoint_evaluation_stays_missing():
    exp = {
        "model": "Qwen/Qwen3-8B-Base",
        "experiment_id": "run",
        "checkpoints": [{"checkpoint_id": "saved"}],
    }
    row = sum(sh.group_checkpoints([exp], {}).values(), [])[0]
    assert row["_heldout_acc"] is None
    assert not row["_heldout_measured"]
    assert math.isnan(sh.bootstrap_ci([None])[0])


def test_c2_empty_sweep_and_training_values_not_heldout():
    assert gs.build_rows([1_000_000], {}, [4, 8]) == []
    assert gs._row_heldout_acc(empirical("x", last10_avg=0.9, final_reward=0.9)) is None
    assert gs._bootstrap_ci([0.8]) == (0.8, None, None)
    assert gs._row_token_budget({"steps": 100, "group_size": 8}) is None


def test_c2_provenance_retained_to_tsv(tmp_path):
    rows = gs.build_rows([100], {8: [empirical("x", total_tokens=100, heldout_accuracy=0.9)]}, [8])
    gs.write_tsv(rows, tmp_path / "out.tsv")
    assert "measured_run_receipts" in (tmp_path / "out.tsv").read_text()
    assert rows[0]["heldout_acc_ci_low"] is None


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("table1_main_results", ({},)),
        ("table2_cross_library", ()),
        ("table3_gsm8k_scaling", ()),
        ("table4_ppo_vs_grpo", ({},)),
        ("trl_cross_seed_summary", ()),
    ],
)
def test_c3_legacy_statistics_require_receipts(name, args):
    call = getattr(functions("compute_statistics.py", [name]), name)
    with pytest.raises(c.EvidenceUnavailable):
        call(*args)


def test_c3_measured_analysis_preserves_run_units_and_correction():
    shared = {k: "same" for k in MATCH_KEYS}
    records = [
        empirical(f"{arm}{i}", arm=arm, comparison_id="pair", accuracy=0.1 * i + offset, **shared)
        for arm, offset in [("a", 0), ("b", 0.1)]
        for i in range(3)
    ]
    hypotheses = [
        dict(hypothesis_id="h", arm_a="a", arm_b="b", comparison_id="pair", endpoint="accuracy")
    ]
    result = analyze_measured_runs(records, hypotheses, n_boot=100)
    assert result["comparisons"][0]["n"] == [3, 3]
    assert result["comparisons"][0]["p_corrected"] == result["comparisons"][0]["p_raw"]
    with pytest.raises(c.EvidenceUnavailable):
        analyze_measured_runs(records + [records[0]], hypotheses, n_boot=100)
    records[0]["model_revision"] = "different"
    with pytest.raises(c.EvidenceUnavailable):
        analyze_measured_runs(records, hypotheses, n_boot=100)


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("reward_humaneval", "grpo_10x_runner.py"),
        ("reward_humaneval_binary", "round2_runner.py"),
        ("reward_humaneval_dense", "round2_runner.py"),
    ],
)
@pytest.mark.parametrize(
    "completion", ["", "def f(x): return 999", "def f(x): return x+1", "def f(", "while True: pass"]
)
def test_c4_all_generated_code_fails_closed_without_isolation(name, file, completion):
    grader = getattr(functions("10x_structural_ceiling/" + file, [name]), name)
    with pytest.raises(c.IsolatedEvaluatorUnavailable):
        grader(completion, "def check(candidate):\n    assert candidate(1) == 2", "f")


def test_c4_candidate_program_contains_prompt_and_check_invocation():
    program = c.humaneval_program(
        "def f(x):\n", "    return x+1", "def check(candidate):\n    assert candidate(1)==2", "f"
    )
    assert program.endswith("check(f)\n")
    assert program.startswith("def f(x):")
    compile(program, "<fixture>", "exec")  # compile only, never execute candidate code


@pytest.mark.parametrize(
    "output",
    ["{}", '{"arguments":{"x":"location"}}', "[]", '{"name":"get_weather","arguments":[]}'],
)
def test_c5_invalid_calls_never_receive_success(output):
    scorer = functions("implementations/collab/Qwen2.5-1.5B_tool_call_eval.py", ["score"]).score
    result = scorer(output, "get_current_weather", {"location": "Tokyo"})
    assert result["score"] == 0
    assert not result["task_success"]


def test_c5_wrong_tool_values_and_bool_number_confusion_fail():
    assert (
        c.tool_score(
            '{"name":"weather","arguments":{"location":"Paris"}}', "weather", {"location": "Tokyo"}
        )
        == 0
    )
    assert c.tool_score('{"name":"calc","arguments":{"n":true}}', "calc", {"n": 1}) == 0
    assert (
        c.tool_score(
            '{"name":"weather","arguments":{"location":"Tokyo"}}', "weather", {"location": "Tokyo"}
        )
        == 1
    )


def test_c5_unrelated_calls_do_not_receive_scripted_observation():
    module = functions(
        "implementations/collab/Qwen2.5-3B_multiturn_eval.py",
        ["run_scenario", "score_chain"],
        SYSTEM_PROMPT="test",
        infer=lambda model, history: '{"name":"unrelated","arguments":{}}',
    )
    scenario = dict(
        user="weather",
        tool_calls=[dict(name="weather", arguments={"location": "Tokyo"})],
        tool_responses=["secret correct weather"],
        accepted_answers=["rainy"],
    )
    responses, history = module.run_scenario(None, scenario, max_turns=1)
    assert "secret correct weather" not in str(history)
    assert "Invalid tool" in history[-1]["content"]
    assert not module.score_chain(["a sufficiently long but wrong answer"], scenario)[
        "task_success"
    ]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("#### 1,200", 1200),
        (r"\boxed{\frac{1}{2}}", 0.5),
        (r"\boxed{2} revised \boxed{999}", 999),
        (r"\boxed{2} revised \boxed{", None),
    ],
)
def test_c6_final_numeric_rule(response, expected):
    assert c.final_numeric(response) == expected


def test_c6_adapter_reward_regressions():
    for file in ["grpo_10x_runner.py", "round2_runner.py"]:
        module = functions("10x_structural_ceiling/" + file, ["reward_math", "reward_gsm8k"])
        assert module.reward_math(r"\boxed{\frac{1}{2}}", "0.5") == 1
        assert module.reward_math(r"\boxed{2} then \boxed{999}", "2") == 0
        assert module.reward_gsm8k("#### 1,200", "1200") == 1


def test_c6_nested_schema_and_density():
    assert (
        c.json_objects('schema: {"name":"t","parameters":{"properties":{"x":{"type":"object"}}}}')[
            0
        ]["parameters"]["properties"]["x"]["type"]
        == "object"
    )
    assert c.execute_fixture_tool("calculate_density", dict(population=13960000, area=2194))[
        "result"
    ] == pytest.approx(13960000 / 2194)
    assert "error" in c.execute_fixture_tool("calculate", {})


def test_c6_dpo_minus_two_is_semantically_different():
    module = functions(
        "10x_structural_ceiling/gsm8k_dpo.py",
        ["corrupt_answer"],
        random=SimpleNamespace(choice=lambda xs: -2),
    )
    assert module.corrupt_answer("1") == "-1"
    assert c.numeric_value(module.corrupt_answer("1")) != c.numeric_value("1")


def test_c7_sampler_refresh_after_every_update_independent_of_saves():
    for file in ["grpo_10x_runner.py", "round2_runner.py"]:
        text = (ROOT / "10x_structural_ceiling" / file).read_text()
        update = text.index("tc.optim_step(")
        refresh = text.index("sc = tc.save_weights_and_get_sampling_client()", update)
        checkpoint = text.index("if save_interval", update)
        assert update < refresh < checkpoint
        assert "behavior_policy_version" in text


def test_c8_completion_alignment_and_eos_padding():
    assert c.completion_alignment([11, 12], [21, 22]) == (
        [11, 12, 21],
        [12, 21, 22],
        [0.0, 1.0, 1.0],
    )
    assert c.response_token_mask([5, 2, 2, 2], 2) == [1.0, 1.0, 0.0, 0.0]
    for filename in [
        "campaign",
        "curriculum_grpo",
        "hard_curriculum",
        "groupsize_zvf",
        "parallel_sweep",
        "token_budget",
        "p4_surprise",
    ]:
        source = (ROOT / "openings" / f"{filename}.py").read_text()
        assert "completion_alignment(pid" in source
        assert "completion_mask" in source
        assert not re.search(r"fid\[1:\]\s*\+\s*\[0\]", source)


def test_c9_exact_turn_loss_uses_saved_ids_and_pre_backward_denominator():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 4, 10))
            self.device = "cpu"
            self.seen = []

        def forward(self, input_ids, attention_mask):
            self.seen.append((input_ids.tolist(), attention_mask.tolist()))
            return SimpleNamespace(logits=self.logits)

    module = functions(
        "implementations/collab/Qwen2.5-3B_multiturn_grpo.py", ["compute_reinforce_loss"]
    )
    r = c.TurnRecord((1, 2), (1, 1), (3, 4), "v0", (("temperature", 1.0),))
    model = Model()
    loss = module.compute_reinforce_loss(model, None, None, [r], 2.0, denominator=4)
    assert model.seen == [([[1, 2, 3, 4]], [[1, 1, 1, 1]])]
    assert loss == pytest.approx(math.log(10))
    assert torch.count_nonzero(model.logits.grad[:, 0]) == 0  # prompt target excluded
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.generated_ids = (99,)


def test_c10_attempt_budget_exists_and_reward_precedes_filter():
    source = (ROOT / "implementations/collab/Qwen2.5-3B_multiturn_grpo.py").read_text()
    assert "attempts < max_attempts" in source
    assert "sampled_tokens < max_sampled_tokens" in source
    assert source.index('"sampled_reward"') < source.index("if abs(rel_reward)")
    for file in ["groupsize_zvf.py", "p4_surprise.py"]:
        source = (ROOT / "openings" / file).read_text()
        assert source.index("br.extend(rr)") < source.index(
            "continue", source.index("br.extend(rr)")
        )


def test_c11_cli_command_and_arm_independent(monkeypatch):
    calls = []
    module = functions(
        "causal_zvf_experiment.py",
        ["main"],
        argparse=__import__("argparse"),
        DEFAULT_CONFIG={"model": "m"},
        run_training_arm=lambda *a: calls.append(a),
        run_bin_phase=None,
        run_evaluate_phase=None,
    )
    monkeypatch.setattr(__import__("sys"), "argv", ["prog", "train", "--phase", "dead"])
    module.main()
    assert calls[0][0] == "dead"


@pytest.mark.parametrize("rate", ["NaN", "inf", "-1"])
def test_c11_cost_rate_rejects_nonfinite_negative(monkeypatch, rate):
    module = functions("tool_use_zvf_sweep/cost_estimate.py", ["_rate"], os=__import__("os"))
    monkeypatch.setenv("RATE", rate)
    with pytest.raises(SystemExit):
        module._rate("RATE")


def test_c11_modal_runner_has_no_generated_source_patching():
    source = (ROOT / "modal_batch_runner.py").read_text()
    assert "src.replace" not in source
    assert '"--max_steps"' in source
    compile(source, "modal_batch_runner.py", "exec")


def test_c12_zero_endpoint_retained():
    module = functions("survival_analysis.py", ["_safe_float", "_coerce_record"])
    result = module._coerce_record({"last10_avg": 0.0, "steps": 10})
    assert result["last10"] == 0.0


def test_c12_binning_service_error_is_unmeasured():
    def fail(*args):
        raise RuntimeError("offline")

    module = functions(
        "causal_zvf_experiment.py",
        ["bin_prompts_by_difficulty", "wilson_ci"],
        sample_prompt=fail,
        time=__import__("time"),
    )
    examples = [{"prompt_ids": [1], "answer": "1"}]
    bins = module.bin_prompts_by_difficulty(None, None, examples, verbose=False)
    assert not any(bins.values())
    assert examples[0]["binning_status"] == "unmeasured_service_error"


def test_c13_group_zvf_is_not_equal_step_means_or_scalar_gradients():
    module = functions("tool_use_reward_analysis.py", ["_zvf_from_trace"])
    assert module._zvf_from_trace([0.5, 0.5]) is None
    assert c.group_metrics([0, 1])["reward_zvf"] == 0
    assert c.group_metrics([0.4, 0.4])["reward_zvf"] == 1
    from platform_hybrid.experiments.signal_starvation.metrics import signal_metrics

    result = signal_metrics([1, 1], [1, -1], [1, 1])
    assert result["parameter_update_norm"] is None
    assert "exact_zero_update" not in result


def test_c14_pid_direction_windup_and_entropy():
    spec = importlib.util.spec_from_file_location("pid06", ROOT / "implementations/p7_zvf_pid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pid = module.ZVF_PIDController(target_zvf=0.4)
    assert pid.step(0.8) > 8
    for _ in range(200):
        pid.step(1.0)
    assert pid.current_g == 16
    saturated_integral = pid.integral
    for _ in range(100):
        pid.step(1.0)
    assert pid.integral == saturated_integral
    target_states = [pid.step(pid.target) for _ in range(20)]
    assert len(set(target_states[1:])) == 1
    mod = functions("variance_mitigation_integration.py", ["reward_shaping"], MethodConfig=object)
    cfg = SimpleNamespace(entropy_bonus=True, beta_e=0.1)
    with pytest.raises(
        ValueError,
        match="Shared entropy cancels under group centering; provide rollout\\-specific bonuses",
    ):
        mod.reward_shaping(cfg, [0, 1], 0.5)
    shaped = mod.reward_shaping(cfg, [0, 1], [0, 1])
    assert shaped[1] - shaped[0] != 1


def test_c15_disjoint_windows_and_undefined_zero_ranking():
    mod = functions(
        "zvf_predictive_validation.py",
        ["make_record"],
        RunRecord=object,
        metric_arrays=lambda _: (np.zeros(10), np.zeros(10), np.zeros(10)),
        MIN_STEPS=15,
    )
    assert mod.make_record(Path("synthetic"), "x", {"step_log": [{}] * 10}) is None
    assert c.topk_overlap([0, 0], [0, 0], 1) is None


def test_c16_prompt_pairing_rejects_positionally_different_sets():
    mod = functions("p0_gsm8k_paired.py", ["align", "majority_correct_indicator"])
    with pytest.raises(
        ValueError, match="Prompt sets differ; declare a common prompt_order across all seeds"
    ):
        mod.align([(0, 1), (1, 0)], [(1, 0), (2, 1)])
    with pytest.raises(ValueError, match="Duplicate prompt IDs"):
        mod.align([(0, 1), (0, 0)], [(0, 1)])
    assert mod.align([(0, 1), (1, 0)], [(1, 1), (2, 0)], [1]) == ([0], [1])


def test_c16_base_group_filter_enforced():
    mod = functions(
        "base_instruct_paired.py",
        ["train_metrics", "_match_model", "_eligible"],
        _mean=lambda x: sum(x) / len(x),
    )
    assert (
        mod.train_metrics([dict(model="m", task="gsm8k", group_size=4, last10_avg=0.9)], "m")[
            "train_post_rl"
        ]
        is None
    )


def test_c17_survival_cannot_certify_absent_measurements():
    mod = functions(
        "survival_analysis.py",
        ["_f1_zvf_diagnostic", "_f3_ppo_vs_grpo", "_f4_framework_gap", "run"],
    )
    for name in ["_f1_zvf_diagnostic", "_f3_ppo_vs_grpo", "_f4_framework_gap"]:
        with pytest.raises(c.EvidenceUnavailable):
            getattr(mod, name)([])


def test_c18_distillation_padding_invariant_shifted_and_targets_required():
    torch.manual_seed(1)
    student = torch.randn(1, 4, 5, requires_grad=True)
    teacher = torch.randn(1, 4, 5)
    labels = torch.tensor([[-100, -100, 3, 2]])
    attn = torch.ones(1, 4)
    response = torch.tensor([[0, 0, 1, 1]])
    loss, task, kl = c.distillation_loss(student, teacher, labels, attn, response)
    expected = F.cross_entropy(student[:, 1:3].reshape(-1, 5), torch.tensor([3, 2]))
    assert task.item() == pytest.approx(expected.item())
    padded_student = torch.cat([student, torch.randn(1, 2, 5)], 1)
    padded_teacher = torch.cat([teacher, torch.randn(1, 2, 5)], 1)
    padded_loss, *_ = c.distillation_loss(
        padded_student,
        padded_teacher,
        torch.cat([labels, torch.full((1, 2), -100)], 1),
        torch.cat([attn, torch.zeros(1, 2)], 1),
        torch.cat([response, torch.zeros(1, 2)], 1),
    )
    assert loss.item() == pytest.approx(padded_loss.item())
    with pytest.raises(
        ValueError, match="Causal labels, attention and explicit response masks required"
    ):
        c.distillation_loss(student, teacher, None, attn, response)


def test_c19_missing_empirical_render_inputs_rejected():
    mod = functions("render_stat_rigor_tex.py", ["render"])
    with pytest.raises(c.EvidenceUnavailable):
        mod.render({"tables": {}})
    mod = functions("10x_structural_ceiling/analyze_results.py", ["generate_latex_table"])
    with pytest.raises(c.EvidenceUnavailable):
        mod.generate_latex_table(SimpleNamespace(to_dict=lambda _: []))


def test_d5_lag_joins_actual_steps():
    rows = [{"step": i, "reward": 0} for i in [1, 2, 4, 5]]
    assert [(a["step"], b["step"]) for a, b in c.lagged_pairs(rows, 2)] == [(2, 4)]


def test_changed_python_files_parse():
    import subprocess

    paths = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT.parents[1], text=True
    ).splitlines()
    for path in paths:
        if path.endswith(".py"):
            ast.parse((ROOT.parents[1] / path).read_text())


def test_c16_seed_fallback_identity_preserves_two_runs(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps(
            [
                dict(
                    seed=seed,
                    model="m",
                    task="math",
                    tag="same",
                    step_log=[dict(step=i, reward=0.5, zvf=0.0, gu=1.0) for i in range(6)],
                )
                for seed in [1, 2]
            ]
        )
    )

    @dataclasses.dataclass
    class RunData:
        key: str
        model: str
        task: str
        step_data: list

    module = functions(
        "zvf_lagged_regression.py",
        ["walk_records", "load_runs", "extract_step_data"],
        SOURCE_FILES=[path],
        RunData=RunData,
        as_float=lambda value, default=math.nan: float(value) if value is not None else default,
    )
    runs = module.load_runs()
    assert len(runs) == 2
    assert runs[0].key != runs[1].key


@pytest.mark.parametrize("all_correct", [True, False])
def test_c8_c10_mock_opening_end_to_end(monkeypatch, all_correct):
    """One local mock rollout/update/evaluation, no model/dataset/service calls."""
    import sys
    from types import ModuleType

    logs, datums, gradients = [], [], []

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Tokenizer:
        def encode(self, *args, **kwargs):
            return [11, 12]

        def decode(self, tokens, **kwargs):
            return "#### 1" if tokens[0] == 7 else "#### 9"

    class Client:
        def save_weights_for_sampler(self, **kwargs):
            return Future(SimpleNamespace(path="mock://weights"))

        def create_sampling_client(self, **kwargs):
            return self

        def sample(self, prompt, num_samples, **kwargs):
            seqs = [[7]] if num_samples == 1 else ([[7], [7]] if all_correct else [[7], [8]])
            return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=s) for s in seqs]))

        def forward_backward_custom(self, data, loss_fn):
            datums.extend(data)
            lp = [
                torch.zeros(len(d.loss_fn_inputs["target_tokens"].data), requires_grad=True)
                for d in data
            ]
            loss, _ = loss_fn(data, lp)
            loss.backward()
            gradients.extend([p.grad.tolist() for p in lp])
            return Future(SimpleNamespace(metrics={"loss": loss.item()}))

        def optim_step(self, *args, **kwargs):
            return Future(None)

    client = Client()
    tinker = ModuleType("tinker")
    types = ModuleType("tinker.types")
    tinker.ServiceClient = lambda **kwargs: SimpleNamespace(
        create_lora_training_client=lambda **kw: client
    )
    types.ModelInput = SimpleNamespace(from_ints=lambda xs: list(xs))
    types.Datum = lambda **kw: SimpleNamespace(**kw)
    types.TensorData = lambda **kw: SimpleNamespace(**kw)
    types.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    types.AdamParams = lambda **kw: SimpleNamespace(**kw)
    tinker.types = types
    wandb = ModuleType("wandb")
    wandb.init = lambda **kw: SimpleNamespace(summary={})
    wandb.log = lambda data, **kw: logs.append(data)
    wandb.finish = lambda: None
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *a, **k: [
        {"question": str(i), "answer": "#### 1"} for i in range(5)
    ]
    for name, module in [
        ("tinker", tinker),
        ("tinker.types", types),
        ("wandb", wandb),
        ("transformers", transformers),
        ("datasets", datasets),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    module = functions("openings/groupsize_zvf.py", ["run_one"])
    result = module.run_one(("synthetic-test", 2, 0, 1, 1, "mock", 2, 1e-5, 1))
    assert result["heldout_before"] == 1
    assert result["heldout_after"] == 1
    if all_correct:
        assert not datums
        assert logs[-1]["reward"] == 1
        assert logs[-1]["zvf"] == 1
    else:
        assert len(datums) == 2
        assert logs[-1]["reward"] == 0.5
        assert logs[-1]["zvf"] == 0
        assert datums[0].model_input == [11, 12]
        assert datums[0].loss_fn_inputs["target_tokens"].data == [12, 7]
        assert all(g[0] == 0 and g[1] != 0 for g in gradients)


def test_c15_detector_rejects_one_run_per_class():
    module = functions("openings/p8_detector.py", ["cv_auroc"])
    with pytest.raises(ValueError, match="two independent runs"):
        module.cv_auroc(
            np.zeros((10, 1)), np.array([0] * 5 + [1] * 5), groups=["a"] * 5 + ["b"] * 5
        )


def test_c15_online_autocorrelation_uses_prefixes(tmp_path):
    path = tmp_path / "one_tensors.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"run_id": "run", "method": "grpo", "step": i, "value": i}) for i in range(4)
        )
    )
    module = functions(
        "openings/p8_detector.py",
        ["load_steps"],
        DATA_DIR=tmp_path,
        glob=__import__("glob"),
        build_features=lambda d: {"reward_mean": d["value"]},
        lag1_autocorr=lambda values: len(values),
        SURROGATE={"aero"},
    )
    assert [r["features"]["lag1_autocorr"] for r in module.load_steps()] == [1, 2, 3, 4]


def test_c12_modal_timeout_persists_unique_terminal_receipt(monkeypatch, tmp_path):
    import subprocess

    actual_path = Path

    # Function-local pathlib import uses a bounded remapping only for /results.
    def remapped(path):
        return (
            tmp_path / path.removeprefix("/results/")
            if str(path).startswith("/results/")
            else actual_path(path)
        )

    import pathlib

    monkeypatch.setattr(pathlib, "Path", remapped)

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 3000)

    monkeypatch.setattr(subprocess, "run", timeout)
    module = functions(
        "modal_runner.py",
        ["run_experiment"],
        os=__import__("os"),
        EXPERIMENTS={"x": "x.py"},
        results_vol=SimpleNamespace(commit=lambda: None),
    )
    first = module.run_experiment("x", "x.py", 42, "invocation")
    second = module.run_experiment("x", "x.py", 42, "invocation")
    assert first["status"] == second["status"] == "timeout"
    assert first["attempt_id"] != second["attempt_id"]
    receipts = list(tmp_path.glob("invocation/x/seed_42/*/result.json"))
    assert len(receipts) == 2
    assert all(json.loads(p.read_text())["success"] is False for p in receipts)


def test_d9_resume_rejects_changed_grader_or_prompt_contract():
    manifest = dict(
        protocol_version=c.PROTOCOL_VERSION,
        code_revision="a",
        package_versions={},
        model_revision="m",
        objective="group_centered_reinforce",
        seed=1,
        rng_state={},
        data_hash="d",
        prompt_hash="p",
        grader_version=c.GRADER_VERSION,
        budgets={},
        checkpoint_paths=[],
    )
    assert c.validate_resume(manifest, manifest) == {}
    with pytest.raises(c.EvidenceUnavailable):
        c.validate_resume(manifest, {**manifest, "prompt_hash": "changed"})


def test_c6_toolbench_nested_action_parser_preserves_complete_payload():
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: json.dumps(messages))
    module = functions(
        "implementations/collab/Qwen2.5-3B_multiturn_sft.py",
        ["parse_toolbench_example"],
        tokenizer=tokenizer,
        SYSTEM_PROMPT="system",
    )
    convs = [
        {"from": "user", "value": "Do the task"},
        {
            "from": "assistant",
            "value": 'Action: nested\nAction Input: {"options":{"keys":[1,2],"x":{"y":3}}}',
        },
        {"from": "tool", "value": "done"},
        {
            "from": "assistant",
            "value": 'Action: Finish\nAction Input: {"final_answer":"Completed the requested task"}',
        },
    ]
    text = module.parse_toolbench_example({"conversations": convs})
    messages = json.loads(text)
    assert json.loads(messages[2]["content"])["arguments"]["options"]["x"]["y"] == 3


def test_c11_invalid_cap_fails_before_launch(monkeypatch):
    mod = functions("tool_use_zvf_sweep/run_sweep.py", ["main"], argparse=__import__("argparse"))
    monkeypatch.setattr(__import__("sys"), "argv", ["runner", "example.yaml", "--cap-usd", "NaN"])
    with pytest.raises(SystemExit):
        mod.main()


def test_c11_bare_sampler_id_fails_before_external_import():
    mod = functions("causal_zvf_experiment.py", ["run_evaluate_phase"])
    with pytest.raises(ValueError, match="API\\-returned sampler checkpoint paths are required"):
        mod.run_evaluate_phase(["run123"], "model", 100, "unused")


def test_c5_duplicate_and_extra_tool_fields_invalid():
    assert c.json_object('{"name":"a","name":"b"}') is None
    assert c.tool_score('{"name":"a","arguments":{},"extra":true}', "a", {}) == 0


def test_c19_measured_table_renders_authoritative_counts_and_decision():
    mod = functions("render_stat_rigor_tex.py", ["render"])
    payload = dict(
        protocol_version=c.PROTOCOL_VERSION,
        analysis_unit="independent_run",
        family_size=1,
        correction_scope="one prespecified family",
        comparisons=[
            dict(
                hypothesis_id="test_hypothesis",
                run_ids=[["a", "b"], ["c", "d"]],
                receipt_ids=[["a", "b"], ["c", "d"]],
                n=[2, 2],
                ci95=[-0.1, 0.2],
                effect=0.05,
                p_corrected=0.5,
                reject=False,
            )
        ],
    )
    text = mod.render(payload)
    assert "\\begin{tabular}" in text
    assert "2/2" in text
    assert "not rejected" in text
    payload["comparisons"][0]["n"] = [5, 5]
    with pytest.raises(c.EvidenceUnavailable):
        mod.render(payload)

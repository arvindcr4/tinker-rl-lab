"""Offline behavioral regressions for review 07; never launch provider jobs."""

import argparse
import ast
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from platform_hybrid.experiments import framework_contracts as c

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "platform_hybrid/experiments"


def load(relative):
    path = E / relative
    name = "review07_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extracted(relative, names, namespace=None):
    """Import only pure definitions from legacy modules with launch-time side effects."""
    tree = ast.parse((E / relative).read_text())
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names
    ]
    ns = {"__name__": "test_extracted", **(namespace or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, "exec"), ns)
    return SimpleNamespace(**ns)


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        ("42", "42", 1.0),
        ("1,234", "1234", 1.0),
        (r"\boxed{1}, final \boxed{2}", "2", 1.0),
        (r"<think>\boxed{2}</think>\boxed{3}", "2", 0.0),
        (r"\boxed{17} incidentally 18", "18", 0.0),
        (r"\boxed{1} then \boxed{", "1", 0.0),
        (r"<think>\boxed{2}", "2", 0.0),
        (r"\boxed{nan}", "0", 0.0),
    ],
)
def test_identical_framework_grading(response, gold, expected):
    ax = load("axolotl/gsm8k_rewards.py")
    modal = load("modal/_modal_common.py")
    live = load("tinker-runs/live_zvf_probe.py")
    assert c.score_gsm8k(response, gold) == expected
    assert ax.gsm8k_correctness_reward(
        [[{"role": "assistant", "content": response}]], answer=[gold]
    ) == [expected]
    assert modal.boxed_only_reward(response, gold) == expected
    assert modal.boxed_or_last_number_reward(response, gold) == expected
    assert live.reward_fn(response, gold) == expected


def test_reward_batch_cardinality_and_representation():
    ax = load("axolotl/gsm8k_rewards.py")
    with pytest.raises(ValueError):
        ax.gsm8k_correctness_reward(["12"])
    with pytest.raises(ValueError):
        ax.gsm8k_correctness_reward(["12", "13"], answer=["12"])
    with pytest.raises(TypeError):
        ax.gsm8k_correctness_reward([[{"content": 12}]], answer=["12"])


@pytest.mark.parametrize("algorithm,expected", [("grpo", [1.0, 1.0]), ("dr_grpo", [0.25, 0.5])])
def test_n10_length_loss_and_gradient(algorithm, expected):
    losses, gradients = [], []
    for length in (2, 4):
        lp = torch.full((length,), -1.0, requires_grad=True)
        advantage = c.token_advantage(1.0, length, 8, algorithm)
        loss = -advantage * lp.sum()
        loss.backward()
        losses.append(loss.item())
        gradients.append(lp.grad.tolist())
    assert losses == expected
    assert gradients[0] == [-expected[0] / 2] * 2
    assert gradients[1] == [-expected[1] / 4] * 4
    assert c.token_advantage(1.0, 4, 8, algorithm, 2) == c.token_advantage(1.0, 4, 8, algorithm) / 2
    with pytest.raises(ValueError):
        c.token_advantage(1.0, 0, 8, algorithm)


def test_real_sdk_datum_alignment_and_prompt_gradient_mask():
    import tinker.types as T

    datum = c.aligned_datum(T, [10, 11, 12], [20, 21])
    assert datum.model_input.to_ints() == [10, 11, 12, 20]
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 12, 20, 21]
    lp = torch.ones(4, requires_grad=True)
    c.response_logprobs(datum, lp).sum().backward()
    assert lp.grad.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_dapo_surrogate_equal_baseline_for_binary_groups_through_four():
    import tinker.types as T

    cell = load("tinker-runs/cell_runner.py")
    for size in range(2, 5):
        for rewards in itertools.product((0.0, 1.0), repeat=size):
            mean = sum(rewards) / size
            std = (sum((r - mean) ** 2 for r in rewards) / size) ** 0.5 + 1e-8
            adv = [(r - mean) / std for r in rewards]
            probe = SimpleNamespace(_ADVS=adv)
            cell.install_loss_arm(probe, "dapo")
            data = [c.aligned_datum(T, [10, 11], [20, 21]) for _ in rewards]
            lp = [torch.tensor([99.0, -1.0, -2.0], requires_grad=True) for _ in rewards]
            actual, _ = probe.loss_fn(data, lp)
            expected = torch.stack([-a * l[1:].sum() for a, l in zip(adv, lp)]).mean()
            assert torch.allclose(actual, expected)
            assert probe.OBJECTIVE_IDENTITY["canonical_algorithm_equivalent"] is False


def test_group_zvf_homogeneous_failure_and_success():
    assert c.group_zvf([[0] * 8, [1] * 8]) == 1.0
    assert c.group_zvf([[0, 1], [0, 0]]) == 0.5
    with pytest.raises(ValueError):
        c.group_zvf([])


class Done:
    def __init__(self, value):
        self.value = value

    def result(self, *args, **kwargs):
        return self.value


class FakeTraining:
    model_id = "fake:train:0"

    def __init__(self, starved=False, fail_sampling=False):
        self.starved, self.fail_sampling = starved, fail_sampling
        self.saved, self.updates, self.data = [], 0, []

    def save_weights_for_sampler(self, name):
        self.saved.append(name)
        return Done(SimpleNamespace(path="tinker://fake/" + name))

    def create_sampling_client(self, **kwargs):
        return self

    def sample(self, *args, num_samples, **kwargs):
        if self.fail_sampling:
            raise RuntimeError("offline synthetic sampling failure")
        return Done(
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(tokens=[1 if self.starved else i % 2], logprobs=[-1.0])
                    for i in range(num_samples)
                ]
            )
        )

    def forward_backward_custom(self, data, loss_fn):
        self.data.extend(data)
        lp = [
            torch.ones(len(d.loss_fn_inputs["target_tokens"].data), requires_grad=True)
            for d in data
        ]
        loss, metrics = loss_fn(data, lp)
        loss.backward()
        return Done(SimpleNamespace(metrics=metrics))

    def forward_backward(self, data, loss_fn):
        self.data.extend(data)
        return Done(SimpleNamespace(metrics={"loss": 0.0}))

    def optim_step(self, *args):
        self.updates += 1
        return Done(None)

    def save_state(self, name, **kwargs):
        return Done(SimpleNamespace(path="state://" + name))


def install_fake_provider(monkeypatch, tc):
    import tinker
    import transformers

    svc = SimpleNamespace(
        create_lora_training_client=lambda **kw: tc,
        create_sampling_client=lambda **kw: tc,
        create_training_client_from_state_with_optimizer=lambda path: tc,
    )
    monkeypatch.setattr(tinker, "ServiceClient", lambda **kw: svc)
    tok = SimpleNamespace(encode=lambda *a, **kw: [9, 8], decode=lambda ids, **kw: str(ids[0]))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tok)
    monkeypatch.setenv("TINKER_API_KEY", "offline-test-placeholder")
    return tok


def probe_args(**changes):
    return argparse.Namespace(
        **{
            **dict(
                model="test/model",
                seed=42,
                rank=4,
                steps=2,
                group=2,
                batch=1,
                lr=1e-5,
                temperature=0.8,
                top_p=0.95,
                max_tokens=8,
                max_prompt_tokens=8,
                example_limit=2,
                save_every=2,
                tag="test",
                loss="grpo",
                resume=False,
                no_wandb=True,
            ),
            **changes,
        }
    )


def test_probe_refresh_is_independent_from_persistence(monkeypatch, tmp_path):
    module = load("tinker-runs/live_zvf_probe.py")
    tc = FakeTraining()
    install_fake_provider(monkeypatch, tc)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_env_file", lambda p: None)
    monkeypatch.setattr(module, "load_gsm8k_examples", lambda *a: [("question", "1")])
    result = module.run(probe_args())
    assert tc.saved == ["s0", "s1", "s2", "final"]
    assert result["successful_updates"] == tc.updates == 2
    assert result["completion_contract_satisfied"] is True
    assert result["comparison_contract_satisfied"] is False
    assert result["objective_id"] == "group_standardized_pg_response_sum_v3"
    assert result["train_state_step"] == 2
    with pytest.raises(ValueError, match="specification"):
        module.run(probe_args(resume=True, lr=2e-5))


def test_resume_initialization_failure_preserves_prior_trace(monkeypatch, tmp_path):
    module = load("tinker-runs/live_zvf_probe.py")
    install_fake_provider(monkeypatch, FakeTraining())
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_env_file", lambda p: None)

    def fail(*args):
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr(module, "load_gsm8k_examples", fail)
    args = probe_args(resume=True)
    spec = c.experiment_spec(
        {k: v for k, v in vars(args).items() if k not in {"resume", "tag", "no_wandb"}}
    )
    prior = {
        **spec,
        "status": "failed",
        "reward_trace": [0.5],
        "step_log": [{"step": 1, "reward": 0.5}],
        "train_state_step": 1,
        "train_state_path": "state://1",
    }
    (tmp_path / "test.json").write_text(json.dumps(prior))
    with pytest.raises(RuntimeError):
        module.run(args)
    result = json.loads((tmp_path / "test.json").read_text())
    assert result["reward_trace"] == [0.5]
    assert json.loads((tmp_path / "test.prior.json").read_text()) == prior


def test_n10_fully_starved_steps_are_retained(monkeypatch, tmp_path):
    module = load("tinker-runs/scripts/n10_gsm8k_cot_seed_expansion_20260704.py")
    tc = FakeTraining(starved=True)
    install_fake_provider(monkeypatch, tc)
    monkeypatch.setattr(module, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(module, "load_gsm8k", lambda seed: ([("q", "1")], [("q", "1")]))
    args = argparse.Namespace(
        **dict(
            model="test/model",
            rank=4,
            group=2,
            batch=1,
            steps=3,
            lr=1e-5,
            temp=0.8,
            top_p=0.95,
            max_tokens=8,
            max_prompt_tokens=8,
            k_epochs=2,
            n_eval=1,
            no_wandb=True,
            wandb_project="offline",
        )
    )
    result = module.run_arm("grpo", 42, args)
    assert result["mean_zvf"] == 1.0
    assert len(result["step_log"]) == 3
    assert tc.updates == result["successful_updates"] == 0
    assert all(s["skip_reason"] == "zero_advantage_groups" for s in result["step_log"])
    assert result["ratio_clip"] is None


def test_wrong_model_reference_is_quarantined():
    m = load("tinker-runs/tinker_wandb_registry.py")
    rid = "abc:train:2"
    tr = {
        "training_run_id": rid,
        "base_uuid": "abc",
        "base_model": "Qwen/base",
        "normalized_model": "Qwen/base",
    }

    def wr(model, time):
        return dict(
            referenced_tinker_ids=[rid],
            referenced_uuids=["abc"],
            model=model,
            config={"tinker_run_id": rid},
            summary={"reward": 0.9},
            url=model,
            updated_at=time,
            normalized_model=model,
        )

    row = m.correlate([tr], [wr("Qwen/base", "1"), wr("Qwen/instruct", "2")], 30)[0]
    assert row["match_confidence"] == "conflict_or_unverified"
    assert row["wandb_url"] is row["final_metric"] is None
    assert m.normalize_model("Qwen/base") != m.normalize_model("Qwen/base-Instruct")
    rect = load("tinker-runs/rectify_correlation_with_hf.py")
    assert rect.normalize_run_id("abc") == "abc"
    assert not rect.labels_consistent("Qwen/base", "Qwen/base-Instruct")


def test_wave6_plan_is_unique_and_preserves_horizon():
    m = load("tinker-runs/wave6_resume.py")
    plan = m._compute_rerun_plan({}, 32, 0.8, 2, steps=30)
    assert len(plan) == len({p["tag"] for p in plan}) == 12
    assert {p["steps"] for p in plan} == {30}


def test_mega_identity_and_humaneval_assembly():
    m = load("tinker-runs/scripts/mega_cell_campaign_20260704.py")
    a = m.Cell("m", "gsm8k_easy", 8, 0.8, 42)
    b = m.Cell("m", "gsm8k_easy", 8, 0.8, 42, max_tokens=128)
    assert a.id != b.id
    script = m.humaneval_script(
        "", "def add(a, b):\n    return a + b", "def check(f):\n    assert f(1, 2) == 3", "add"
    )
    compile(script, "<test-only>", "exec")  # assembly only; never execute generated code
    from platform_hybrid.experiments.research_contracts import IsolatedEvaluatorUnavailable

    with pytest.raises(IsolatedEvaluatorUnavailable):
        m.run_humaneval_test(
            "", "def add(a, b):\n    return a + b", "def check(f):\n    assert f(1, 2) == 3", "add"
        )
    cells, skipped = m.ordered_cells(["m"], set(), max_tokens=77)
    assert all(x.max_tokens == 77 for x in cells)


def test_step_accounting_and_receipt():
    m = load("tinker-runs/build_claim_run_table.py")
    assert m.get_step({"config": {"steps": 200}, "summary": {"train/global_step": 3}}) == 3
    assert m.get_step({"config": {"steps": 200}}) is None
    with pytest.raises(ValueError):
        c.require_completed({"peak": 0.5})
    assert c.completion_receipt(200, 3, "run", "checkpoint")["status"] == "partial"


def test_export_branches_and_missing_identity():
    m = load("tinker-runs/parallel_push_hf.py")
    assert m.branch_of("h2h2_grpo_G4_s42_a") != m.branch_of("h2h2_grpo_G8_s42_a")
    with pytest.raises(ValueError):
        c.export_identity({"checkpoint": "sampler://1"})
    with pytest.raises(ValueError):
        m.file_hashes(SimpleNamespace(siblings=[]))


def test_inventory_structured_names_and_errors(monkeypatch):
    m = load("tinker-runs/modal_inventory.py")
    mapping = {
        "/": [
            {"path": "dir.with.dots", "type": "directory"},
            {"path": "extensionless", "type": "file"},
        ],
        "dir.with.dots": [{"path": "dir.with.dots/result.json", "type": "file"}],
    }
    monkeypatch.setattr(m, "vol_ls", lambda vol, path: mapping[path])
    assert m.walk("v") == ["dir.with.dots/result.json", "extensionless"]

    def fail(*a):
        raise RuntimeError("listing unavailable")

    monkeypatch.setattr(m, "vol_ls", fail)
    with pytest.raises(RuntimeError):
        m.walk("v")


def test_audit_only_pools_compatible_completed_estimands():
    m = load("tinker-runs/wandb_zvf_audit_agg.py")
    config = {"model": "m", "comparison_hash": "spec", "metric_definitions": {"zvf": "group-v1"}}
    runs = [
        ("finished", config, {"completion_contract_satisfied": True, "zvf": 1.0}, "r1"),
        ("crashed", config, {"completion_contract_satisfied": False, "zvf": 0.0}, "r2"),
        (
            "finished",
            {**config, "metric_definitions": {"zvf": "retained-only"}},
            {"completion_contract_satisfied": True, "zvf": 0.0},
            "r3",
        ),
    ]
    result = m.aggregate(runs)
    assert len(result["groups"]) == 2
    assert result["excluded"]["completion_unverified"] == 1
    assert all(row["zvf"]["n"] == 1 for row in result["groups"].values())


def test_h2h_missing_zvf_stays_missing_and_duplicate_rejected():
    m = load("tinker-runs/h2h_summarize.py")
    spec = c.experiment_spec({"model": "m", "seed": 42})
    record = {
        **spec,
        **c.completion_receipt(1, 1, "r", "c"),
        "tag": "a",
        "objective_id": "surrogate",
        "step_log": [{}],
    }
    manifest = {"cells": {"a": spec}}
    result = m.summarize([record], manifest)
    assert result["by_objective"]["surrogate"]["zvf"] == {"mean": None, "n": 0}
    with pytest.raises(ValueError, match="duplicate"):
        m.summarize([record, {**record, "last10_avg": 0.5}], manifest)


def test_heldout_sampling_failures_do_not_become_wrong_answers(monkeypatch):
    shared = load("modal/_modal_common.py")
    monkeypatch.setitem(sys.modules, "_modal_common", shared)
    # Suppress optional codecarbon import side effects.
    monkeypatch.setitem(sys.modules, "codecarbon", None)
    module = load("modal/modal_heldout_eval.py")
    tc = FakeTraining(fail_sampling=True)
    tok = install_fake_provider(monkeypatch, tc)
    monkeypatch.setattr(module, "_get_tokenizer", lambda model: tok)
    result = module.evaluate_checkpoint(
        {"checkpoint": "tinker://fake", "model": "m"}, [("q1", "1"), ("q2", "2")]
    )
    assert result["status"] == "incomplete"
    assert result["heldout_n"] == 2
    assert result["heldout_accuracy"] is None
    assert all(p["reward"] is None and "error" in p for p in result["per_problem"])


def test_tokenizer_fallback_is_not_silent(monkeypatch):
    shared = load("modal/_modal_common.py")
    monkeypatch.setitem(sys.modules, "_modal_common", shared)
    monkeypatch.setitem(sys.modules, "codecarbon", None)
    module = load("modal/modal_heldout_eval.py")
    import transformers

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic tokenizer lookup failure")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="fallback forbidden"):
        module._get_tokenizer("unknown-model")


def test_export_needs_matching_receipt_and_verifies_commit(monkeypatch, tmp_path):
    module = load("tinker-runs/parallel_push_hf.py")
    source = {
        "checkpoint": "tinker://actual",
        "model_revision": "model-commit",
        "config_hash": "spec",
    }
    record = {**source, **c.completion_receipt(1, 1, "r", "tinker://actual")}
    hashes = {"adapter_model.safetensors": "weight-sha", "adapter_config.json": "config-sha"}
    info = SimpleNamespace(
        sha="old",
        siblings=[SimpleNamespace(rfilename=k, lfs=None, blob_id=v) for k, v in hashes.items()],
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps({"source": {**source, "checkpoint": "wrong"}, "file_hashes": hashes})
    )
    calls = []
    api = SimpleNamespace(
        model_info=lambda *a, **kw: SimpleNamespace(
            sha="new-immutable-commit"
            if kw.get("revision") == "new-immutable-commit"
            else info.sha,
            siblings=info.siblings,
        ),
        create_branch=lambda *a, **kw: calls.append("branch"),
        upload_file=lambda **kw: SimpleNamespace(oid="new-immutable-commit"),
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    result = module.push("tag", record, api=api, download=lambda *a, **kw: str(receipt_path))
    assert result["revision"] == "new-immutable-commit"
    assert result["status"] == "verified"
    assert len(calls) == 2  # Existing weights alone never permit a skip.
    assert calls[1][1]["check"] is True
    with pytest.raises(ValueError, match="differs"):
        c.export_identity({**record, "checkpoint": "different"})


def test_backfill_identity_tracks_content(monkeypatch, tmp_path):
    module = load("tinker-runs/wandb_backfill.py")
    monkeypatch.setattr(module, "REPO", tmp_path)
    source = tmp_path / "result.json"
    source.write_text('{"reward": 0.5}')
    configs = []
    run = SimpleNamespace(summary={}, log_artifact=lambda a: None, finish=lambda: None)

    def init(**kwargs):
        configs.append(kwargs["config"])
        return run

    monkeypatch.setattr(module.wandb, "init", init)
    monkeypatch.setattr(
        module.wandb, "Artifact", lambda **kw: SimpleNamespace(add_file=lambda p: None)
    )
    seen = set()
    assert module.log_file("p", "display", source, [], "notes", seen)
    assert not module.log_file("p", "renamed", source, [], "notes", seen)
    source.write_text('{"reward": 0.7}')
    assert module.log_file("p", "display", source, [], "notes", seen)
    assert configs[0]["source_identity"] != configs[1]["source_identity"]


def test_wave6_legacy_and_changed_horizon_not_reused():
    module = load("tinker-runs/wave6_resume.py")
    exp = module._compute_rerun_plan({}, 32, 0.8, 2, 30)[0]
    legacy = {exp["tag"]: {"status": "completed"}}
    with pytest.raises(ValueError):
        module._compute_rerun_plan(legacy, 32, 0.8, 2, 30)
    record = {**c.wave6_spec(exp), **c.completion_receipt(30, 30, "r", "c")}
    with pytest.raises(ValueError):
        module._compute_rerun_plan({exp["tag"]: record}, 32, 0.8, 2, 200)
    plan = module._compute_rerun_plan({exp["tag"]: record}, 32, 0.8, 2, 30)
    assert exp["tag"] not in {e["tag"] for e in plan}


def test_comparison_rejects_missing_effective_settings_and_overlap():
    requested = {
        k: "fixed"
        for k in (
            "model_revision",
            "tokenizer_revision",
            "template_hash",
            "data_revision",
            "scorer_id",
            "objective_id",
            "trainable_modules",
            "loss_reduction",
            "kl",
            "sampling",
            "optimizer",
            "budget",
            "selection_rule",
            "environment_digest",
            "code_revision",
        )
    }
    requested.update(train_item_ids=["train"], dev_item_ids=["dev"], test_item_ids=["test"])
    receipt = {
        **c.completion_receipt(2, 2, "r", "c"),
        "observed_prompts": 4,
        "observed_completions": 8,
        "observed_tokens": 16,
    }
    assert c.validate_comparison(requested, requested, receipt)["comparison_contract_satisfied"]
    with pytest.raises(ValueError, match="unavailable"):
        c.validate_comparison(requested, {**requested, "model_revision": None}, receipt)
    overlap = {**requested, "dev_item_ids": ["train"]}
    with pytest.raises(ValueError, match="disjoint"):
        c.validate_comparison(overlap, overlap, receipt)


def test_n10_actual_datums_use_response_and_full_rollout_batch_normalization(monkeypatch, tmp_path):
    module = load("tinker-runs/scripts/n10_gsm8k_cot_seed_expansion_20260704.py")
    tc = FakeTraining()
    install_fake_provider(monkeypatch, tc)

    def sample(*args, num_samples, **kwargs):
        sequences = [
            SimpleNamespace(tokens=[0], logprobs=[-1.0]),
            SimpleNamespace(tokens=[1, 1, 1], logprobs=[-1.0, -1.0, -1.0]),
        ]
        return Done(SimpleNamespace(sequences=sequences[:num_samples]))

    tc.sample = sample
    monkeypatch.setattr(module, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(module, "load_gsm8k", lambda seed: ([("q", "1")], [("q", "1")]))
    args = argparse.Namespace(
        **dict(
            model="test/model",
            rank=4,
            group=2,
            batch=1,
            steps=1,
            lr=1e-5,
            temp=0.8,
            top_p=0.95,
            max_tokens=8,
            max_prompt_tokens=8,
            k_epochs=1,
            n_eval=1,
            no_wandb=True,
            wandb_project="offline",
        )
    )
    result = module.run_arm("grpo", 42, args)
    assert result["successful_updates"] == 1
    assert result["completion_contract_satisfied"] is True
    advantages = [d.loss_fn_inputs["advantages"].data for d in tc.data]
    assert advantages[0] == pytest.approx([0.0, -0.5])
    assert advantages[1] == pytest.approx([0.0, 1 / 6, 1 / 6, 1 / 6])
    assert all(d.loss_fn_inputs["target_tokens"].data[-1] != 8 for d in tc.data)


def test_no_global_wandb_logging_patches_in_parallel_training():
    for relative in ("campaign_v2.py", "wave6_ablations.py", "scripts/tinker_parallel_runner.py"):
        tree = ast.parse((E / "tinker-runs" / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assert not any(
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "wandb"
                    and t.attr == "log"
                    for t in node.targets
                )


def test_verl_subprocess_failure_stops_before_metrics_or_export(monkeypatch):
    import os
    import subprocess
    import datasets
    import wandb

    path = E / "modal/modal_grpo_verl.py"
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_verl_qwen3_8b"
    )
    node.decorator_list = []
    namespace = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    dataset = SimpleNamespace(column_names=["question", "answer"], to_parquet=lambda p: None)
    dataset.map = lambda *args, **kwargs: dataset
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "_vram_patched", True, raising=False)
    monkeypatch.setattr(wandb, "login", lambda **kwargs: None)
    exits = []
    run = SimpleNamespace(
        id="orchestrator", entity="test", url="offline", finish=lambda **kw: exits.append(kw)
    )
    monkeypatch.setattr(wandb, "init", lambda **kw: run)
    import uuid

    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="exact-trainer-id"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Failure must not read telemetry or upload")

    monkeypatch.setattr(wandb, "Api", forbidden)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))
    result = namespace["run_verl_qwen3_8b"]()
    assert result["status"] == "failed"
    assert result["last10_avg"] is None
    assert result["successful_updates"] is None
    assert result["trainer_run_id"] == "exact-trainer-id"
    assert exits == [{"exit_code": 1}]


def test_resume_hash_is_verified_not_just_compared():
    spec = c.experiment_spec({"steps": 1})
    corrupt = copy.deepcopy(spec)
    corrupt["requested_config"]["steps"] = 2
    with pytest.raises(ValueError, match="specification"):
        c.validate_resume(corrupt, corrupt)


def test_delegated_sources_are_inspected_not_inferred(tmp_path):
    cli = tmp_path / "cli.py"
    cli.write_text('PRESETS: dict = {"gsm8k": {"group_size": 4, "lr": 3e-5}}\n')
    tinker = c.read_tinker_preset(cli)
    assert tinker["preset"]["group_size"] == 4
    assert tinker["preset"]["lr"] == 3e-5
    assert len(tinker["source_sha256"]) == 64
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("trainer:\n  policy: {}\ntrainer:\n  logger: wandb\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        c.read_skyrl_recipe(duplicate)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("trainer:\n  policy:\n    model:\n      path: other-model\n")
    with pytest.raises(ValueError, match="does not match"):
        c.read_skyrl_recipe(wrong)

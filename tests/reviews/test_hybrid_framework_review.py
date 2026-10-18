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
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, "exec"), ns)  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
    return SimpleNamespace(**ns)


@pytest.mark.parametrize(
    ("response", "gold", "expected"),
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
    with pytest.raises(ValueError, match="One answer is required per completion"):
        ax.gsm8k_correctness_reward(["12"])
    with pytest.raises(ValueError, match="One answer is required per completion"):
        ax.gsm8k_correctness_reward(["12", "13"], answer=["12"])
    with pytest.raises(TypeError):
        ax.gsm8k_correctness_reward([[{"content": 12}]], answer=["12"])


@pytest.mark.parametrize(
    ("algorithm", "expected"), [("grpo", [1.0, 1.0]), ("dr_grpo", [0.25, 0.5])]
)
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
    with pytest.raises(
        ValueError, match="Positive response length, token budget and batch size required"
    ):
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
            expected = torch.stack([-a * l[1:].sum() for a, l in zip(adv, lp, strict=True)]).mean()
            assert torch.allclose(actual, expected)
            assert probe.OBJECTIVE_IDENTITY["canonical_algorithm_equivalent"] is False


def test_group_zvf_homogeneous_failure_and_success():
    assert c.group_zvf([[0] * 8, [1] * 8]) == 1.0
    assert c.group_zvf([[0, 1], [0, 0]]) == 0.5
    with pytest.raises(ValueError, match="Nonempty finite prompt groups required"):
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
    with pytest.raises(ValueError, match="Verified completion receipt required"):
        c.require_completed({"peak": 0.5})
    assert c.completion_receipt(200, 3, "run", "checkpoint")["status"] == "partial"


def test_export_branches_and_missing_identity():
    m = load("tinker-runs/parallel_push_hf.py")
    assert m.branch_of("h2h2_grpo_G4_s42_a") != m.branch_of("h2h2_grpo_G8_s42_a")
    with pytest.raises(
        ValueError, match="Export requires exact checkpoint, model revision and config hash"
    ):
        c.export_identity({"checkpoint": "sampler://1"})
    with pytest.raises(ValueError, match="Complete adapter weights/config missing"):
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
    with pytest.raises(ValueError, match="Invalid or legacy immutable specification"):
        module._compute_rerun_plan(legacy, 32, 0.8, 2, 30)
    record = {**c.wave6_spec(exp), **c.completion_receipt(30, 30, "r", "c")}
    with pytest.raises(
        ValueError, match="Immutable experiment specification mismatch; start a new attempt"
    ):
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
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
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


@pytest.mark.parametrize("ownership", ["conflicting", "matching", "single", "malformed"])
def test_registry_explicit_owner_aliases_must_be_unique_and_consistent(ownership):
    module = load("tinker-runs/tinker_wandb_registry.py")
    a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa:train:0"
    b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb:train:0"
    runs = [
        dict(
            training_run_id=rid,
            base_uuid=rid.split(":")[0],
            base_model="Qwen/base",
            normalized_model="qwen/base",
        )
        for rid in (a, b)
    ]
    config = {"tinker_run_id": a}
    if ownership != "single":
        config["training_run_id"] = {"conflicting": b, "matching": a, "malformed": [a]}[ownership]
    wandb = dict(
        referenced_tinker_ids=[a, b],
        referenced_uuids=[],
        model="Qwen/base",
        config=config,
        summary={"reward": 0.9},
        url="https://wandb.invalid/same-run",
        normalized_model="qwen/base",
    )
    first, second = module.correlate(runs, [wandb], 30)
    if ownership in {"matching", "single"}:
        assert first["provenance_verified"] is True
        assert first["final_metric"] == 0.9
    else:
        assert first["provenance_verified"] is False
        assert first["wandb_url"] is first["final_metric"] is None
    assert second["provenance_verified"] is False
    assert second["wandb_url"] is second["final_metric"] is None


def offline_training_module(monkeypatch, relative, training=None):
    """Import full training modules while replacing every external boundary."""
    import datasets
    import wandb

    monkeypatch.setitem(sys.modules, "codecarbon", None)
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-placeholder")
    install_fake_provider(monkeypatch, training or FakeTraining())
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **kw: [{"question": "q", "answer": "#### 1"}]
    )
    monkeypatch.setattr(wandb, "login", lambda **kw: None)
    monkeypatch.setattr(wandb, "init", lambda **kw: None)
    return load(relative)


def test_current_wave6_initialization_failure_is_retryable(monkeypatch):
    module = offline_training_module(monkeypatch, "tinker-runs/wave6_ablations.py")
    resume = load("tinker-runs/wave6_resume.py")
    exp = resume._compute_rerun_plan({}, 32, 0.8, 2, 30)[0]

    def fail():
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr(module, "_load_gsm8k", fail)
    failed = module.run_one(exp)
    assert failed["status"] == "failed"
    assert failed["successful_updates"] == 0
    assert failed["completion_contract_satisfied"] is False
    c.validate_resume(c.wave6_spec(exp), failed)
    assert exp in resume._compute_rerun_plan({exp["tag"]: failed}, 32, 0.8, 2, 30)


class CrashedExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, *args):
        class Crash:
            def result(self):
                raise RuntimeError("synthetic process crash")

        return Crash()


def test_wave6_launch_worker_crash_keeps_retryable_spec(monkeypatch):
    module = offline_training_module(monkeypatch, "tinker-runs/wave6_ablations.py")
    resume = load("tinker-runs/wave6_resume.py")
    exp = module.build_experiments()[0]
    monkeypatch.setattr(module, "build_experiments", lambda: [exp])
    monkeypatch.setattr(module, "ProcessPoolExecutor", CrashedExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module, "_save", lambda *a, **kw: None)
    (failed,) = module.launch()
    assert failed["status"] == "crashed"
    assert failed["successful_updates"] is None
    assert exp["tag"] in {
        e["tag"] for e in resume._compute_rerun_plan({exp["tag"]: failed}, 32, 0.8, 2, module.STEPS)
    }


def test_wave6_resume_worker_crashes_remain_retryable(monkeypatch, tmp_path):
    module = load("tinker-runs/wave6_resume.py")
    path = tmp_path / "wave6.json"
    path.write_text(json.dumps({"metadata": {"steps": 30}, "runs": []}))
    monkeypatch.setenv("WAVE6_RESULTS_PATH", str(path))
    monkeypatch.setattr(sys, "argv", ["wave6_resume.py", "1"])
    monkeypatch.setitem(sys.modules, "wave6_ablations", SimpleNamespace(run_one=lambda exp: None))
    monkeypatch.setattr(module, "ProcessPoolExecutor", CrashedExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    module.main()
    saved = json.loads(path.read_text())
    assert len(saved["runs"]) == 12
    assert all(r["status"] == "crashed" and r["successful_updates"] is None for r in saved["runs"])
    assert (
        len(module._compute_rerun_plan({r["tag"]: r for r in saved["runs"]}, 32, 0.8, 2, 30)) == 12
    )


def test_live_resume_without_checkpoint_leaves_original_bytes_untouched(monkeypatch, tmp_path):
    module = load("tinker-runs/live_zvf_probe.py")
    install_fake_provider(monkeypatch, FakeTraining())
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_env_file", lambda p: None)

    def forbidden(*args, **kwargs):
        pytest.fail("No initialization should occur without a resumable checkpoint")

    monkeypatch.setattr(module, "load_gsm8k_examples", forbidden)
    import wandb

    monkeypatch.setattr(wandb, "init", forbidden)
    args = probe_args(resume=True, no_wandb=False)
    spec = c.experiment_spec(
        {k: v for k, v in vars(args).items() if k not in {"resume", "tag", "no_wandb"}}
    )
    prior = {
        **spec,
        "status": "failed",
        "run_id": "old:train:0",
        "reward_trace": [0.5],
        "step_log": [{"step": 1, "reward": 0.5}],
    }
    path = tmp_path / "test.json"
    before = json.dumps(prior, indent=3).encode()
    path.write_bytes(before)
    with pytest.raises(RuntimeError, match="no train_state_path.*new attempt tag"):
        module.run(args)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def n8_args(tmp_path, **changes):
    return argparse.Namespace(
        **{
            **dict(
                out_dir=str(tmp_path),
                smoke=False,
                k=64,
                model="test/model",
                n_prompts=78,
                temperature=0.8,
                top_p=0.95,
                max_tokens=512,
                seed=42,
                chunk_size=16,
                concurrency=1,
            ),
            **changes,
        }
    )


def test_n8_smoke_does_not_replace_full_manifest_or_break_full_resume(monkeypatch, tmp_path):
    import asyncio

    module = load("tinker-runs/scripts/n8_passrate_spectrum_20260704.py")
    normal = n8_args(tmp_path)
    spec = c.experiment_spec({**vars(normal), "scorer_id": c.SCORER_ID})
    legacy = tmp_path / "experiment_spec.json"
    before = json.dumps(spec, indent=3)
    legacy.write_text(before)
    samples = tmp_path / "passrates.jsonl"
    sample_bytes = json.dumps({"prompt_id": "old", "k_obtained": 64, "p_hat": 0.5}) + "\n"
    samples.write_text(sample_bytes)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic dataset stop before providers")

    monkeypatch.setattr(module, "load_dataset", fail)
    smoke = n8_args(tmp_path, smoke=True, k=4, n_prompts=2, chunk_size=4)
    for args in (smoke, normal, smoke, normal):
        with pytest.raises(RuntimeError, match="synthetic dataset stop"):
            asyncio.run(module.run_all(args))
    assert legacy.read_text() == before
    assert samples.read_text() == sample_bytes
    c.validate_resume(spec, json.loads((tmp_path / "passrates.spec.json").read_text()))
    smoke_spec = c.experiment_spec({**vars(smoke), "scorer_id": c.SCORER_ID})
    c.validate_resume(smoke_spec, json.loads((tmp_path / "passrates_smoke.spec.json").read_text()))


def test_n8_orphaned_mode_manifest_is_not_overwritten(monkeypatch, tmp_path):
    import asyncio

    module = load("tinker-runs/scripts/n8_passrate_spectrum_20260704.py")
    args = n8_args(tmp_path)
    before = json.dumps(c.experiment_spec({**vars(args), "scorer_id": c.SCORER_ID}))
    manifest = tmp_path / "passrates.spec.json"
    manifest.write_text(before)
    with pytest.raises(ValueError, match="specification mismatch"):
        asyncio.run(module.run_all(n8_args(tmp_path, k=32)))
    assert manifest.read_text() == before


@pytest.mark.parametrize("relative", ["wave6_ablations.py", "campaign_v2.py"])
@pytest.mark.parametrize(("count", "empty_tokens"), [(0, False), (1, False), (3, False), (2, True)])
def test_training_rejects_wrong_group_population_before_optimizer(
    monkeypatch, relative, count, empty_tokens
):
    tc = FakeTraining()
    tc.sample = lambda *a, **kw: Done(
        SimpleNamespace(
            sequences=[
                SimpleNamespace(tokens=[] if empty_tokens else [1], logprobs=[-1.0])
                for _ in range(count)
            ]
        )
    )
    module = offline_training_module(monkeypatch, "tinker-runs/" + relative, tc)
    exp = dict(
        tag="audit",
        model="test/model",
        sweep="baseline",
        wave="audit",
        rank=32,
        temperature=0.8,
        batch=1,
        steps=1,
        group_size=2,
        seed=42,
        lr=1e-5,
    )
    result = module.run_one(exp) if relative.startswith("wave6") else module.run_experiment(exp)
    assert result["status"] == "failed"
    assert "sample population" in result["error"]
    assert tc.updates == 0
    assert tc.data == []


@pytest.mark.parametrize(("count", "empty_tokens"), [(0, False), (1, False), (3, False), (2, True)])
def test_mega_rejects_wrong_group_population_without_completed_artifacts(
    monkeypatch, tmp_path, count, empty_tokens
):
    import asyncio
    from types import MethodType

    module = load("tinker-runs/scripts/mega_cell_campaign_20260704.py")
    monkeypatch.setattr(module, "STOP_PATH", tmp_path / "STOP")
    monkeypatch.setattr(module, "PROMPTS_PER_CELL", 1)

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    async def sample_async(**kwargs):
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[] if empty_tokens else [1]) for _ in range(count)]
        )

    async def tokenizer_async(model):
        return SimpleNamespace(encode=lambda *a, **kw: [9, 8], decode=lambda *a, **kw: "1")

    rows, failures = [], []

    async def write_outputs(cell, row, *args):
        rows.append(row)

    async def record_failure(cell, reason):
        failures.append(reason)

    state = SimpleNamespace(
        args=SimpleNamespace(max_prompt_tokens=1024, cell_timeout_sec=10),
        stop_requested=False,
        completed=0,
        failed=0,
        examples={"gsm8k_easy": [{"question": "q", "answer": "1", "source_idx": 0}]},
        svc=SimpleNamespace(
            create_sampling_client=lambda **kw: SimpleNamespace(sample_async=sample_async)
        ),
        token_lock=asyncio.Lock(),
        cumulative_tokens=0,
        tokenizer_for_async=tokenizer_async,
        _write_cell_outputs=write_outputs,
        _record_failure=record_failure,
        wandb=SimpleNamespace(
            log_cell=lambda *a: pytest.fail("Malformed cell cannot be logged as complete")
        ),
    )
    state.run_cell = MethodType(module.Campaign.run_cell, state)
    asyncio.run(
        module.Campaign._run_with_sem(
            state, module.Cell("test/model", "gsm8k_easy", 2, 0.8, 42), asyncio.Semaphore(1)
        )
    )
    assert state.completed == 0
    assert state.failed == 1
    assert failures
    assert not rows


@pytest.mark.parametrize(
    "completion",
    [
        "    return a + b\n",
        "```python\n    return a + b\n```",
        "    def helper():\n        return a + b\n    return helper()\n",
    ],
)
def test_humaneval_body_completion_keeps_indentation_and_is_not_executed(completion):
    module = load("tinker-runs/scripts/mega_cell_campaign_20260704.py")
    from platform_hybrid.experiments.research_contracts import IsolatedEvaluatorUnavailable

    prompt = 'def add(a, b):\n    """Return the sum."""\n'
    tests = "def check(f):\n    assert f(1, 2) == 3"
    compile(module.humaneval_script(prompt, completion, tests, "add"), "<assembly-only>", "exec")
    with pytest.raises(IsolatedEvaluatorUnavailable):
        module.run_humaneval_test(prompt, completion, tests, "add")


def test_wave6_late_failure_retains_observed_trace_and_retry_identity(monkeypatch):
    tc = FakeTraining()
    original_sample = tc.sample
    calls = []

    def sample(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("synthetic failure after one update")
        return original_sample(*args, **kwargs)

    tc.sample = sample
    module = offline_training_module(monkeypatch, "tinker-runs/wave6_ablations.py", tc)
    exp = dict(
        tag="late",
        sweep="baseline",
        rank=32,
        temperature=0.8,
        batch=1,
        steps=2,
        group_size=2,
        seed=42,
        lr=1e-5,
    )
    result = module.run_one(exp)
    c.validate_resume(c.wave6_spec(exp), result)
    assert result["status"] == "failed"
    assert result["successful_updates"] == tc.updates == 1
    assert result["trainer_run_id"] == tc.model_id
    assert len(result["reward_trace"]) == len(result["step_log"]) == 1
    assert result["completion_contract_satisfied"] is False


@pytest.mark.parametrize("pool_count", [0, 1])
def test_mega_incomplete_prompt_pool_stops_before_sampler(monkeypatch, pool_count):
    import asyncio

    module = load("tinker-runs/scripts/mega_cell_campaign_20260704.py")
    monkeypatch.setattr(module, "PROMPTS_PER_CELL", 2)

    async def tokenizer_async(model):
        return None

    state = SimpleNamespace(
        tokenizer_for_async=tokenizer_async,
        examples={"gsm8k_easy": [{"question": "q", "answer": "1"}] * pool_count},
        svc=SimpleNamespace(
            create_sampling_client=lambda **kw: pytest.fail("Incomplete pool cannot sample")
        ),
    )
    with pytest.raises(ValueError, match="Fixed prompt population incomplete"):
        asyncio.run(
            module.Campaign.run_cell(state, module.Cell("test/model", "gsm8k_easy", 2, 0.8, 42))
        )


@pytest.mark.parametrize(
    ("rewards", "lengths"),
    [([], []), ([[]], [[]]), ([[1]], [[1]]), ([[0, 1]], [[1]]), ([[0, 1]], [[1, 0]])],
)
def test_mega_metrics_reject_incomplete_vectors(rewards, lengths):
    module = load("tinker-runs/scripts/mega_cell_campaign_20260704.py")
    with pytest.raises(ValueError, match="fixed-size nonempty"):
        module.metric_summary(rewards, lengths, 2)


@pytest.mark.parametrize("recover", [False, True])
def test_n8_chunks_validate_each_response_without_accumulating_partial_samples(
    monkeypatch, tmp_path, recover
):
    import asyncio
    import tinker.types as T

    module = load("tinker-runs/scripts/n8_passrate_spectrum_20260704.py")
    counts = iter([1, 3, 2, 2]) if recover else itertools.cycle([1, 3])
    calls = []

    async def sample_async(**kwargs):
        count = next(counts)
        calls.append(count)
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1]) for _ in range(count)])

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(module.wandb, "log", lambda *a, **kw: None)
    path = tmp_path / "samples.jsonl"
    state = dict(prompts_done=0, completions_done=0, sum_p=0.0, p_hats=[])
    args = n8_args(tmp_path, k=4, n_prompts=1, chunk_size=2)
    task = module.process_prompt(
        "item",
        "question",
        "1",
        args,
        SimpleNamespace(encode=lambda *a, **kw: [9], decode=lambda *a, **kw: "1"),
        SimpleNamespace(sample_async=sample_async),
        T.SamplingParams(max_tokens=8, temperature=0.8),
        asyncio.Semaphore(1),
        asyncio.Lock(),
        path,
        state,
    )
    if recover:
        asyncio.run(task)
        row = json.loads(path.read_text())
        assert calls == [1, 3, 2, 2]
        assert row["k_obtained"] == row["n_correct"] == state["completions_done"] == 4
        assert row["rewards"] == [1, 1, 1, 1]
    else:
        with pytest.raises(RuntimeError, match="Incomplete fixed sample population"):
            asyncio.run(task)
        assert not path.exists()
        assert state["prompts_done"] == state["completions_done"] == 0
        assert len(calls) == 8

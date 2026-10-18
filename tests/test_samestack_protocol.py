"""CPU-only synthetic protocol regressions. No data/model downloads or training."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from platform_hybrid.experiments.samestack_protocol import (
    GRADING_SPEC_ID,
    PROTOCOL_HASH,
    PROTOCOL_ID,
    boxed_reward,
    budget_accounting,
    clear_inherited_generation_defaults,
    completion_mask_rows,
    exact_number,
    generation_configs,
    last_boxed_content,
    prompt_schedules,
    resolved_token_ids,
    strict_json_dump,
)


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/samestack_grading_v2.json").read_text())["cases"],
)
def test_exact_grading_spec(case):
    assert boxed_reward(case["text"], case["gold"]) == case["reward"]
    if case.get("status") == "unparsable":
        malformed = last_boxed_content(case["text"])
        assert exact_number(malformed) is None
        assert boxed_reward(r"\boxed{42}", malformed) == 0


def test_grouping_reward_contract_has_new_identity_and_preserves_v1_fixture():
    fixtures = Path(__file__).parent / "fixtures"
    old = json.loads((fixtures / "samestack_grading_v1.json").read_text())
    current = json.loads((fixtures / "samestack_grading_v2.json").read_text())
    assert current["spec_id"] == GRADING_SPEC_ID == "boxed-exact-rational-v2"
    assert current["cases"][: len(old["cases"])] == old["cases"]
    assert PROTOCOL_ID == "samestack-gsm8k-cot-v4-policy-exact-eos-grouping"
    assert PROTOCOL_HASH != "5ede5418a2078388f1e124d42db0506847300f97405b84d2d56dd752dd1a46e7"


def test_policy_prompt_stream_does_not_shift_for_critic_warmup():
    state = random.getstate()
    grpo = prompt_schedules(42, 7473, 30, 0, 8)
    ppo = prompt_schedules(42, 7473, 30, 4, 8)
    assert grpo["policy"] == ppo["policy"]
    assert grpo["policy_sampling_seed"] == ppo["policy_sampling_seed"]
    assert ppo["warmup_sampling_seed"] != ppo["policy_sampling_seed"]
    assert grpo["policy"][0] != ppo["critic_warmup"][0]
    assert random.getstate() == state
    g2 = prompt_schedules(42, 7473, 30, 0, 32)
    assert all(
        g8row == g2row[:8] for g8row, g2row in zip(grpo["policy"], g2["policy"], strict=True)
    )


def test_budget_is_policy_matched_not_total_exposure_matched():
    baseline = budget_accounting(30, 64, 8, 0)
    critic = budget_accounting(30, 64, 8, 4)
    assert baseline["policy_prompt_presentations"] == critic["policy_prompt_presentations"] == 240
    assert baseline["total_prompt_presentations"] == 240
    assert critic["total_prompt_presentations"] == 272
    assert baseline["total_generations"] == 1920
    assert critic["total_generations"] == 2176
    assert critic["total_exposure_matched"] is False


@pytest.mark.parametrize("metric", ["zvf", "mean_reward", "value_loss_pre_fit"])
def test_strict_json_rejects_nonfinite_but_accepts_inapplicable_null(tmp_path, metric):
    with (tmp_path / "result.json").open("w") as handle:
        strict_json_dump({"zvf": None, "t_statistic": None}, handle)
    with pytest.raises(ValueError, match="Out of range float"):
        with (tmp_path / "bad.json").open("w") as handle:
            strict_json_dump({metric: float("nan")}, handle)


@pytest.mark.parametrize("pad", [0, 2, 3])
def test_eos_masks_include_first_stop_exclude_padding_and_keep_left_prompt(pad):
    torch = pytest.importorskip("torch")
    from platform_hybrid.experiments.samestack_protocol import completion_mask
    from platform_hybrid.experiments.samestack_training import rollout_attention

    rows = [[2, pad, pad, pad], [9, 3, pad, pad], [9, 8, 7, 6], [9, 2, pad, pad]]
    expected = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]
    comp = torch.tensor(rows)
    mask = completion_mask(comp, [2, 3])
    assert mask.tolist() == expected
    assert completion_mask_rows(rows, [2, 3]) == expected
    prompt_mask = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]])
    attention, same_mask = rollout_attention(prompt_mask, comp, [2, 3])
    assert torch.equal(attention[:, :3], prompt_mask)
    assert torch.equal(attention[:, 3:], mask)
    assert torch.equal(mask, same_mask)


def test_loss_gradient_includes_eos_and_excludes_subsequent_padding():
    torch = pytest.importorskip("torch")
    from platform_hybrid.experiments.samestack_protocol import completion_mask
    from platform_hybrid.experiments.samestack_training import clipped_response_loss

    log_probs = torch.zeros((1, 4), requires_grad=True)
    mask = completion_mask(torch.tensor([[9, 2, 2, 2]]), [2, 3]).float()
    loss = clipped_response_loss(
        log_probs, torch.zeros_like(log_probs), torch.ones(1), mask, 0.2, 1
    )
    loss.backward()
    assert log_probs.grad.tolist() == [[-0.5, -0.5, 0.0, 0.0]]


def test_token_ids_include_both_eos_but_not_distinct_pad():
    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def get_vocab(self):
            return {"<|im_end|>": 3}

    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=2),
        generation_config=SimpleNamespace(eos_token_id=[2, 3]),
    )
    assert resolved_token_ids(Tokenizer(), model) == ([2, 3], 0)


@pytest.mark.parametrize("pad_equals_eos", [False, True])
def test_generation_scores_match_recomputed_raw_log_probs(pad_equals_eos):
    """Compare normalized generation scores, not merely old/current ratio=1.

    The downloaded model's old generation defaults are deliberately poisoned.
    A tiny randomly initialized CPU GPT-2 tests the actual generation path.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from platform_hybrid.experiments.samestack_training import (
        completion_log_probs,
        rollout_attention,
    )

    torch.manual_seed(123)
    pad = 2 if pad_equals_eos else 0
    config = transformers.GPT2Config(
        vocab_size=31,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        eos_token_id=[2, 3],
        pad_token_id=pad,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.1
    model.generation_config.suppress_tokens = [8]
    train, evaluation = generation_configs([2, 3], pad, 5)
    clear_inherited_generation_defaults(model, train)
    assert model.generation_config.suppress_tokens is None
    for generation in (train, evaluation):
        assert generation.top_k == 0
        assert generation.top_p == generation.temperature == generation.repetition_penalty == 1.0
        assert generation.num_beams == 1
        assert generation.eos_token_id == [2, 3]
    assert train.do_sample
    assert not evaluation.do_sample
    prompt = torch.tensor([[pad, pad, 7, 8], [pad, 4, 5, 6]])
    prompt_attention = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    train.return_dict_in_generate = True
    train.output_scores = True
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt, attention_mask=prompt_attention, generation_config=train
        )
        completion = generated.sequences[:, prompt.shape[1] :]
        attention, mask = rollout_attention(prompt_attention, completion, [2, 3])
        recomputed, _ = completion_log_probs(model, generated.sequences, attention, prompt.shape[1])
        generation_scores = torch.stack(generated.scores, dim=1).float()
        sampled = generation_scores.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(sampled[mask], recomputed[mask], rtol=2e-5, atol=2e-6)
    assert PROTOCOL_ID != "samestack_gsm8k_cot_v2"


def test_source_payload_hashes_verified_and_tampering_rejected(tmp_path):
    from platform_hybrid.experiments.samestack_manifest import verify_source_files
    from platform_hybrid.experiments.samestack_protocol import file_sha256

    names = [
        "samestack_manifest",
        "samestack_protocol",
        "samestack_training",
        "samestack_contract",
        "samestack_dependencies",
        "modal/modal_samestack_gsm8k_cot_v2",
    ]
    source = {"source_files": {}}
    for name in names:
        relative = f"platform_hybrid/experiments/{name}.py"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source only\n")
        source["source_files"][relative] = file_sha256(path)
    verify_source_files(source, tmp_path)
    path.write_text("# changed source\n")
    with pytest.raises(ValueError, match="Source payload hash mismatch"):
        verify_source_files(source, tmp_path)


def test_generation_defaults_cleared_on_peft_style_wrapper():
    transformers = pytest.importorskip("transformers")
    underlying = SimpleNamespace(
        generation_config=transformers.GenerationConfig(top_k=20, repetition_penalty=1.1)
    )
    wrapper = SimpleNamespace(
        generation_config=underlying.generation_config, get_base_model=lambda: underlying
    )
    train, _ = generation_configs([2, 3], 2, 5)
    clear_inherited_generation_defaults(wrapper, train)
    assert wrapper.generation_config.top_k == 0
    assert underlying.generation_config.top_k == 0
    assert underlying.generation_config.repetition_penalty == 1.0


@pytest.mark.parametrize("smoke", [False, True])
def test_actual_generation_and_source_manifest_fit_completion_contract(tmp_path, smoke):
    """Synthetic bundle only: no pretrained model, dataset, optimizer or experiment."""
    import gzip
    from samestack_fixtures import synthetic_run
    from platform_hybrid.experiments.samestack_contract import load_completed, write_completion
    from platform_hybrid.experiments.samestack_manifest import capture_source
    from platform_hybrid.experiments.samestack_protocol import canonical_hash

    record, texts = synthetic_run("ppo_8x8", smoke=smoke)
    train, evaluation = generation_configs([2, 3], 2, 512)
    manifest = record["manifest"]
    generation = {"train": train.to_dict(), "eval": evaluation.to_dict()}
    manifest["generation"] = generation
    manifest["effective_config"]["generation"] = generation
    record["generation"] = generation
    manifest["source"] = capture_source()
    manifest["config_hash"] = canonical_hash(manifest["effective_config"])
    identity = {
        "protocol_hash": manifest["protocol_hash"],
        "config_hash": manifest["config_hash"],
        "environment": manifest["environment"],
        "environment_hash": manifest["environment_hash"],
        "source_commit": manifest["source"]["commit"],
        "source_dirty_hash": manifest["source"]["dirty_hash"],
    }
    manifest["run_id"] = canonical_hash(identity)
    path = tmp_path / "ppo_8x8_s42.json"
    path.write_text(json.dumps(record, allow_nan=False))
    with gzip.open(tmp_path / "ppo_8x8_s42_texts.json.gz", "wt") as handle:
        json.dump(texts, handle, allow_nan=False)
    adapter = tmp_path / "adapters/ppo_8x8_s42"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"FAKE ARTIFACT: NOT MODEL WEIGHTS")
    for name in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json"):
        (adapter / name).write_text('{"synthetic": true}')
    for phase, config in generation.items():
        (adapter / f"{phase}_generation_config.json").write_text(
            json.dumps(config, allow_nan=False)
        )
    write_completion(path, expected=identity)
    assert load_completed(path, expected=identity) is not None
    assert load_completed(path, expected=dict(identity, source_dirty_hash="0" * 64)) is None


def test_core_colab_import_has_no_service_or_execution_side_effects(monkeypatch):
    import importlib.util
    import subprocess
    import sys

    monkeypatch.setenv("SAMESTACK_EXECUTION_ENV", "colab")
    original_modal = sys.modules.get("modal")

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not run commands or experiments")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    path = (
        Path(__file__).resolve().parents[1]
        / "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
    )
    spec = importlib.util.spec_from_file_location("samestack_import_only_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app is None
    assert module.run_arm is module.run_arm_local
    assert sys.modules.get("modal") is original_modal

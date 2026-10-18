"""Synthetic CPU contracts: no optimizers, model downloads or provider calls."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from platform_hybrid.experiments.student_distillation import (
    PROTOCOL_VERSION,
    student_trajectory_loss,
)


class ToyModel(torch.nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(7, dtype=torch.float) + offset)
        self.generated = []
        self.scored = []
        self.next_token = 3
        self.fail = False

    def generate(
        self, *, input_ids, attention_mask, generation_config, max_new_tokens, use_model_defaults
    ):
        assert not self.training
        assert use_model_defaults is False
        assert not torch.is_grad_enabled()
        assert generation_config.do_sample
        assert generation_config.temperature == 1.0
        assert generation_config.top_k == 0
        assert generation_config.top_p == 1.0
        assert generation_config.num_beams == 1
        self.generated.append(input_ids.clone())
        if self.fail:
            raise RuntimeError("synthetic generation failure")
        # Different response lengths plus padding after EOS exercise token weighting.
        continuation = [self.next_token, 6, 6] if input_ids[0, 0] == 1 else [6, 6, 6]
        return torch.cat((input_ids, input_ids.new_tensor([continuation[:max_new_tokens]])), dim=1)

    def forward(self, input_ids, attention_mask):
        assert not self.training
        self.scored.append((input_ids.clone(), attention_mask.clone()))
        # History-dependent logits detect incorrect causal alignment.
        scale = input_ids.cumsum(1).unsqueeze(-1) / 10.0
        return SimpleNamespace(logits=self.weight * scale)


def run(student, teacher, **kwargs):
    return student_trajectory_loss(
        student,
        teacher,
        torch.tensor([[0, 1, 2], [4, 5, 0]]),
        torch.tensor([[0, 1, 1], [1, 1, 0]]),
        max_new_tokens=3,
        eos_token_id=6,
        pad_token_id=6,
        behavior_version=kwargs.pop("behavior_version", 0),
        **kwargs,
    )


def test_student_samples_teacher_only_scores_and_token_mean_matches_manual_kl():
    student, teacher = ToyModel(), ToyModel()
    with torch.no_grad():
        teacher.weight.copy_(torch.flip(teacher.weight, dims=[0]))
    loss, outputs, records = run(student, teacher)
    assert len(student.generated) == 2
    assert teacher.generated == []
    assert student.generated[0].tolist() == [[1, 2]]
    assert student.generated[1].tolist() == [[4, 5]]
    assert records[0].input_ids == (1, 2, 3, 6)
    assert records[1].input_ids == (4, 5, 6)
    assert records[0].response_mask == (False, False, True, True)
    assert all(row.protocol_version == PROTOCOL_VERSION for row in records)
    settings = json.loads(records[0].sampling_config_json)
    assert settings["top_k"] == 0
    assert settings["temperature"] == 1.0
    assert settings["max_new_tokens"] == 3
    assert torch.equal(student.scored[0][0], teacher.scored[0][0])
    assert student.scored[0][1].tolist() == [[True] * 4, [True, True, True, False]]
    # Response targets at columns 2,3 and 2 use causal logits 1,2 and 1.
    s = torch.stack((outputs.logits[0, 1], outputs.logits[0, 2], outputs.logits[1, 1]))
    ids = student.scored[0][0]
    t = teacher.weight * ids.cumsum(1).unsqueeze(-1) / 10.0
    t = torch.stack((t[0, 1], t[0, 2], t[1, 1]))
    expected = (
        torch.nn.functional.kl_div((s / 2).log_softmax(-1), (t / 2).softmax(-1), reduction="none")
        .sum(-1)
        .mean()
        * 4
    )
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert student.weight.grad is not None
    assert torch.isfinite(student.weight.grad).all()
    assert teacher.weight.grad is None
    assert student.training
    assert teacher.training


def test_every_call_regenerates_and_records_are_immutable():
    student, teacher = ToyModel(), ToyModel()
    _, _, old = run(student, teacher)
    student.next_token = 2
    _, _, fresh = run(student, teacher, behavior_version=1)
    assert len(student.generated) == 4
    assert old[0].input_ids == (1, 2, 3, 6)
    assert fresh[0].input_ids == (1, 2, 2, 6)
    assert fresh[0].behavior_version == 1
    with pytest.raises(AttributeError):
        old[0].input_ids = ()


def test_failure_restores_modes():
    student, teacher = ToyModel(), ToyModel()
    teacher.eval()
    student.fail = True
    with pytest.raises(RuntimeError, match="synthetic"):
        run(student, teacher)
    assert student.training
    assert not teacher.training


@pytest.mark.parametrize("mask", [[[0, 0, 0]], [[1, 0, 1]], [[1, 2, 1]]])
def test_invalid_prompts_fail_closed(mask):
    with pytest.raises(ValueError, match="prompt"):
        student_trajectory_loss(
            ToyModel(),
            ToyModel(),
            torch.tensor([[1, 2, 3]]),
            torch.tensor(mask),
            max_new_tokens=3,
            eos_token_id=6,
            pad_token_id=6,
            behavior_version=0,
        )


def test_no_eos_generation_budget_and_zero_kl():
    student, teacher = ToyModel(), ToyModel()
    loss, _, records = student_trajectory_loss(
        student,
        teacher,
        torch.tensor([[1, 2]]),
        torch.ones((1, 2)),
        max_new_tokens=1,
        eos_token_id=6,
        pad_token_id=6,
        behavior_version=0,
    )
    assert records[0].input_ids == (1, 2, 3)
    assert loss.item() == pytest.approx(0, abs=1e-6)


def test_trainer_consumes_prompt_only_and_refreshes_each_call():
    from platform_hybrid.experiments.implementations.trl_distillation import (
        OnPolicyDistillationTrainer,
    )

    cls = OnPolicyDistillationTrainer
    trainer = cls.__new__(cls)
    trainer.teacher_model = ToyModel()
    trainer.max_new_tokens = 3
    trainer.eos_token_id = trainer.pad_token_id = 6
    trainer.temperature = 2.0
    trainer.state = SimpleNamespace(global_step=9)
    model = ToyModel()
    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2))}
    trainer.compute_loss(model, inputs)
    assert trainer.last_trajectories[0].behavior_version == 9
    trainer.compute_loss(model, inputs)
    assert len(model.generated) == 2
    with pytest.raises(ValueError, match="prompt"):
        trainer.compute_loss(model, {**inputs, "labels": inputs["input_ids"]})


def test_real_transformers_generation_with_random_tiny_cpu_models():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=7,
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=16,
        eos_token_id=6,
        pad_token_id=6,
    )
    # Randomly initialized tiny models, never loaded or optimized.
    student, teacher = GPT2LMHeadModel(config), GPT2LMHeadModel(config)
    student.generation_config.top_k = 2  # Must not leak into this protocol.
    loss, _, records = student_trajectory_loss(
        student,
        teacher,
        torch.tensor([[1, 2]]),
        torch.ones((1, 2)),
        max_new_tokens=2,
        eos_token_id=6,
        pad_token_id=6,
        behavior_version=0,
    )
    assert torch.isfinite(loss)
    assert 3 <= len(records[0].input_ids) <= 4
    assert json.loads(records[0].sampling_config_json)["top_k"] == 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_tokenizer_compatibility_checks_both_ids_and_pipeline():
    from platform_hybrid.experiments.implementations.trl_distillation import (
        validate_tokenizer_compatibility as validate,
    )

    def tokenizer(pipeline="same", vocab=None):
        return SimpleNamespace(
            get_vocab=lambda: {"a": 0, "b": 1} if vocab is None else vocab,
            eos_token_id=1,
            pad_token_id=1,
            bos_token_id=None,
            is_fast=True,
            backend_tokenizer=SimpleNamespace(to_str=lambda: pipeline),
        )

    student = SimpleNamespace(config=SimpleNamespace(vocab_size=2))
    teacher = SimpleNamespace(config=SimpleNamespace(vocab_size=2))
    validate(tokenizer(), tokenizer(), student, teacher)
    with pytest.raises(ValueError, match="pipelines"):
        validate(tokenizer(), tokenizer("different"), student, teacher)
    with pytest.raises(ValueError, match="vocabularies"):
        validate(tokenizer(), tokenizer(vocab={"a": 1, "b": 0}), student, teacher)
    with pytest.raises(ValueError, match="distinct"):
        validate(tokenizer(), tokenizer(), student, student)
    slow = tokenizer()
    slow.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizers"):
        validate(tokenizer(), slow, student, teacher)


def test_real_trainer_initialization_and_loss_without_training(tmp_path):
    pytest.importorskip("accelerate", minversion="1.1.0")
    from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
    from platform_hybrid.experiments.implementations.trl_distillation import (
        OnPolicyDistillationTrainer,
    )

    config = GPT2Config(vocab_size=7, n_layer=1, n_head=1, n_embd=8, n_positions=16)
    student, teacher = GPT2LMHeadModel(config), GPT2LMHeadModel(config)
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {str(i): i for i in range(7)},
        eos_token_id=6,
        pad_token_id=6,
        bos_token_id=None,
        is_fast=True,
        backend_tokenizer=SimpleNamespace(to_str=lambda: "synthetic identical pipeline"),
    )
    trainer = OnPolicyDistillationTrainer(
        model=student,
        teacher_model=teacher,
        student_tokenizer=tokenizer,
        teacher_tokenizer=tokenizer,
        eos_token_id=6,
        pad_token_id=6,
        max_new_tokens=1,
        args=TrainingArguments(
            output_dir=str(tmp_path / "fresh"),
            use_cpu=True,
            report_to=[],
            remove_unused_columns=False,
        ),
    )
    loss = trainer.compute_loss(
        student, {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2))}
    )
    assert torch.isfinite(loss)
    assert trainer.protocol_version == PROTOCOL_VERSION
    assert student.config.distillation_protocol == PROTOCOL_VERSION
    assert all(not p.requires_grad for p in teacher.parameters())
    assert trainer.optimizer is None

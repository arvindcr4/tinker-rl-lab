"""CPU seed regressions with inert loaders, trainers and publication methods."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")
from transformers import BatchEncoding

from platform_hybrid.experiments.implementations import trl_distillation as distillation
from platform_hybrid.experiments.student_distillation import student_trajectory_loss


class SamplingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(7))
        self.config = SimpleNamespace(vocab_size=7)
        self.device = torch.device("cpu")

    def generate(self, input_ids, *, max_new_tokens=None, generation_config=None, **kwargs):
        budget = max_new_tokens if generation_config is None else generation_config.max_new_tokens
        tokens = torch.randint(0, 6, (1, budget))
        return torch.cat((input_ids, tokens), dim=1)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 7))


class Tokenizer:
    eos_token_id = pad_token_id = 6
    bos_token_id = None
    eos_token = "eos"
    is_fast = True
    backend_tokenizer = SimpleNamespace(to_str=lambda: "identical synthetic pipeline")

    def get_vocab(self):
        return {str(i): i for i in range(7)}

    def __call__(self, prompt, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

    def decode(self, tokens, **kwargs):
        return ",".join(map(str, tokens.tolist()))


class InertTrainer:
    """Capture arguments without optimizer steps, writes or provider calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]

    def train(self, **kwargs):
        assert kwargs == {"resume_from_checkpoint": False}

    def save_model(self, *args):
        pass

    def push_to_hub(self):
        pass


@pytest.mark.parametrize("entry_point", ["off_policy", "fixed_history"])
def test_legacy_entry_points_seed_models_sampling_and_trainer(monkeypatch, entry_point):
    loads = []

    def load_model(*args, **kwargs):
        model = SamplingModel()
        loads.append((torch.initial_seed(), model.weight.detach().clone()))
        return model

    monkeypatch.setattr(
        distillation, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=load_model)
    )
    monkeypatch.setattr(
        distillation,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
    )
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: None, finish=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(SFTConfig=SimpleNamespace, SFTTrainer=InertTrainer)
    )
    monkeypatch.setattr(distillation, "TrainingArguments", SimpleNamespace)
    monkeypatch.setattr(distillation, "FixedHistoryDistillationTrainer", InertTrainer)
    monkeypatch.setattr(distillation.torch.cuda, "empty_cache", lambda: None)

    def run(seed, ambient_seed):
        loads.clear()
        torch.manual_seed(ambient_seed)
        config = distillation.DistillationConfig(seed=seed)
        if entry_point == "off_policy":
            trainer = distillation.train_off_policy_distillation(config, ["synthetic prompt"])
            samples = trainer.kwargs["train_dataset"]["completion"]
        else:
            trainer = distillation.train_fixed_history_distillation(
                config, SimpleNamespace(column_names=["labels", "response_mask"])
            )
            samples = None
        assert [item[0] for item in loads] == [seed, seed]
        assert trainer.args.seed == seed
        assert trainer.args.data_seed == seed
        return torch.stack([item[1] for item in loads]), samples

    with torch.random.fork_rng(devices=[]):
        first_weights, first_samples = run(371, 1)
        repeated_weights, repeated_samples = run(371, 999)
        changed_weights, changed_samples = run(982, 1)
    assert torch.equal(first_weights, repeated_weights)
    assert first_samples == repeated_samples
    assert not torch.equal(first_weights, changed_weights)
    if entry_point == "off_policy":
        assert first_samples != changed_samples


def test_student_trajectory_entry_point_preserves_caller_seed_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(distillation, "OnPolicyDistillationTrainer", InertTrainer)
    args = SimpleNamespace(
        seed=371,
        data_seed=982,
        output_dir=str(tmp_path / "fresh"),
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
    )
    student, teacher = SamplingModel(), SamplingModel()
    before = torch.random.get_rng_state().clone()
    trainer = distillation.train_on_policy_distillation(
        distillation.DistillationConfig(alpha=0.0, seed=42),
        SimpleNamespace(column_names=["input_ids", "attention_mask"]),
        student=student,
        teacher=teacher,
        student_tokenizer=Tokenizer(),
        teacher_tokenizer=Tokenizer(),
        training_args=args,
    )
    assert trainer.args is args
    assert (args.seed, args.data_seed) == (371, 982)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_student_sampling_replays_caller_rng_and_advances_without_reseeding():
    with torch.random.fork_rng(devices=[]):
        student, teacher = SamplingModel(), SamplingModel()

        def sample():
            _, _, records = student_trajectory_loss(
                student,
                teacher,
                torch.tensor([[1, 2]]),
                torch.ones((1, 2)),
                max_new_tokens=8,
                eos_token_id=6,
                pad_token_id=6,
                behavior_version=0,
            )
            return records[0].input_ids[2:]

        torch.manual_seed(371)
        expected_first = tuple(torch.randint(0, 6, (8,)).tolist())
        expected_second = tuple(torch.randint(0, 6, (8,)).tolist())
        expected_state = torch.random.get_rng_state().clone()
        torch.manual_seed(371)
        first, second = sample(), sample()
        assert first == expected_first
        assert second == expected_second
        assert torch.equal(torch.random.get_rng_state(), expected_state)
        torch.manual_seed(371)
        assert sample() == first

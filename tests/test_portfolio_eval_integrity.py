"""Offline provider-boundary regressions for source-validation receipts."""

import json
import socket
import sys
from types import SimpleNamespace

import pytest

from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator
from platform_tinker.tinkerrl.grpo import InMemoryDataset, TrainingExample


@pytest.fixture
def harness(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("portfolio integration must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("TINKER_API_KEY", "tml-" + "offline-test-only")
    # The live entry point selects online tracking; restore process state after
    # the fake integration so later tests retain their own tracking policy.
    monkeypatch.setenv("WANDB_MODE", "offline")
    calls = []
    finished = []
    run = SimpleNamespace(
        id="offline",
        url="offline",
        summary={},
        log=lambda *_a, **_k: None,
        finish=lambda **kw: finished.append(kw["exit_code"]),
    )
    sequence = SimpleNamespace(tokens=[7, 8], stop_reason="stop")
    state = SimpleNamespace(
        sequences=[sequence],
        score=1.0,
        prompt_length=3,
        mutate=False,
        sampler_requests=[],
        tokenizer_requests=[],
        tracking_config=None,
    )
    tokenizer = SimpleNamespace(
        encode=lambda *_a, **_k: list(range(state.prompt_length)),
        decode=lambda ids, **_k: "answer",
    )

    def sample(prompt, **kwargs):
        calls.append((list(prompt), kwargs))
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=state.sequences))

    rows = [
        TrainingExample(prompt=suite, metadata={"suite_id": suite, "source_id": "fixture"})
        for suite in ("api_bank_rlvr_train", "swe_gym_train")
    ]
    monkeypatch.setattr(
        evaluator, "make_pavlov_non_xlam_dataset", lambda **_: InMemoryDataset(train=[], test=rows)
    )

    def score(*_):
        if state.mutate:
            for response in state.sequences:
                response.tokens.clear()
        return state.score

    monkeypatch.setattr(evaluator, "PavlovNonXLAMReward", lambda: SimpleNamespace(score=score))

    def init_tracking(**kwargs):
        state.tracking_config = kwargs["config"]
        return run

    def create_sampler(**kwargs):
        state.sampler_requests.append(kwargs)
        return SimpleNamespace(sample=sample)

    def load_tokenizer(*args, **kwargs):
        state.tokenizer_requests.append((args, kwargs))
        return tokenizer

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init_tracking))
    types = SimpleNamespace(
        ModelInput=SimpleNamespace(from_ints=lambda ids: ids), SamplingParams=lambda **kw: kw
    )
    monkeypatch.setitem(
        sys.modules,
        "tinker",
        SimpleNamespace(
            types=types,
            ServiceClient=lambda **_: SimpleNamespace(create_sampling_client=create_sampler),
        ),
    )
    monkeypatch.setitem(sys.modules, "tinker.types", types)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
    )
    output = tmp_path / "receipt.json"
    return SimpleNamespace(
        state=state,
        calls=calls,
        finished=finished,
        output=output,
        invoke=lambda *extra: evaluator.main(
            ["--base-model", "--per-suite", "1", "--out", str(output), *extra]
        ),
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reward_cannot_be_scored_receipt(harness, score):
    harness.state.score = score
    with pytest.raises(ValueError, match="finite"):
        harness.invoke()
    assert not harness.output.exists()
    assert harness.finished == [1]


@pytest.mark.parametrize("count", [0, 2])
def test_provider_must_return_exactly_requested_one_response(harness, count):
    harness.state.sequences = [SimpleNamespace(tokens=[7])] * count
    with pytest.raises(ValueError, match="exactly one"):
        harness.invoke()
    assert not harness.output.exists()
    assert harness.finished == [1]


@pytest.mark.parametrize(
    "extra", [("--max-tokens", "0"), ("--temperature", "nan"), ("--temperature", "-1")]
)
def test_invalid_sampling_options_fail_before_provider_work(harness, extra):
    with pytest.raises(SystemExit):
        harness.invoke(*extra)
    assert harness.calls == []
    assert harness.finished == []
    assert not harness.output.exists()


def test_over_budget_prompt_is_rejected_without_sampling(harness):
    harness.state.prompt_length = 2050
    with pytest.raises(ValueError, match="refusing task-removing truncation"):
        harness.invoke()
    assert harness.calls == []
    assert not harness.output.exists()
    assert harness.finished == [1]


def test_reward_callback_cannot_rewrite_response_length_evidence(harness):
    harness.state.mutate = True
    assert harness.invoke() == 0
    receipt = json.loads(harness.output.read_text())
    assert receipt["records"][0]["response_tokens"] == 2
    assert receipt["records"][1]["response_tokens"] == 0


def test_full_prompt_receipt_matches_nontruncated_boundary(harness):
    harness.state.prompt_length = 2048
    assert harness.invoke() == 0
    receipt = json.loads(harness.output.read_text())
    assert all(
        row["original_prompt_tokens"] == row["prompt_tokens"] == 2048 for row in receipt["records"]
    )
    assert all(row["prompt_truncated"] is False for row in receipt["records"])
    assert receipt["sampling"]["prompt_truncation"] == "reject_over_budget"
    for record, (sent, _) in zip(receipt["records"], harness.calls):
        assert record["prompt_token_ids_sha256"] == evaluator._sha256_text(
            json.dumps(sent, separators=(",", ":"))
        )

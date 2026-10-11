"""Native input validation and receipts using only local synthetic fixtures."""

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from platform_modal.openrlhf import inputs, trainer
from platform_modal.openrlhf.config import OpenRLHFConfig
from tests._shared_fakes import write_native_input_fixture


def dataset_rows(count=500):
    return [
        {"question": f"Synthetic question {index}?", "answer": "Reasoning\n#### 1,234.5"}
        for index in range(count)
    ]


def fake_dataset(rows):
    load = Mock(return_value=rows)
    return load, patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=load)})


def test_materialization_preserves_messages_and_normalizes_complete_labels(tmp_path):
    load, modules = fake_dataset(dataset_rows())
    path = tmp_path / "input.jsonl"
    with modules:
        trainer._prepare_gsm8k_500_jsonl(str(path))
    load.assert_called_once_with("openai/gsm8k", "main", split="train[:500]")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 500
    assert rows[0]["prompt"][0] == {"role": "system", "content": trainer.GSM8K_SYSTEM_PROMPT}
    assert rows[0]["prompt"][1]["role"] == "user"
    assert rows[0]["prompt"][1]["content"].startswith("Synthetic question 0?")
    assert rows[0]["label"] == "1234.5"
    assert "<|system|>" not in path.read_text()
    receipt = inputs.inspect_native_input(str(path))
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["rows"] == 500
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["source_revision"] is None
    inputs.ensure_input_unchanged(str(path), receipt)


def test_invalid_source_fails_before_writing_a_partial_artifact(tmp_path):
    bad_rows = [
        {"question": "q", "answer": "no final delimiter"},
        {"question": "q", "answer": "####"},
        {"question": "q", "answer": "#### NaN"},
        {"question": "q", "answer": "#### 1..2"},
        {"question": "q", "answer": "#### 12,34"},
        {"question": "q", "answer": "#### 42 ignored suffix"},
        {"question": "", "answer": "#### 42"},
    ]
    for index, bad in enumerate(bad_rows):
        rows = dataset_rows()
        rows[-1] = bad
        _, modules = fake_dataset(rows)
        path = tmp_path / f"bad-{index}.jsonl"
        with modules, pytest.raises(ValueError, match="GSM8K row"):
            trainer._prepare_gsm8k_500_jsonl(str(path))
        assert not path.exists()
    for count in (0, 499, 501):
        _, modules = fake_dataset(dataset_rows(count))
        path = tmp_path / f"count-{count}.jsonl"
        with modules, pytest.raises(ValueError, match="500"):
            trainer._prepare_gsm8k_500_jsonl(str(path))
        assert not path.exists()


def test_materialization_never_overwrites_an_existing_artifact(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("preserve")
    _, modules = fake_dataset(dataset_rows())
    with modules, pytest.raises(FileExistsError):
        trainer._prepare_gsm8k_500_jsonl(str(path))
    assert path.read_text() == "preserve"


def test_input_snapshot_rejects_bad_rows_labels_and_legacy_raw_delimiters(tmp_path):
    path = tmp_path / "input.jsonl"
    write_native_input_fixture(path)
    valid = json.loads(path.read_text().splitlines()[0])
    for change in (
        {"label": ""},
        {"label": "NaN"},
        {"label": 42},
        {"label": " 42 "},
        {"prompt": "<|system|>unrendered"},
        {"prompt": []},
        {"prompt": [{"role": "user", "content": "q"}]},
        {"prompt": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]},
    ):
        row = dict(valid, **change)
        path.write_text((json.dumps(row) + "\n") * 500)
        with pytest.raises(ValueError, match="native input row"):
            inputs.inspect_native_input(str(path))
    write_native_input_fixture(path, rows=499)
    with pytest.raises(ValueError, match="500 rows"):
        inputs.inspect_native_input(str(path))
    path.write_text("not-json\n" * 500)
    with pytest.raises(ValueError, match="invalid JSON"):
        inputs.inspect_native_input(str(path))
    duplicate = json.dumps(valid)[:-1] + ', "label": "99"}'
    path.write_text((duplicate + "\n") * 500)
    with pytest.raises(ValueError, match="invalid JSON"):
        inputs.inspect_native_input(str(path))
    path.write_bytes(b"\xff\n" * 500)
    with pytest.raises(ValueError, match="UTF-8"):
        inputs.inspect_native_input(str(path))


def native_config():
    return OpenRLHFConfig(
        algorithm={"algorithm": "grpo", "sample_num": 6},
        train_batch_size=8,
        remote_rm_url="http://127.0.0.1:8765/get_reward",
    )


def test_bad_prepared_input_never_launches_provider_or_writes_success(tmp_path):
    output = tmp_path / "run"

    def prepare(path):
        return write_native_input_fixture(path, label="")

    with (
        patch.object(trainer, "_validate_native_runtime", return_value="0.8.4"),
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl", side_effect=prepare),
        patch.object(trainer.subprocess, "run") as launch,
        pytest.raises(ValueError, match="invalid label"),
    ):
        trainer.run_openrlhf_training(native_config(), str(output))
    launch.assert_not_called()
    assert not (output / "result.json").exists()


def test_input_mutation_after_zero_exit_prevents_success_receipt(tmp_path):
    output = tmp_path / "run"

    def mutate(*_args, **_kwargs):
        (output / "gsm8k500.jsonl").write_text("changed during subprocess")

    with (
        patch.object(trainer, "_validate_native_runtime", return_value="0.8.4"),
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture),
        patch.object(trainer.subprocess, "run", side_effect=mutate),
        pytest.raises(RuntimeError, match="input changed"),
    ):
        trainer.run_openrlhf_training(native_config(), str(output))
    assert not (output / "result.json").exists()
    assert (output / "gsm8k500.jsonl").read_text() == "changed during subprocess"


def test_nonregular_input_is_rejected_and_open_snapshot_is_pinned(tmp_path):
    source = tmp_path / "input.jsonl"
    write_native_input_fixture(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    link = tmp_path / "symlink.jsonl"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="regular file"):
        inputs.inspect_native_input(str(link))
    replacement = tmp_path / "replacement.jsonl"
    write_native_input_fixture(replacement, label="99")
    real_open = os.open

    def swap(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == source:
            os.replace(replacement, source)
        return descriptor

    with patch.object(os, "open", side_effect=swap):
        receipt = inputs.inspect_native_input(str(source))
    assert receipt["sha256"] == digest
    with pytest.raises(RuntimeError, match="input changed"):
        inputs.ensure_input_unchanged(str(source), receipt)

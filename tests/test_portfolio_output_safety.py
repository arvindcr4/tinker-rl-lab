"""Offline guards against overwriting or failing to persist evaluation evidence."""

from types import SimpleNamespace

import pytest

from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator
from tests import test_portfolio_eval_integrity

harness = test_portfolio_eval_integrity.harness


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
def test_existing_output_is_rejected_before_provider_work(harness, kind):
    if kind == "file":
        harness.output.write_text("existing receipt")
    else:
        harness.output.symlink_to("missing-target.json")
    with pytest.raises(SystemExit):
        harness.invoke()
    assert harness.calls == []
    assert harness.finished == []
    if kind == "file":
        assert harness.output.read_text() == "existing receipt"
    else:
        assert harness.output.is_symlink()
        assert not harness.output.exists()


def test_competing_receipt_created_during_evaluation_is_not_overwritten(harness, monkeypatch):
    def score(*_):
        harness.output.write_text("competing receipt")
        return 1.0

    monkeypatch.setattr(evaluator, "PavlovNonXLAMReward", lambda: SimpleNamespace(score=score))
    with pytest.raises(FileExistsError):
        harness.invoke()
    assert harness.output.read_text() == "competing receipt"
    assert harness.finished == [1]
    assert not list(harness.output.parent.glob(".receipt.json.*.tmp"))


def test_bad_output_parent_fails_before_sampling(harness):
    parent = harness.output.parent / "not-a-directory"
    parent.write_text("existing file")
    with pytest.raises(FileExistsError):
        evaluator.main(["--base-model", "--per-suite", "1", "--out", str(parent / "receipt.json")])
    assert harness.calls == []
    assert harness.finished == []
    assert parent.read_text() == "existing file"


def test_unsupported_atomic_publication_fails_before_provider_work(harness, monkeypatch):
    def no_links(*_args, **_kwargs):
        raise OSError("link publication unavailable")

    monkeypatch.setattr("os.link", no_links)
    with pytest.raises(OSError, match="link publication unavailable"):
        harness.invoke()
    assert harness.calls == []
    assert harness.finished == []
    assert not harness.output.exists()
    assert not list(harness.output.parent.glob(".grpo-write-check-*"))

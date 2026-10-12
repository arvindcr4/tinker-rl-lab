"""Requested tokenizer revisions must not attest managed model weights."""

import json

import pytest

from tests import test_portfolio_eval_integrity

harness = test_portfolio_eval_integrity.harness


@pytest.mark.parametrize("trained", [False, True])
def test_receipt_labels_request_only_model_provenance(harness, trained):
    if trained:
        from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator

        evaluator.main(
            [
                "--sampler-path",
                "tinker://synthetic/checkpoint",
                "--hf-repo",
                "synthetic/repo",
                "--hf-revision",
                "main",
                "--hf-commit",
                "a" * 40,
                "--per-suite",
                "1",
                "--out",
                str(harness.output),
            ]
        )
    else:
        harness.invoke()
    receipt = json.loads(harness.output.read_text())
    source = receipt["model_source"]
    assert source == harness.state.tracking_config["model_source"]
    expected_request = (
        {"model_path": "tinker://synthetic/checkpoint"}
        if trained
        else {"base_model": receipt["model"]["model_id"]}
    )
    assert harness.state.sampler_requests == [expected_request]
    assert harness.state.tokenizer_requests == [
        (
            (source["tokenizer_request"]["model_id"],),
            {"revision": source["tokenizer_request"]["revision"]},
        )
    ]
    assert receipt["model"]["model_revision_scope"] == "tokenizer_request_only"
    assert source["tokenizer_request"]["revision"] == receipt["model"]["model_revision"]
    assert source["managed_base_weights"] == {"status": "unverified", "commit_sha": None}
    assert source["hf_checkpoint_binding"]["status"] == (
        "unverified" if trained else "not_applicable"
    )
    assert source["sampler_request"]["mode"] == (
        "checkpoint_path" if trained else "base_model_name"
    )


@pytest.mark.parametrize("path", ["", " "])
def test_empty_trained_selector_cannot_fall_back_to_base_model(harness, path):
    from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator

    with pytest.raises(SystemExit):
        evaluator.main(
            [
                "--sampler-path",
                path,
                "--hf-repo",
                "synthetic/repo",
                "--hf-revision",
                "main",
                "--hf-commit",
                "a" * 40,
                "--per-suite",
                "1",
                "--out",
                str(harness.output),
            ]
        )
    assert harness.calls == []
    assert harness.finished == []


def test_base_model_cannot_claim_unused_hf_binding(harness):
    with pytest.raises(SystemExit):
        harness.invoke(
            "--hf-repo", "synthetic/repo", "--hf-revision", "main", "--hf-commit", "a" * 40
        )
    assert harness.calls == []
    assert harness.finished == []

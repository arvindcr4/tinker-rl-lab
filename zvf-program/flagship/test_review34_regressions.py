"""Synthetic boundary tests; no native scoring parity or provider execution."""
import json
import subprocess
from unittest.mock import patch
import pytest
import frontier_swe_eval as frontier
import eval_pavlov_sdab as sdab
import mle_bench_eval as mle
import modal_e1_swe_bench_pro_full as e1


def test_native_timeout_is_structured(tmp_path):
    with patch.object(frontier, "_run", side_effect=subprocess.TimeoutExpired("fake", 1)):
        code, reward, _ = frontier._native_verify(task_dir=tmp_path, image="fake", submission_dir=tmp_path, candidate_mount="/x", output_dir=tmp_path / "out", cpus=1, memory_mb=64)
    assert code == 124 and reward is None


@pytest.mark.parametrize("value", [True, "1", -1, 1.5, None])
def test_usage_is_not_coerced(value):
    with pytest.raises(RuntimeError, match="usage counts"):
        sdab._result_row({"task_id": "test-task", "score": .5, "prompt_tokens": value, "sample_tokens": 0, "tool_calls": 0}, {})


def test_image_manifest_semantics_even_after_rehash():
    task = {"instance_id": "test", "dockerhub_tag": "tag"}
    image = {"instance_id": "test", "tag": "tag", "tag_uri": f"{e1.DOCKER_REPOSITORY}:tag", "digest": "sha256:" + "a" * 64, "immutable_uri": f"{e1.DOCKER_REPOSITORY}@sha256:" + "a" * 64}
    manifest = {"schema_version": "pavlov-e1-image-manifest-v1", "count": 1, "images": [image], "manifest_sha256": e1._sha256_text(e1._stable_json([image]))}
    e1._validate_image_manifest(manifest, [task])
    image["immutable_uri"] = "wrong@sha256:" + "a" * 64
    manifest["manifest_sha256"] = e1._sha256_text(e1._stable_json([image]))
    with pytest.raises(RuntimeError, match="binding mismatch"):
        e1._validate_image_manifest(manifest, [task])


@pytest.mark.parametrize("value", ["false", 0, 1, None, {}])
def test_evaluator_outcomes_require_boolean(tmp_path, value):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/eval_results.json").write_text(json.dumps({"i": value}))
    with pytest.raises(RuntimeError, match="boolean outcomes"):
        e1._aggregate_receipt(root=tmp_path, run_dir=tmp_path, rows=[], tasks=[], generations=[], candidates=[{"instance_id": "i"}], image_manifest={}, seed=1, max_tokens=1, temperature=0)


def test_rules_probe_preserves_unknown_and_rejects_truthy_text():
    assert mle.kaggle_rule_acceptance("synthetic", {})["accepted"] is None
    assert mle.kaggle_rule_acceptance("synthetic", {"accepted": False})["accepted"] is False
    with pytest.raises(ValueError, match="boolean"):
        mle.kaggle_rule_acceptance("synthetic", {"accepted": "false"})

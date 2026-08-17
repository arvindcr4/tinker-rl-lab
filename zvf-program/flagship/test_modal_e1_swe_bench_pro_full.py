from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import modal_e1_swe_bench_pro_full as runner
    from .e1_swe_bench_pro_full_eval import _load_image_map
    from .modal_e1_swe_bench_pro_full import (
        GENERATION_FIELDS,
        _build_prompt,
        _extract_diff,
        _generation_inputs,
        _path_candidates,
        _sanitize_task,
        _search_terms,
        _shared_tinker_sampler,
    )
except ImportError:
    import modal_e1_swe_bench_pro_full as runner
    from e1_swe_bench_pro_full_eval import _load_image_map
    from modal_e1_swe_bench_pro_full import (
        GENERATION_FIELDS,
        _build_prompt,
        _extract_diff,
        _generation_inputs,
        _path_candidates,
        _sanitize_task,
        _search_terms,
        _shared_tinker_sampler,
    )


VALID_PATCH = """diff --git a/a.py b/a.py
index 7898192..6178079 100644
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
"""


def _row() -> dict[str, object]:
    row: dict[str, object] = {key: f"value-{key}" for key in GENERATION_FIELDS}
    row.update(
        {
            "patch": "gold",
            "test_patch": "hidden tests",
            "fail_to_pass": "['secret']",
            "pass_to_pass": "['secret']",
            "selected_test_files_to_run": "['secret.py']",
            "before_repo_set_cmd": "secret command",
        }
    )
    return row


class E1FullRunnerTests(unittest.TestCase):
    def test_model_boundary_contains_only_allowlisted_fields(self) -> None:
        task = _sanitize_task(_row())
        self.assertEqual(set(task), set(GENERATION_FIELDS))
        prompt = _build_prompt(task, {"src/a.py": "old = True\n"})
        self.assertNotIn("hidden tests", prompt)
        self.assertNotIn("secret command", prompt)

    def test_extracts_valid_diff_before_trailing_prose(self) -> None:
        patch, reason = _extract_diff(VALID_PATCH + "I will explain the patch now.\n")
        self.assertEqual(patch, VALID_PATCH)
        self.assertEqual(reason, "valid unified diff")

    def test_rejects_placeholder_hunks(self) -> None:
        patch, reason = _extract_diff(VALID_PATCH.replace("@@ -1 +1 @@", "@@ ... @@"))
        self.assertEqual(patch, "")
        self.assertIn("concrete hunk", reason)

    def test_source_discovery_extracts_paths_and_identifiers(self) -> None:
        text = "Path: src/user/email.js. Call `getEmailForValidation` and db.mget(keys)."
        self.assertIn("src/user/email.js", _path_candidates(text))
        terms = _search_terms(text)
        self.assertIn("getEmailForValidation", terms)
        self.assertIn("mget", terms)

    def test_digest_manifest_loader_rejects_uri_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.json"
            path.write_text(
                '{"images":[{"instance_id":"i","digest":"sha256:abc",'
                '"immutable_uri":"repo@sha256:def"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "URI/digest mismatch"):
                _load_image_map(path)

    def test_tinker_sampler_is_reused_within_container(self) -> None:
        original_service = runner._TINKER_SERVICE
        original_sampler = runner._TINKER_SAMPLER

        class FakeService:
            def __init__(self, **_: object) -> None:
                fake_tinker.service_count += 1

            def create_sampling_client(self, **_: object) -> object:
                return object()

        class FakeTinker:
            ServiceClient = FakeService
            service_count = 0

        fake_tinker = FakeTinker()
        try:
            runner._TINKER_SERVICE = None
            runner._TINKER_SAMPLER = None
            first = _shared_tinker_sampler(fake_tinker, "run-one")
            second = _shared_tinker_sampler(fake_tinker, "run-two")
            self.assertIs(first, second)
            self.assertEqual(fake_tinker.service_count, 1)
        finally:
            runner._TINKER_SERVICE = original_service
            runner._TINKER_SAMPLER = original_sampler

    def test_pre_sampling_infra_receipt_is_archived_and_retried(self) -> None:
        task = _sanitize_task(_row())
        with tempfile.TemporaryDirectory() as directory:
            tasks_dir = Path(directory)
            task_dir = tasks_dir / str(task["instance_id"])
            task_dir.mkdir(parents=True)
            (task_dir / "source_context.json").write_text(
                json.dumps({"files": {"src/a.py": "old = True\n"}}),
                encoding="utf-8",
            )
            (task_dir / "generation.json").write_text(
                json.dumps(
                    {
                        "status": "INFRA_ERROR",
                        "phase": "pre_sampling",
                        "sample_started": False,
                        "wandb_run_id": "retry-me",
                    }
                ),
                encoding="utf-8",
            )
            (task_dir / "generation_response.txt").write_text("", encoding="utf-8")

            pending, projected_total = _generation_inputs(
                [task],
                tasks_dir,
                seed=1818,
                max_tokens=128,
                temperature=0.2,
            )

            self.assertEqual(len(pending), 1)
            self.assertGreater(projected_total, 0.0)
            self.assertEqual(
                pending[0]["pre_sampling_recovery"]["wandb_run_id"], "retry-me"
            )
            self.assertFalse((task_dir / "generation.json").exists())
            self.assertTrue(
                (
                    task_dir
                    / "generation_attempts/pre_sampling_retry-me.json"
                ).is_file()
            )

    def test_interrupted_gpu_attempt_preserves_backend_and_cost_metadata(self) -> None:
        task = _sanitize_task(_row())
        sources = {"src/a.py": "old = True\n"}
        recovery = {
            "index": 0,
            "instance_id": task["instance_id"],
            "wandb_run_id": "lost-gpu-run",
            "wandb_url": "https://wandb.example/runs/lost-gpu-run",
            "source_sha256": runner._sha256_text(runner._stable_json(sources)),
            "prompt_sha256": runner._sha256_text(_build_prompt(task, sources)),
            "projected_tinker_usd": 0.0,
            "original_wandb_status": None,
            "original_candidate_patch_sha256": None,
            "prompt_tokens": None,
            "response_tokens": None,
            "estimated_tinker_usd": 0.0,
            "estimated_modal_gpu_usd": 0.1,
            "modal_gpu_seconds": 10.0,
            "generation_backend": "modal_gpu_vllm_merged_peft",
            "gpu_type": "A100-80GB",
            "sample_started": True,
            "sample_completed": False,
            "failure_class": "parallel_handoff",
        }
        original = runner.INTERRUPTED_ATTEMPTS
        try:
            runner.INTERRUPTED_ATTEMPTS = (recovery,)
            with tempfile.TemporaryDirectory() as directory:
                task_dir = Path(directory) / str(task["instance_id"])
                task_dir.mkdir(parents=True)
                (task_dir / "source_context.json").write_text(
                    json.dumps({"files": sources}), encoding="utf-8"
                )

                runner._recover_interrupted_attempts([task], Path(directory))

                receipt = json.loads(
                    (task_dir / "generation.json").read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["status"], "GENERATION_ARTIFACT_LOST")
                self.assertEqual(receipt["generation_backend"], recovery["generation_backend"])
                self.assertEqual(receipt["estimated_modal_gpu_usd"], 0.1)
                self.assertFalse(receipt["additional_sampling_performed"])
        finally:
            runner.INTERRUPTED_ATTEMPTS = original


if __name__ == "__main__":
    unittest.main()

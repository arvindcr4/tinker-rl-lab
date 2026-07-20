import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "run_colab_e1_resume", HERE / "run_colab_e1_resume.py"
)
RESUME = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(RESUME)


class E1ExactResumeTests(unittest.TestCase):
    def test_launch_metadata_is_excluded_from_original_fingerprint_payload(self):
        source = {
            "schema_version": "colab-e1-confirmatory-unit-v1",
            "arm": "grpo",
            "seed": 71,
            "fingerprint": "fingerprint",
            "status": "launching",
            "session": "session",
            "hf_repo": "repo",
            "wandb_run_name": "run",
            "execution_plan": [],
            "launcher_retry_policy": {
                "exec_attempts": 3,
                "exec_retry_seconds": 60,
                "preserve_session": True,
            },
            "updated_at": "now",
        }
        self.assertEqual(
            RESUME.fingerprint_payload(source),
            {
                "schema_version": "colab-e1-confirmatory-unit-v1",
                "arm": "grpo",
                "seed": 71,
            },
        )

    def test_remote_result_is_reconstructed_from_manifest_provenance(self):
        manifest = {
            "evidence_class": "confirmatory",
            "audit_record": {"arm": "dapo", "seed": 23},
            "wandb": {
                "run_id": "c0d53921",
                "run_url": "https://wandb.ai/entity/project/runs/c0d53921",
            },
        }
        self.assertEqual(
            RESUME.remote_result_from_manifest(
                manifest, hf_repo="owner/repo", hf_commit="commit"
            ),
            {
                "evidence_class": "confirmatory",
                "audit_record": {"arm": "dapo", "seed": 23},
                "remote": {
                    "hf_repo": "owner/repo",
                    "hf_commit": "commit",
                    "wandb_run_id": "c0d53921",
                    "wandb_run_url": "https://wandb.ai/entity/project/runs/c0d53921",
                },
            },
        )

    def test_remote_result_requires_complete_wandb_provenance(self):
        with self.assertRaisesRegex(RuntimeError, "W&B run URL"):
            RESUME.remote_result_from_manifest(
                {
                    "evidence_class": "confirmatory",
                    "audit_record": {"arm": "dapo", "seed": 23},
                    "wandb": {"run_id": "c0d53921"},
                },
                hf_repo="owner/repo",
                hf_commit="commit",
            )


if __name__ == "__main__":
    unittest.main()

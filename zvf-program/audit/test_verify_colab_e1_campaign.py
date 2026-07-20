import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "verify_colab_e1_campaign", HERE / "verify_colab_e1_campaign.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class CampaignVerificationTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "arm": "grpo",
            "seed": 11,
            "evidence_class": "confirmatory",
            "heldout_n": 4,
            "heldout_score": 0.5,
            "last10_reward": 0.6,
            "mean_zvf": 0.7,
            "mean_gu": 0.3,
            "collapse": False,
            "rollouts": 480,
            "wall_clock_seconds": 10.0,
            "stack_fingerprint": "a" * 64,
            "fingerprint": "f" * 64,
            "treatment_changes": [],
            "remote": {"wandb_run_id": "run123"},
        }
        self.manifest = {
            "audit_record": {
                field: self.record[field] for field in VERIFY.AUDIT_FIELDS
            },
            "run_config": {
                "unit_fingerprint": "f" * 64,
                "stack_fingerprint": "a" * 64,
            },
            "remote_checkpoint_steps": [5, 10],
            "heldout_trace": [
                {
                    "index": index,
                    "correct": index % 2 == 0,
                    "completion_sha256": f"{index:064x}",
                }
                for index in range(4)
            ],
            "wandb": {"run_id": "run123"},
        }

    def test_required_files_cover_every_checkpoint_and_final_adapter(self):
        required = VERIFY.required_hf_files([5, 10])
        self.assertIn("run_manifest.json", required)
        self.assertIn("final/adapter_model.safetensors", required)
        self.assertIn("checkpoints/checkpoint-5/optimizer.pt", required)
        self.assertIn("checkpoints/checkpoint-10/trainer_state.json", required)

    def test_matching_manifest_pair_is_accepted(self):
        self.assertEqual(
            VERIFY.validate_manifest_pair(
                self.record, self.manifest, checkpoint_steps=[5, 10]
            ),
            [],
        )

    def test_manifest_pair_rejects_fingerprint_score_and_wandb_drift(self):
        broken = json.loads(json.dumps(self.manifest))
        broken["run_config"]["unit_fingerprint"] = "x" * 64
        broken["heldout_trace"][0]["correct"] = False
        broken["wandb"]["run_id"] = "other"
        errors = VERIFY.validate_manifest_pair(
            self.record, broken, checkpoint_steps=[5, 10]
        )
        self.assertTrue(any("unit fingerprint" in error for error in errors))
        self.assertTrue(any("held-out trace" in error for error in errors))
        self.assertTrue(any("W&B" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

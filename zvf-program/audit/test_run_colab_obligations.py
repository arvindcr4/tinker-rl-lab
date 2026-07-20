import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_colab_obligations.py")
SPEC = importlib.util.spec_from_file_location("run_colab_obligations", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ColabObligationRunnerTests(unittest.TestCase):
    def test_fingerprint_is_order_independent(self):
        self.assertEqual(
            MODULE.fingerprint({"arm": "grpo", "seed": 11}),
            MODULE.fingerprint({"seed": 11, "arm": "grpo"}),
        )

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "result.json"
            MODULE.atomic_json(path, {"status": "launching"})
            MODULE.atomic_json(path, {"status": "completed", "value": 1})
            self.assertEqual(
                json.loads(path.read_text()),
                {"status": "completed", "value": 1},
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_result_parser_accepts_colab_ansi_prefix(self):
        payload = {"units": [{"arm": "grpo", "seed": 11}]}
        lines = ["noise\n", "\x1b[32mE3_RESULT " + json.dumps(payload) + "\x1b[0m\n"]
        self.assertEqual(MODULE.result_from_log(lines), payload)

    def test_wandb_verifier_extracts_run_identity(self):
        response = io.BytesIO(
            json.dumps(
                {
                    "data": {
                        "project": {
                            "run": {"name": "run123", "state": "finished"}
                        }
                    }
                }
            ).encode()
        )
        with mock.patch.object(MODULE, "urlopen", return_value=response) as opened:
            record = MODULE.verify_wandb_run(
                "secret-key",
                "https://wandb.ai/entity-name/project-name/runs/run123",
            )
        self.assertEqual(record["run_id"], "run123")
        self.assertEqual(record["state"], "finished")
        request = opened.call_args.args[0]
        self.assertNotIn("secret-key", request.full_url)
        self.assertNotIn(b"secret-key", request.data)


if __name__ == "__main__":
    unittest.main()

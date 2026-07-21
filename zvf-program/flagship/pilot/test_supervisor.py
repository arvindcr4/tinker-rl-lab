from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilot.launcher import build_campaign_manifest
from pilot.protocol import load_protocol
from pilot.supervisor import initial_state, main, ready_jobs


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()
        self.manifest = build_campaign_manifest(self.protocol)
        self.state = initial_state(self.manifest)
        self.temporary = tempfile.TemporaryDirectory(dir=self.protocol.path.parent)
        payload = json.loads(self.protocol.path.read_text())
        payload["status"] = "locked_not_authorized"
        payload["authorization"]["gpu"] = False
        self.locked_path = Path(self.temporary.name) / "pilot_preregistration.json"
        self.locked_path.write_text(json.dumps(payload))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initial_scheduler_exposes_only_the_preflight(self) -> None:
        ready = ready_jobs(self.manifest, self.state, capacity=3)
        self.assertEqual(ready, ["preflight__a100_stack_smoke"])

    def test_three_corpus_jobs_unlock_after_preflight(self) -> None:
        self.state["jobs"]["preflight__a100_stack_smoke"]["status"] = "accepted"
        ready = ready_jobs(self.manifest, self.state, capacity=3)
        self.assertEqual(len(ready), 3)
        self.assertTrue(all(job_id.startswith("corpus__") for job_id in ready))

    def test_units_become_ready_only_after_their_corpus_is_accepted(self) -> None:
        corpus = "corpus__balanced_equal_length__s11"
        self.state["jobs"][corpus]["status"] = "accepted"
        ready = ready_jobs(self.manifest, self.state, capacity=10)
        matching = [job_id for job_id in ready if "balanced_equal_length__s11" in job_id]
        self.assertEqual(len(matching), 4)

    def test_run_refuses_before_subprocess_when_protocol_is_locked(self) -> None:
        with mock.patch("subprocess.Popen") as popen:
            with self.assertRaisesRegex(SystemExit, "allocation is forbidden"):
                main(["--protocol", str(self.locked_path), "--run"])
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("aggregate_audit", HERE / "aggregate_audit.py")
aggregate_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate_audit)


class StatisticsTests(unittest.TestCase):
    def test_verdicts_are_fail_closed(self):
        self.assertEqual(
            aggregate_audit.verdict((0.01, 0.03), (0.012, 0.028), 0.005, 0.01, 0.02),
            "RETAINS",
        )
        self.assertEqual(
            aggregate_audit.verdict((0.001, 0.02), (0.002, 0.018), 0.005, 0.01, None),
            "SURVIVES",
        )
        self.assertEqual(
            aggregate_audit.verdict((-0.009, 0.009), (-0.008, 0.008), 0.008, 0.01, None),
            "DISAPPEARS",
        )
        self.assertEqual(
            aggregate_audit.verdict((-0.04, -0.01), (-0.035, -0.015), 0.01, 0.01, None),
            "REVERSES",
        )
        self.assertEqual(
            aggregate_audit.verdict((-0.02, 0.02), (-0.015, 0.015), 0.02, 0.01, None),
            "INCONCLUSIVE",
        )

    def test_bootstrap_is_deterministic(self):
        first = aggregate_audit.paired_bootstrap_ci([0.01, 0.02, 0.03, 0.04], 0.95)
        second = aggregate_audit.paired_bootstrap_ci([0.01, 0.02, 0.03, 0.04], 0.95)
        self.assertEqual(first, second)

    def test_latex_results_are_derived_from_complete_aggregate(self):
        report = {
            "status": "COMPLETE",
            "n_seeds": 8,
            "results": {
                arm: {
                    "controlled_delta": 0.001,
                    "ci95": [-0.0045, 0.00675],
                    "achieved_mde_80": 0.008667,
                    "verdict": "DISAPPEARS" if arm == "dapo" else "INCONCLUSIVE",
                }
                for arm in aggregate_audit.LATEX_ARM_NAMES
            },
        }
        rendered = aggregate_audit.render_latex_results(report)
        self.assertIn(r"\newcommand{\AuditDAPODelta}{+0.00100}", rendered)
        self.assertIn(r"\newcommand{\AuditDAPOCILow}{-0.00450}", rendered)
        self.assertIn(r"\newcommand{\AuditDAPOMDE}{0.00867}", rendered)
        self.assertIn(r"\newcommand{\AuditDAPOVerdict}{DISAPPEARS}", rendered)

    def test_latex_results_reject_incomplete_aggregate(self):
        with self.assertRaisesRegex(aggregate_audit.AuditError, "COMPLETE"):
            aggregate_audit.render_latex_results({"status": "INCOMPLETE"})


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.prereg = aggregate_audit.load_json(HERE / "preregistration.json")

    def record(self, directory: Path, arm="grpo", seed=11):
        manifest = directory / f"{arm}-{seed}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "heldout_trace": [
                        {
                            "index": index,
                            "correct": index % 2 == 0,
                            "completion_sha256": f"{index:064x}",
                        }
                        for index in range(500)
                    ]
                }
            )
            + "\n"
        )
        return {
            "arm": arm,
            "seed": seed,
            "heldout_n": 500,
            "heldout_score": 0.5,
            "last10_reward": 0.4,
            "mean_zvf": 0.2,
            "mean_gu": 3.0,
            "collapse": False,
            "rollouts": 100,
            "wall_clock_seconds": 60.0,
            "stack_fingerprint": "sha256:shared-minus-treatment-fields",
            "treatment_changes": self.prereg["core_stratum"]["arms"][arm]["allowed_changes"],
            "manifest_path": manifest.name,
        }

    def test_missing_units_never_emit_verdicts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory)
            indexed, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "grpo-11.json", record)]
            )
            self.assertEqual(errors, [])
            missing = aggregate_audit.missing_units(self.prereg, indexed)
            self.assertEqual(len(missing), 39)

    def test_undeclared_treatment_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory, arm="dapo")
            record["treatment_changes"] = ["dynamic_sampling", "tokenizer"]
            _, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "dapo-11.json", record)]
            )
            self.assertTrue(any("treatment_changes" in error for error in errors))

    def test_unhashed_manifest_is_rejected_and_not_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory)
            manifest = directory / record["manifest_path"]
            manifest.write_text(
                json.dumps(
                    {
                        "heldout_trace": [
                            {"index": index, "correct": index % 2 == 0}
                            for index in range(500)
                        ]
                    }
                )
                + "\n"
            )
            indexed, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "grpo-11.json", record)]
            )
            self.assertEqual(indexed, {})
            self.assertTrue(any("completion hashes" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

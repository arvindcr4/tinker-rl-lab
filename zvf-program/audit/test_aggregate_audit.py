import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("aggregate_audit", HERE / "aggregate_audit.py")
aggregate_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate_audit)


class StatisticsTests(unittest.TestCase):
    def test_small_sample_mde_matches_exact_paired_t_power(self):
        differences = [-0.002, -0.012, -0.006, 0.002, 0.0, 0.016, 0.0, 0.010]
        mde = aggregate_audit.achieved_mde_80(differences)
        self.assertAlmostEqual(mde, 0.0101158932289, places=10)
        assert mde > 0.01

    def test_verdicts_are_fail_closed(self):
        assert (
            aggregate_audit.verdict(
                (0.01, 0.03), (0.012, 0.028), 0.005, 0.01, 0.02, difference_rejected=True
            )
            == "RETAINS"
        )
        assert (
            aggregate_audit.verdict(
                (0.001, 0.02), (0.002, 0.018), 0.005, 0.01, None, difference_rejected=True
            )
            == "SURVIVES"
        )
        assert (
            aggregate_audit.verdict(
                (-0.009, 0.009), (-0.008, 0.008), 0.008, 0.01, None, difference_rejected=False
            )
            == "DISAPPEARS"
        )
        assert (
            aggregate_audit.verdict(
                (-0.04, -0.01), (-0.035, -0.015), 0.01, 0.01, None, difference_rejected=True
            )
            == "REVERSES"
        )
        assert (
            aggregate_audit.verdict(
                (-0.02, 0.02), (-0.015, 0.015), 0.02, 0.01, None, difference_rejected=False
            )
            == "INCONCLUSIVE"
        )
        assert (
            aggregate_audit.verdict(
                (0.001, 0.02), (0.002, 0.018), 0.005, 0.01, None, difference_rejected=False
            )
            == "INCONCLUSIVE"
        )

    def test_bh_rejections_are_executable_not_documentary(self):
        rejections = aggregate_audit.benjamini_hochberg(
            {"dapo": 0.001, "gspo": 0.02, "drgrpo": 0.04, "aero": 0.5}
        )
        assert rejections == {"dapo": True, "gspo": True, "drgrpo": False, "aero": False}

    def test_bootstrap_is_deterministic(self):
        first = aggregate_audit.paired_bootstrap_ci([0.01, 0.02, 0.03, 0.04], 0.95)
        second = aggregate_audit.paired_bootstrap_ci([0.01, 0.02, 0.03, 0.04], 0.95)
        assert first == second

    def test_latex_results_are_derived_from_complete_aggregate(self):
        report = {
            "status": "COMPLETE",
            "n_seeds": 8,
            "results": {
                arm: {
                    "controlled_delta": 0.001,
                    "ci95": [-0.0045, 0.00675],
                    "achieved_mde_80": 0.0101158932289,
                    "verdict": "INCONCLUSIVE",
                }
                for arm in aggregate_audit.LATEX_ARM_NAMES
            },
        }
        rendered = aggregate_audit.render_latex_results(report)
        assert r"\newcommand{\AuditDAPODelta}{+0.00100}" in rendered
        assert r"\newcommand{\AuditDAPOCILow}{-0.00450}" in rendered
        assert r"\newcommand{\AuditDAPOMDE}{0.01012}" in rendered
        assert r"\newcommand{\AuditDAPOVerdict}{INCONCLUSIVE}" in rendered

    def test_exact_mde_above_margin_cannot_disappear(self):
        assert (
            aggregate_audit.verdict(
                (-0.0045, 0.00675),
                (-0.004, 0.006),
                0.0101158932289,
                0.01,
                None,
                difference_rejected=False,
            )
            == "INCONCLUSIVE"
        )

    def test_latex_results_reject_incomplete_aggregate(self):
        with pytest.raises(aggregate_audit.AuditError, match="COMPLETE"):
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
        record = {
            "arm": arm,
            "seed": seed,
            "heldout_n": 500,
            "heldout_score": 0.5,
            "last10_reward": 0.4,
            "mean_zvf": 0.2,
            "mean_gu": 0.8,
            "collapse": False,
            "rollouts": 100,
            "wall_clock_seconds": 60.0,
            "stack_fingerprint": "b" * 64,
            "treatment_changes": self.prereg["core_stratum"]["arms"][arm]["allowed_changes"],
            "manifest_path": manifest.name,
        }
        record.update(
            evidence_class="confirmatory",
            fingerprint="a" * 64,
            remote_verification={
                "hf_repo": "owner/repo",
                "hf_commit": "c" * 40,
                "wandb": {"state": "finished"},
            },
        )
        data = json.loads(manifest.read_text())
        data.update(
            evidence_class="confirmatory",
            audit_record=dict(record),
            run_config={
                **aggregate_audit.execution_contract(self.prereg["core_stratum"]),
                "arm": arm,
                "seed": seed,
                "treatment_changes": record["treatment_changes"],
                "unit_fingerprint": record["fingerprint"],
                "stack_fingerprint": record["stack_fingerprint"],
            },
        )
        manifest.write_text(json.dumps(data))
        return record

    def test_missing_units_never_emit_verdicts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory)
            indexed, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "grpo-11.json", record)]
            )
            assert errors == []
            missing = aggregate_audit.missing_units(self.prereg, indexed)
            assert len(missing) == 39

    def test_undeclared_treatment_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory, arm="dapo")
            record["treatment_changes"] = ["dynamic_sampling", "tokenizer"]
            _, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "dapo-11.json", record)]
            )
            assert any("treatment_changes" in error for error in errors)

    def test_unhashed_manifest_is_rejected_and_not_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            record = self.record(directory)
            manifest = directory / record["manifest_path"]
            manifest.write_text(
                json.dumps(
                    {
                        "heldout_trace": [
                            {"index": index, "correct": index % 2 == 0} for index in range(500)
                        ]
                    }
                )
                + "\n"
            )
            indexed, errors = aggregate_audit.validate_records(
                self.prereg, [(directory / "grpo-11.json", record)]
            )
            assert indexed == {}
            assert any("completion hashes" in error for error in errors)


if __name__ == "__main__":
    unittest.main()

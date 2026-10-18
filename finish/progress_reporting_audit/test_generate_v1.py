import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import generate_v1 as g
import pytest

ROOT = Path(__file__).resolve().parents[2]


def audit(rows):
    return {
        "full_suite_denominator": 300,
        "outcomes": rows,
        "terminal_attempts": len(rows),
        "native_resolved_count": sum(r["resolved"] for r in rows),
    }


def outcome(task, status="NATIVE_REPORT_VERIFIED", resolved=False):
    return {"instance_id": task, "status": status, "resolved": resolved, "run_id": "synthetic"}


def episode(task, reward):
    return {
        "task_id": task,
        "end_time": "synthetic",
        "termination_reason": "user_stop",
        "reward_info": {"reward": reward},
    }


class MetricsTests(unittest.TestCase):
    def test_coverage_not_accuracy(self):
        m = g.measure("public", 1967, 1967, 450, complete=True)
        assert m["coverage"]["fraction"] == 1
        self.assertAlmostEqual(m["accuracy_or_success"]["fraction"], 450 / 1967)
        assert not m["full_original_suite_claim"]

    def test_unknown_is_not_zero(self):
        m = g.measure("coverage only", 40, 75)
        assert m["accuracy_or_success"] is None

    def test_zero_score_is_present(self):
        assert g.measure("zero", 2, 97, 0)["accuracy_or_success"]["fraction"] == 0

    def test_invalid_counts(self):
        for n, d in ((True, 2), (-1, 2), (3, 2), (1, 0), (1, 1.0), (float("nan"), 2)):
            with self.subTest(n=n, d=d), pytest.raises(g.EvidenceError):
                g.ratio(n, d)

    def test_partial_cannot_be_complete(self):
        with pytest.raises(g.EvidenceError):
            g.measure("partial", 13, 255, 13, complete=True)

    def test_swe_errors_separate(self):
        # Summary-only evidence is no longer sufficient for native coverage.
        with pytest.raises(g.EvidenceError):
            g.aggregate_swe(
                [
                    audit(
                        [
                            outcome("a", resolved=True),
                            outcome("b", "NATIVE_PATCH_APPLY_FAILED"),
                            outcome("c", "NATIVE_EMPTY_PATCH"),
                        ]
                    )
                ]
            )

    def test_swe_duplicate_replay_rejected(self):
        a = audit([outcome("a")])
        with pytest.raises(g.EvidenceError):
            g.aggregate_swe([a, a])

    def test_swe_unknown_not_native_zero(self):
        with pytest.raises(g.EvidenceError):
            g.aggregate_swe([audit([outcome("a", "UNKNOWN")])])

    def test_swe_denominator_drift(self):
        a = audit([])
        a["full_suite_denominator"] = 731
        with pytest.raises(g.EvidenceError):
            g.aggregate_swe([a])

    def test_tau_separate_runs(self):
        a = g.tau_run("run03", [episode("a", 1), episode("b", 0)])
        b = g.tau_run("run04", [episode("a", 0)])
        assert a["accuracy_or_success"]["fraction"] == 0.5
        assert b["accuracy_or_success"]["fraction"] == 0
        assert a["coverage"]["denominator"] == 97

    def test_tau_duplicate_within_run(self):
        with pytest.raises(g.EvidenceError):
            g.tau_run("run", [episode("a", 1), episode("a", 0)])

    def test_tau_wrong_task(self):
        with pytest.raises(g.EvidenceError):
            g.tau_run("run", [episode("a", 1)], ["b"])

    def test_tau_unfinished_and_nonfinite(self):
        for e in (
            episode("a", float("nan")),
            episode("a", True),
            dict(episode("a", 1), end_time=None),
        ):
            with pytest.raises(g.EvidenceError):
                g.tau_run("run", [e])

    def test_omni_exclusions_not_accuracy_denominator(self):
        m = g.measure("parser accepted", 4426, 4428, 2271)
        assert m["coverage"]["denominator"] == 4428
        assert m["accuracy_or_success"]["denominator"] == 4426
        assert not m["complete_named_scope"]

    def test_e7_primary_not_inventory(self):
        assert g.EXPECTED_ORIGINAL[6] == 28
        assert g.EXPECTED_PUBLIC[6] == 1507

    def test_private_denominators_unknown(self):
        for index in (5, 9, 11, 12, 13):
            assert g.EXPECTED_ORIGINAL[index] is None

    def test_complete_report_contract(self):
        rows = [
            {
                "lane": f"E{i + 1}",
                "original": {"suite_id": g.ORIGINAL[i][1], "denominator": g.EXPECTED_ORIGINAL[i]},
                "portfolio": {
                    "suite_id": g.PUBLIC_SUITES[i],
                    "declared_denominator": g.EXPECTED_PUBLIC[i],
                    "measurements": [],
                },
            }
            for i in range(14)
        ]
        rows[7]["portfolio"]["measurements"] = [g.measure("public", 1967, 1967, 450, complete=True)]
        report = {
            "rows": rows,
            "total_lanes": 14,
            "complete_named_portfolio_scopes": ["E8"],
            "all_original_suites_complete": False,
            "replacement_score_lanes": ["E8"],
            "replacement_score_lane_count": 1,
        }
        g.validate_report(report)
        altered = copy.deepcopy(report)
        altered["rows"][6]["original"]["denominator"] = 46
        with pytest.raises(g.EvidenceError):
            g.validate_report(altered)
        altered = copy.deepcopy(report)
        altered["rows"][7]["portfolio"]["measurements"][0]["full_original_suite_claim"] = True
        with pytest.raises(g.EvidenceError):
            g.validate_report(altered)

    def test_14_lane_duplicates_and_wrong_count(self):
        with pytest.raises(g.EvidenceError):
            g.validate_report({"rows": [{"lane": "E1"}] * 14})
        with pytest.raises(g.EvidenceError):
            g.validate_report({"rows": []})


class FileTests(unittest.TestCase):
    def setUp(self):
        if shutil.disk_usage(ROOT).free < g.FLOOR:
            self.skipTest("Disk floor: no writes")
        # Synthetic tests model adequate disk; the explicit floor test overrides it.
        disk_patch = patch.object(g.shutil, "disk_usage", return_value=shutil.disk_usage(ROOT))
        disk_patch.start()
        self.addCleanup(disk_patch.stop)
        self.temp = tempfile.TemporaryDirectory(prefix="review11-report-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_hash_tamper(self):
        (self.root / "x.json").write_text("{}")
        r = g.Reader(self.root)
        r.load("x.json")
        (self.root / "x.json").write_text('{"changed":true}')
        with pytest.raises(g.EvidenceError):
            r.stable()

    def test_new_receipt_race(self):
        r = g.Reader(self.root)
        r.glob("*.json")
        (self.root / "x.json").write_text("{}")
        with pytest.raises(g.EvidenceError):
            r.stable()

    def test_expected_hash(self):
        (self.root / "x.json").write_text("{}")
        with pytest.raises(g.EvidenceError):
            g.Reader(self.root).load("x.json", "0" * 64)

    def test_escape(self):
        with pytest.raises(g.EvidenceError):
            g.Reader(self.root).resolve("../outside")

    def test_no_parent_output_mutation(self):
        parent = self.root / "outputs/PES_Phase2_Review_2026-09-12/finish"
        with pytest.raises(g.EvidenceError):
            g.publish(self.root, parent, {})
        assert not parent.exists()

    def test_no_overwrite(self):
        p = self.root / "finish/progress_reporting_audit/old"
        p.mkdir(parents=True)
        (p / "report.json").write_text("PRESERVE")
        with pytest.raises(g.EvidenceError):
            g.publish(self.root, p, {})
        assert (p / "report.json").read_text() == "PRESERVE"

    def test_disk_floor_read_and_publish(self):
        with patch.object(g.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(10, 9, 1)):
            with pytest.raises(g.EvidenceError):
                g.Reader(self.root).guard()
            with pytest.raises(g.EvidenceError):
                g.publish(self.root, self.root / "finish/progress_reporting_audit/new", {})


if __name__ == "__main__":
    unittest.main()

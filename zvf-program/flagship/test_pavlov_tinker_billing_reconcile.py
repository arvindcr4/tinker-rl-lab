"""Regression tests for the offline Pavlov/Tinker billing reconciler."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from flagship.pavlov_tinker_billing_reconcile import (
    HARD_CAP,
    OPERATIONAL_CAP,
    SAFETY_RESERVE,
    BillingReconciliationError,
    reconcile_billing_export,
)


ROOT = Path(__file__).resolve().parents[2]
RECONCILER = ROOT / "zvf-program" / "flagship" / "pavlov_tinker_billing_reconcile.py"


def _d(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _receipt(
    run_id: str,
    *,
    status: str = "completed",
    live: Any = "0",
    pending: Any = "0",
    base_estimated: Any = "0",
    base_unreconciled: Any | None = None,
    rejected: Any = "0",
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_id": run_id, "status": status}
    if _d(live):
        result["live_billed_usd"] = str(live)
    if _d(pending):
        result["pending_billing_usd"] = str(pending)
    if _d(base_estimated):
        result["base_eval_estimated_usd"] = str(base_estimated)
    if base_unreconciled is not None:
        result["base_eval_unreconciled_usd"] = str(base_unreconciled)
    if _d(rejected):
        result["rejected_unreconciled_usd"] = str(rejected)
    result.update(extra)
    return result


def _ledger(*receipts: Mapping[str, Any], proposed: Any = "0", **extra: Any) -> dict[str, Any]:
    live = sum((_d(item.get("live_billed_usd", "0")) for item in receipts), Decimal("0"))
    pending = sum(
        (_d(item.get("pending_billing_usd", "0")) for item in receipts), Decimal("0")
    )
    base_estimated = sum(
        (_d(item.get("base_eval_estimated_usd", "0")) for item in receipts), Decimal("0")
    )
    base_unreconciled = sum(
        (
            _d(
                item.get(
                    "base_eval_unreconciled_usd",
                    item.get("base_eval_estimated_usd", "0")
                    if not _d(item.get("live_billed_usd", "0"))
                    else "0",
                )
            )
            for item in receipts
        ),
        Decimal("0"),
    )
    rejected = sum(
        (_d(item.get("rejected_unreconciled_usd", "0")) for item in receipts), Decimal("0")
    )
    proposal = _d(proposed)
    conservative = pending + base_unreconciled + rejected
    result: dict[str, Any] = {
        "maximum_usd": str(HARD_CAP),
        "operational_cap_usd": str(OPERATIONAL_CAP),
        "safety_reserve_usd": str(SAFETY_RESERVE),
        "receipts": [dict(item) for item in receipts],
        "live_billed_usd": str(live),
        "pending_billing_usd": str(pending),
        "base_eval_estimated_usd": str(base_estimated),
        "base_eval_unreconciled_usd": str(base_unreconciled),
        "rejected_unreconciled_usd": str(rejected),
        "conservative_pending_usd": str(conservative),
        "proposed_arm_usd": str(proposal),
        "projected_usage_usd": str(live + conservative + proposal),
    }
    result.update(extra)
    return result


def _billing_row(
    run_id: str,
    amount: Any,
    *,
    status: str = "completed",
    component: str | None = None,
    row_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_id": run_id, "status": status, "cost_usd": str(amount)}
    if component is not None:
        result["component"] = component
    if row_id is not None:
        result["billing_row_id"] = row_id
    result.update(extra)
    return result


def _assert_no_float(value: Any) -> None:
    if isinstance(value, float):
        raise AssertionError(f"float leaked into Decimal report: {value!r}")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_float(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_float(item)


class BillingReconciliationTests(unittest.TestCase):
    def test_live_billing_matches_once_and_reports_exact_remaining(self) -> None:
        ledger = _ledger(_receipt("r1", live="2.00"))
        report = reconcile_billing_export(
            ledger, {"rows": [_billing_row("r1", "2.00", component="train")]}
        )

        self.assertEqual(report["live_billed_usd"], Decimal("2.00"))
        self.assertEqual(report["projected_usage_usd"], Decimal("2.00"))
        self.assertEqual(report["remaining_authorized_usd"], Decimal("14.50"))
        self.assertTrue(report["another_paid_action_may_start"])
        self.assertEqual(report["components_usd"]["reconciled"]["train"], Decimal("2.00"))

    def test_component_lines_aggregate_without_double_counting(self) -> None:
        ledger = _ledger(_receipt("r1", live="1.50"))
        export = {
            "rows": [
                _billing_row("r1", "1.00", component="train", row_id="line-a"),
                _billing_row("r1", "0.50", component="sample", row_id="line-b"),
            ]
        }

        report = reconcile_billing_export(ledger, export)

        self.assertEqual(report["projected_usage_usd"], Decimal("1.50"))
        self.assertEqual(report["billing_export_raw_total_usd"], Decimal("1.50"))
        self.assertEqual(
            report["components_usd"]["reconciled"],
            {"train": Decimal("1.00"), "sample": Decimal("0.50")},
        )

    def test_pending_billing_is_not_added_twice(self) -> None:
        ledger = _ledger(_receipt("r1", status="pending", pending="3.00"))
        report = reconcile_billing_export(
            ledger,
            {"rows": [_billing_row("r1", "3.00", status="pending", component="train")]},
        )

        self.assertEqual(report["pending_billing_usd"], Decimal("3.00"))
        self.assertEqual(report["live_billed_usd"], Decimal("0"))
        self.assertEqual(report["projected_usage_usd"], Decimal("3.00"))

    def test_pending_export_cannot_replace_already_live_local_billing(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "lagging"):
            reconcile_billing_export(
                _ledger(_receipt("r1", live="2.00")),
                {
                    "rows": [
                        _billing_row("r1", "2.00", status="pending", component="train")
                    ]
                },
            )

    def test_live_billing_supersedes_pending_for_same_run(self) -> None:
        ledger = _ledger(_receipt("r1", status="pending", pending="3.00"))
        export = {
            "rows": [
                _billing_row(
                    "r1", "3.00", status="pending", component="pending", row_id="pending-line"
                ),
                _billing_row("r1", "2.00", status="completed", component="train", row_id="live-line"),
            ]
        }

        report = reconcile_billing_export(ledger, export)

        self.assertEqual(report["live_billed_usd"], Decimal("2.00"))
        self.assertEqual(report["pending_billing_usd"], Decimal("0"))
        self.assertEqual(report["projected_usage_usd"], Decimal("2.00"))
        self.assertEqual(report["cleared_estimate_usd"], Decimal("3.00"))
        self.assertEqual(report["superseded_pending_usd"], Decimal("3.00"))

    def test_base_eval_actual_replaces_estimate_but_keeps_gross_audit_value(self) -> None:
        ledger = _ledger(
            _receipt(
                "eval-1",
                base_estimated="2.50",
                base_unreconciled="2.50",
            )
        )
        export = {
            "rows": [_billing_row("eval-1", "2.00", component="base_eval")]
        }

        report = reconcile_billing_export(ledger, export)

        self.assertEqual(report["base_eval_estimated_usd"], Decimal("2.50"))
        self.assertEqual(report["base_eval_unreconciled_usd"], Decimal("0"))
        self.assertEqual(report["live_billed_usd"], Decimal("2.00"))
        self.assertEqual(report["projected_usage_usd"], Decimal("2.00"))

    def test_base_eval_component_row_ignores_separate_gross_estimate(self) -> None:
        ledger = _ledger(
            _receipt("eval-1", base_estimated="2.50", base_unreconciled="2.50")
        )
        export = {
            "rows": [
                {
                    "run_id": "eval-1",
                    "status": "completed",
                    "cost_usd": "2.00",
                    "estimated_cost_usd": "2.50",
                    "component_costs": {"base_eval": "2.00"},
                }
            ]
        }
        report = reconcile_billing_export(ledger, export)
        self.assertEqual(report["projected_usage_usd"], Decimal("2.00"))

    def test_statusless_base_eval_estimate_remains_conservative_pending(self) -> None:
        receipt = {"run_id": "eval-1", "base_eval_estimated_usd": "2.50"}
        report = reconcile_billing_export(_ledger(receipt), {"rows": []})
        self.assertEqual(report["base_eval_estimated_usd"], Decimal("2.50"))
        self.assertEqual(report["base_eval_unreconciled_usd"], Decimal("2.50"))
        self.assertEqual(report["pending_billing_usd"], Decimal("0"))

    def test_rejected_allowance_is_preserved_without_billing_row(self) -> None:
        report = reconcile_billing_export(
            _ledger(_receipt("failed-1", status="rejected", rejected="4.00")),
            {"rows": []},
        )

        self.assertEqual(report["rejected_unreconciled_usd"], Decimal("4.00"))
        self.assertEqual(report["projected_usage_usd"], Decimal("4.00"))
        self.assertEqual(report["rejected_run_ids"], ["failed-1"])

    def test_rejected_export_cannot_replace_conflicting_local_live_charge(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "conflicts with local live"):
            reconcile_billing_export(
                _ledger(_receipt("r1", live="2.00")),
                {"rows": [_billing_row("r1", "4.00", status="rejected")]},
            )

    def test_missing_run_id_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "missing a run ID"):
            reconcile_billing_export(_ledger(_receipt("r1", live="1")), {"rows": [{"cost_usd": "1"}]})

    def test_malformed_ledger_root_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "ledger root must be an object"):
            reconcile_billing_export([], {"rows": []})

    def test_conflicting_export_row_containers_fail_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "conflicting row containers"):
            reconcile_billing_export(
                _ledger(_receipt("r1", live="1")),
                {"rows": [_billing_row("r1", "1")], "charges": []},
            )

    def test_ambiguous_run_id_aliases_fail_closed(self) -> None:
        row = _billing_row("r1", "1")
        row["receipt_id"] = "r2"
        with self.assertRaisesRegex(BillingReconciliationError, "ambiguous run IDs"):
            reconcile_billing_export(_ledger(_receipt("r1", live="1")), {"rows": [row]})

    def test_duplicate_local_receipts_fail_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "duplicate ledger run ID"):
            reconcile_billing_export(
                _ledger(_receipt("r1", live="1"), _receipt("r1", live="1")), {"rows": []}
            )

    def test_duplicate_billing_row_id_fail_closed(self) -> None:
        rows = [
            _billing_row("r1", "1", row_id="same"),
            _billing_row("r1", "1", component="sample", row_id="same"),
        ]
        with self.assertRaisesRegex(BillingReconciliationError, "duplicate billing row ID"):
            reconcile_billing_export(_ledger(_receipt("r1", live="2")), {"rows": rows})

    def test_duplicate_scalar_component_without_row_ids_is_ambiguous(self) -> None:
        rows = [_billing_row("r1", "1"), _billing_row("r1", "1")]
        with self.assertRaisesRegex(BillingReconciliationError, "ambiguous duplicate"):
            reconcile_billing_export(_ledger(_receipt("r1", live="2")), {"rows": rows})

    def test_unknown_billing_run_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "unknown run IDs"):
            reconcile_billing_export(_ledger(_receipt("r1", live="1")), {"rows": [_billing_row("r2", "1")]})

    def test_unknown_price_model_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "unknown price model"):
            reconcile_billing_export(
                _ledger(_receipt("r1", live="1")),
                {"rows": [_billing_row("r1", "1", model="unregistered-model")]},
            )

    def test_explicit_local_price_model_can_be_reconciled(self) -> None:
        report = reconcile_billing_export(
            _ledger(_receipt("r1", live="1"), model="local-test-model"),
            {"rows": [_billing_row("r1", "1", model="local-test-model")]},
        )
        self.assertEqual(report["projected_usage_usd"], Decimal("1"))

    def test_missing_terminal_export_is_treated_as_lagging(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "lagging"):
            reconcile_billing_export(_ledger(_receipt("r1", live="2")), {"rows": []})

    def test_explicit_lagging_export_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "explicitly lagging"):
            reconcile_billing_export(
                _ledger(_receipt("r1", status="pending", pending="2")),
                {"billing_lagging": True, "rows": []},
            )

    def test_nonfinal_export_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(BillingReconciliationError, "explicitly lagging"):
            reconcile_billing_export(
                _ledger(_receipt("r1", status="pending", pending="2")),
                {"billing_complete": False, "rows": []},
            )

    def test_export_as_of_must_cover_completed_run(self) -> None:
        ledger = _ledger(_receipt("r1", live="1", completed_at="2026-08-09T01:00:00Z"))
        export = {
            "as_of": "2026-08-09T00:59:59Z",
            "rows": [_billing_row("r1", "1")],
        }
        with self.assertRaisesRegex(BillingReconciliationError, "predates completed runs"):
            reconcile_billing_export(ledger, export)

    def test_billing_row_after_export_snapshot_fails_closed(self) -> None:
        export = {
            "as_of": "2026-08-09T00:00:00Z",
            "rows": [
                _billing_row(
                    "r1", "1", billed_at="2026-08-09T00:00:01Z"
                )
            ],
        }
        with self.assertRaisesRegex(BillingReconciliationError, "after billing export as_of"):
            reconcile_billing_export(_ledger(_receipt("r1", live="1")), export)

    def test_operational_cap_boundary_preserves_reserve_but_blocks_next_action(self) -> None:
        report = reconcile_billing_export(
            _ledger(_receipt("r1", live=OPERATIONAL_CAP)),
            {"rows": [_billing_row("r1", OPERATIONAL_CAP)]},
        )

        self.assertEqual(report["projected_usage_usd"], OPERATIONAL_CAP)
        self.assertEqual(report["operational_remaining_usd"], Decimal("0"))
        self.assertEqual(report["remaining_authorized_usd"], Decimal("0"))
        self.assertTrue(report["reserve_preserved"])
        self.assertFalse(report["another_paid_action_may_start"])
        self.assertEqual(report["status"], "REJECT")

    def test_hard_cap_boundary_is_rejected_and_reserve_is_not_preserved(self) -> None:
        report = reconcile_billing_export(
            _ledger(_receipt("r1", live=HARD_CAP)),
            {"rows": [_billing_row("r1", HARD_CAP)]},
        )

        self.assertEqual(report["hard_remaining_usd"], Decimal("0"))
        self.assertFalse(report["reserve_preserved"])
        self.assertFalse(report["another_paid_action_may_start"])
        self.assertEqual(report["status"], "REJECT")
        self.assertTrue(any("operational cap" in reason for reason in report["reasons"]))

    def test_cap_contract_mismatch_fails_closed(self) -> None:
        ledger = _ledger(_receipt("r1", live="1"), operational_cap_usd="16.49")
        with self.assertRaisesRegex(BillingReconciliationError, "operational_cap_usd"):
            reconcile_billing_export(ledger, {"rows": [_billing_row("r1", "1")]})

    def test_ledger_aggregate_mismatch_fails_closed(self) -> None:
        ledger = _ledger(_receipt("r1", live="1"), live_billed_usd="1.01")
        with self.assertRaisesRegex(BillingReconciliationError, "does not reconcile"):
            reconcile_billing_export(ledger, {"rows": [_billing_row("r1", "1")]})

    def test_component_total_mismatch_fails_closed(self) -> None:
        row = {
            "run_id": "r1",
            "status": "completed",
            "cost_usd": "2.10",
            "component_costs": {"train": "1.00", "sample": "1.00"},
        }
        with self.assertRaisesRegex(BillingReconciliationError, "does not reconcile"):
            reconcile_billing_export(_ledger(_receipt("r1", live="2.10")), {"rows": [row]})

    def test_local_component_allocation_cannot_silently_drop_usage(self) -> None:
        receipt = _receipt("r1", status="pending", pending="2.00", component_costs={"train": "1.00"})
        with self.assertRaisesRegex(BillingReconciliationError, "reconciled component total"):
            reconcile_billing_export(_ledger(receipt), {"rows": []})

    def test_report_keeps_decimal_arithmetic_until_display(self) -> None:
        report = reconcile_billing_export(
            _ledger(_receipt("r1", live="0.10"), proposed="0.20"),
            {"rows": [_billing_row("r1", "0.10")]},
        )
        _assert_no_float(report)
        self.assertEqual(report["projected_usage_usd"], Decimal("0.30"))

    def test_cli_emits_json_and_allows_under_cap_action(self) -> None:
        ledger = _ledger(_receipt("r1", live="2"))
        export = {"rows": [_billing_row("r1", "2", component="train")]}
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            ledger_path = directory_path / "ledger.json"
            export_path = directory_path / "billing.json"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            export_path.write_text(json.dumps(export), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(RECONCILER), str(ledger_path), str(export_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(ROOT / "zvf-program")},
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertEqual(output["remaining_authorized_usd"], 14.5)
        self.assertTrue(output["another_paid_action_may_start"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from flagship.pavlov_tinker_budget import (
    BudgetError,
    build_ledger,
    estimate_cost,
    estimate_token_cost,
    guard_launch,
    load_budget,
    load_portfolio_contract,
    load_receipt,
)


HERE = Path(__file__).resolve().parent
BUDGET_PATH = HERE / "pavlov_tinker_budget.json"
PORTFOLIO_PATH = HERE / "pavlovs_domain_contract.json"


class PavlovTinkerBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = load_budget(BUDGET_PATH)
        self.portfolio = load_portfolio_contract(PORTFOLIO_PATH)

    def test_contract_keeps_hard_cap_operational_cap_and_reserve(self) -> None:
        self.assertEqual(Decimal(str(self.budget["maximum_usd"])), Decimal("18.0"))
        self.assertEqual(Decimal(str(self.budget["operational_cap_usd"])), Decimal("16.5"))
        self.assertEqual(Decimal(str(self.budget["safety_reserve_usd"])), Decimal("1.5"))

    def test_conservative_cost_uses_uncached_rate_for_cached_input(self) -> None:
        cost = estimate_token_cost(
            self.budget,
            prefill_tokens=1_000_000,
            cached_prefill_tokens=1_000_000,
            sample_tokens=1_000_000,
        )
        self.assertEqual(cost, Decimal("2.415"))

    def test_steps_need_tokens_and_are_priced_deterministically(self) -> None:
        with self.assertRaises(BudgetError):
            estimate_cost(self.budget, steps=200)
        self.assertEqual(
            estimate_cost(self.budget, steps=200, train_tokens_per_step=1000),
            Decimal("0.2354"),
        )

    def test_base_eval_estimate_is_not_live_billing(self) -> None:
        receipt = {
            "schema_version": "pavlov-xlam-eval-v1",
            "tokenizer_model": self.budget["model"],
            "estimated_cost_usd": 2.25,
            "prompt_tokens": 100,
            "sample_tokens": 100,
            "suite_id": "swe_bench_pro_eval",
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["known_usd"], Decimal("0"))
        self.assertEqual(ledger["base_eval_estimated_usd"], Decimal("2.25"))
        self.assertEqual(ledger["conservative_pending_usd"], Decimal("2.25"))
        self.assertEqual(ledger["receipt_pending_usd"], Decimal("2.25"))
        self.assertEqual(ledger["projected_usd"], Decimal("2.25"))

    def test_live_billing_is_known_and_not_double_counted_as_estimate(self) -> None:
        receipt = {
            "status": "completed",
            "model": self.budget["model"],
            "billed_cost_usd": 3.125,
            "estimated_cost_usd": 3.5,
            "suite_id": "swe_bench_pro_eval",
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["known_usd"], Decimal("3.125"))
        self.assertEqual(ledger["base_eval_estimated_usd"], Decimal("0"))
        self.assertEqual(ledger["conservative_pending_usd"], Decimal("0"))
        self.assertEqual(ledger["projected_usd"], Decimal("3.125"))

    def test_reconciled_base_eval_keeps_estimate_distinct_without_double_charge(self) -> None:
        receipt = {
            "schema_version": "pavlov-xlam-eval-v1",
            "status": "completed",
            "model": self.budget["model"],
            "billed_cost_usd": Decimal("3.125"),
            "base_eval_estimated_usd": Decimal("2.25"),
            "suite_id": "E1",
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["known_usd"], Decimal("3.125"))
        self.assertEqual(ledger["base_eval_estimated_usd"], Decimal("2.25"))
        self.assertEqual(ledger["base_eval_unreconciled_usd"], Decimal("0"))
        self.assertEqual(ledger["conservative_pending_usd"], Decimal("0"))
        self.assertEqual(ledger["projected_usd"], Decimal("3.125"))

    def test_rejected_run_allowance_is_distinct_and_charged_conservatively(self) -> None:
        receipt = {
            "status": "rejected",
            "model": self.budget["model"],
            "maximum_authorized_cost_usd": 4.0,
            "suite_id": "openreward_train",
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["known_usd"], Decimal("0"))
        self.assertEqual(ledger["rejected_run_allowance_usd"], Decimal("4.0"))
        self.assertEqual(ledger["conservative_pending_usd"], Decimal("4.0"))

    def test_unknown_model_fails_closed(self) -> None:
        receipt = {
            "status": "completed",
            "model": "unknown/model",
            "billed_cost_usd": 1.0,
        }
        with self.assertRaises(BudgetError):
            build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)

    def test_missing_receipt_fails_closed(self) -> None:
        with self.assertRaises(BudgetError):
            load_receipt(HERE / "does-not-exist.json", self.budget, portfolio=self.portfolio)

    def test_portfolio_has_complete_training_and_primary_suite_counts(self) -> None:
        assert self.portfolio is not None
        self.assertEqual(len(self.portfolio["training_suite_ids"]), 12)
        self.assertEqual(len(self.portfolio["primary_eval_suite_ids"]), 14)
        self.assertEqual(len(self.portfolio["domains"]), 16)

    def test_suite_and_domain_projection_is_exposed_and_requires_coverage(self) -> None:
        suite_costs = {
            suite_id: Decimal("0") for suite_id in self.portfolio["required_suite_ids"]
        }
        suite_costs["openreward_train"] = Decimal("1")
        domain_costs = {
            domain: Decimal("0") for domain in self.portfolio["domains"]
        }
        domain_costs[self.portfolio["domains"][0]] = Decimal("1")
        proposal = {
            "suite_costs": suite_costs,
            "domain_costs": domain_costs,
            "phase_reservations": {"phase-1": Decimal("1")},
            "enforce_domain_coverage": True,
        }
        ledger = build_ledger(self.budget, projection=proposal, portfolio_contract=self.portfolio)
        self.assertEqual(ledger["proposed_arm_usd"], Decimal("2.0"))
        self.assertEqual(ledger["portfolio_reservation_usd"], Decimal("1.0"))
        decision = guard_launch(ledger)
        self.assertFalse(decision["allowed"])
        self.assertTrue(any("crowd out required domain coverage" in reason for reason in decision["reasons"]))

    def test_operational_cap_equality_is_allowed_and_hard_cap_equality_keeps_reserve_guard(self) -> None:
        at_operational = build_ledger(
            self.budget,
            projection={"cost_usd": Decimal("16.50")},
            portfolio_contract=self.portfolio,
        )
        self.assertEqual(at_operational["projected_usd"], Decimal("16.50"))
        self.assertEqual(at_operational["remaining_usd"], Decimal("0.00"))
        self.assertTrue(guard_launch(at_operational)["allowed"])
        self.assertTrue(at_operational["reserve_preserved"])

        at_hard = build_ledger(
            self.budget,
            projection={"cost_usd": Decimal("18.00")},
            portfolio_contract=self.portfolio,
        )
        self.assertEqual(at_hard["projected_usd"], Decimal("18.00"))
        self.assertEqual(at_hard["hard_remaining_usd"], Decimal("0.00"))
        self.assertFalse(guard_launch(at_hard)["allowed"])
        self.assertFalse(at_hard["reserve_preserved"])

    def test_complete_portfolio_projection_rejects_missing_suite_or_domain_allocations(self) -> None:
        with self.assertRaisesRegex(BudgetError, "missing domain allocations"):
            build_ledger(
                self.budget,
                projection={"suite_costs": {"T1": Decimal("1")}},
                portfolio_contract=self.portfolio,
            )
        with self.assertRaisesRegex(BudgetError, "missing suite allocations"):
            build_ledger(
                self.budget,
                projection={"domain_costs": {self.portfolio["domains"][0]: Decimal("1")}},
                portfolio_contract=self.portfolio,
            )
        with self.assertRaisesRegex(BudgetError, "missing domain allocations"):
            build_ledger(
                self.budget,
                projection={"suite_reservations": {"T1": Decimal("1")}},
                portfolio_contract=self.portfolio,
            )

    def test_complete_portfolio_allocations_reconcile_without_float_conversion(self) -> None:
        suite_costs = {
            suite_id: Decimal("0") for suite_id in self.portfolio["required_suite_ids"]
        }
        domain_costs = {
            domain: Decimal("0") for domain in self.portfolio["domains"]
        }
        ledger = build_ledger(
            self.budget,
            projection={"suite_costs": suite_costs, "domain_costs": domain_costs},
            portfolio_contract=self.portfolio,
        )
        self.assertTrue(ledger["allocation_reconciled"])
        self.assertEqual(
            ledger["reconciliation"]["suite_total_usd"],
            ledger["reconciliation"]["domain_total_usd"],
        )
        self.assertEqual(
            ledger["reconciliation"]["domain_total_usd"],
            ledger["projected_usage_usd"],
        )
        self.assertEqual(
            set(ledger["suite_allocations"]["projected"]),
            set(self.portfolio["required_suite_ids"]),
        )
        self.assertEqual(
            set(ledger["domain_allocations"]["projected"]),
            set(self.portfolio["domains"]),
        )

    def test_duplicate_receipts_fail_closed(self) -> None:
        receipt = {
            "receipt_id": "run-1",
            "status": "completed",
            "model": self.budget["model"],
            "billed_cost_usd": Decimal("1"),
            "suite_id": "T1",
        }
        with self.assertRaisesRegex(BudgetError, "duplicate receipt"):
            build_ledger(
                self.budget,
                [receipt, dict(receipt)],
                portfolio_contract=self.portfolio,
            )

    def test_pending_billing_is_separate_from_live_billing(self) -> None:
        receipt = {
            "receipt_id": "pending-1",
            "status": "pending",
            "model": self.budget["model"],
            "pending_cost_usd": Decimal("2"),
            "suite_id": "T1",
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["live_billed_usd"], Decimal("0"))
        self.assertEqual(ledger["pending_billing_usd"], Decimal("2"))
        self.assertEqual(ledger["receipt_pending_usd"], Decimal("2"))
        self.assertEqual(ledger["conservative_pending_usd"], Decimal("2"))
        self.assertEqual(ledger["projected_usage_usd"], Decimal("2"))

    def test_primary_eval_heldout_proof_tracks_proven_and_pending_subsets(self) -> None:
        receipt = {
            "receipt_id": "eval-1",
            "status": "completed",
            "model": self.budget["model"],
            "billed_cost_usd": Decimal("1"),
            "suite_id": "E1",
            "heldout_receipts": {
                "split_receipt": "split-digest",
                "license_receipt": "license-digest",
                "task_receipt": "task-digest",
                "container_receipt": "container-digest",
                "decontamination_receipt": "decontamination-digest",
            },
        }
        ledger = build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)
        self.assertEqual(ledger["heldout_receipt_proven_count"], 1)
        self.assertEqual(ledger["heldout_pending_count"], 13)
        self.assertEqual(ledger["heldout_receipt_proven_suite_ids"], [self.portfolio["primary_eval_suite_ids"][0]])
        self.assertNotIn("all 14 held-out", repr(ledger).lower())

    def test_heldout_claim_without_immutable_receipts_rejects_launch(self) -> None:
        ledger = build_ledger(
            self.budget,
            projection={"claim_heldout": True, "heldout_suite_ids": ["E1"]},
            portfolio_contract=self.portfolio,
        )
        decision = guard_launch(ledger)
        self.assertFalse(decision["allowed"])
        self.assertTrue(any("immutable split/license/task/container" in reason for reason in decision["reasons"]))

    def test_heldout_claim_does_not_treat_plain_split_labels_as_receipts(self) -> None:
        receipt = {
            "status": "completed",
            "model": self.budget["model"],
            "billed_cost_usd": Decimal("1"),
            "suite_id": "E1",
            "heldout_claim_requested": True,
            "heldout_receipts": {
                "split": "test",
                "license": "licensed",
                "task": "task-1",
                "container": "image:latest",
                "decontamination": "clean",
            },
        }
        with self.assertRaisesRegex(BudgetError, "missing immutable receipts"):
            build_ledger(self.budget, [receipt], portfolio_contract=self.portfolio)

    def test_proposed_arm_crossing_operational_cap_is_rejected(self) -> None:
        proposal = {"cost_usd": 16.51}
        ledger = build_ledger(self.budget, projection=proposal, portfolio_contract=self.portfolio)
        decision = guard_launch(ledger)
        self.assertFalse(decision["allowed"])
        self.assertLess(ledger["hard_remaining_usd"], Decimal("2"))
        self.assertTrue(any("operational cap" in reason for reason in decision["reasons"]))

    def test_cli_emits_json_and_nonzero_for_cap_rejection(self) -> None:
        command = [
            sys.executable,
            "-m",
            "flagship.pavlov_tinker_budget",
            "--budget",
            str(BUDGET_PATH),
            "--projection",
            self._write_json({"cost_usd": 16.51}),
        ]
        result = subprocess.run(
            command,
            cwd=HERE.parent.parent,
            env={**os.environ, "PYTHONPATH": str(HERE.parent)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["decision"], "REJECT")
        self.assertIn("projected_usd", report)

    def _write_json(self, value: object) -> str:
        temp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with temp:
            json.dump(value, temp)
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return temp.name


if __name__ == "__main__":
    unittest.main()

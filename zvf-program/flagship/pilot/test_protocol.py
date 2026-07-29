from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pilot.plan_screening import build_manifest, write_manifest
from pilot.protocol import (
    CONDITION_ORDER,
    FORBIDDEN_E1_ID_FRAGMENTS,
    REGIME_ORDER,
    PilotUnit,
    ProtocolError,
    build_screening_plan,
    execution_blockers,
    load_protocol,
)


class PilotProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_matrix_is_exact_ordered_and_unique(self) -> None:
        units = list(self.protocol.screening_units())
        self.assertEqual(len(units), 24)
        self.assertEqual(len({unit.unit_id for unit in units}), 24)
        self.assertEqual({unit.condition for unit in units}, set(CONDITION_ORDER))
        self.assertEqual({unit.regime for unit in units}, set(REGIME_ORDER))
        for regime in REGIME_ORDER:
            for seed in (11, 23, 37):
                observed = [
                    unit.condition for unit in units if unit.regime == regime and unit.seed == seed
                ]
                self.assertEqual(observed, list(CONDITION_ORDER))

    def test_current_protocol_has_passed_the_explicit_authorization_transition(self) -> None:
        self.protocol.require_gpu_authorization()
        self.assertEqual(execution_blockers(self.protocol), ())

    def test_every_plan_is_dry_run_only_and_has_no_allocation_command(self) -> None:
        manifest = build_manifest()
        self.assertEqual(manifest["unit_count"], 24)
        self.assertFalse(manifest["allocation_allowed"])
        for plan in manifest["units"]:
            self.assertEqual(plan["status"], "dry_run_only")
            self.assertFalse(plan["allocation"]["allowed"])
            self.assertIsNone(plan["allocation"]["command"])
            self.assertEqual(plan["protocol"]["sha256"], manifest["protocol_sha256"])
            self.assertTrue(plan["readiness"]["ready"])
            self.assertEqual(plan["readiness"]["authorization_blockers"], [])

    def test_readiness_has_no_remaining_blockers(self) -> None:
        blockers = execution_blockers(self.protocol)
        self.assertEqual(blockers, ())

    def test_pilot_identities_do_not_overlap_frozen_e1(self) -> None:
        for plan in build_manifest()["units"]:
            for value in plan["identity"].values():
                lowered = value.lower()
                for forbidden in FORBIDDEN_E1_ID_FRAGMENTS:
                    self.assertNotIn(forbidden, lowered)

    def test_remote_identities_are_protocol_scoped_and_corpus_is_shared(self) -> None:
        plans = build_manifest()["units"]
        protocol_suffix = self.protocol.sha256[:8]
        self.assertTrue(all(protocol_suffix in plan["identity"]["hf_repo"] for plan in plans))
        self.assertTrue(
            all(
                plan["protocol"]["source_bundle_sha256"][:8] in plan["identity"]["hf_repo"]
                for plan in plans
            )
        )
        block = [
            plan
            for plan in plans
            if plan["unit"]["regime"] == "balanced_equal_length" and plan["unit"]["seed"] == 11
        ]
        self.assertEqual(len(block), 4)
        self.assertEqual(len({plan["identity"]["corpus_hf_repo"] for plan in block}), 1)
        self.assertEqual(len({plan["identity"]["hf_repo"] for plan in block}), 4)

    def test_planner_sources_are_immutably_bound(self) -> None:
        plan = build_manifest()["units"][0]
        self.assertIn("zvf-program/flagship/pilot/protocol.py", plan["source_bindings"])
        self.assertIn("zvf-program/flagship/pilot/plan_screening.py", plan["source_bindings"])
        self.assertIn(
            "zvf-program/flagship/pilot/provenance/r3-corpus-bindings.json",
            plan["source_bindings"],
        )

    def test_corpus_and_unit_sources_are_separately_bound(self) -> None:
        plan = build_manifest()["units"][0]
        binding = plan["corpus_binding"]
        self.assertEqual(binding["status"], "accepted_complete")
        self.assertEqual(binding["completed_groups"], 100)
        self.assertNotEqual(
            binding["source_bindings_sha256"],
            plan["protocol"]["source_bundle_sha256"],
        )
        seed_23 = self.protocol.corpus_binding("balanced_equal_length", 23)
        self.assertEqual(seed_23["status"], "verified_prefix")
        self.assertEqual(seed_23["completed_groups"], 20)

    def test_unit_plan_rejects_out_of_matrix_values(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "unknown pilot condition"):
            build_screening_plan(self.protocol, PilotUnit("unknown", "balanced_equal_length", 11))
        with self.assertRaisesRegex(ProtocolError, "outside the screening matrix"):
            build_screening_plan(
                self.protocol, PilotUnit("intended_full", "balanced_equal_length", 53)
            )

    def test_atomic_writer_emits_manifest_and_all_unit_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            unit_files = sorted((root / "units").glob("*.json"))
            self.assertEqual(len(unit_files), 24)
            self.assertEqual(manifest["unit_count"], 24)
            for path in unit_files:
                plan = json.loads(path.read_text())
                self.assertEqual(path.stem, plan["unit"]["id"])
                self.assertEqual(
                    plan["fingerprint"],
                    next(
                        unit["fingerprint"]
                        for unit in manifest["units"]
                        if unit["unit"]["id"] == path.stem
                    ),
                )


if __name__ == "__main__":
    unittest.main()

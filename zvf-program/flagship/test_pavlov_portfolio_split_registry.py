from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_portfolio_split_registry import (
    EXPECTED_PRIMARY_EVAL_SUITE_IDS,
    EXPECTED_TRAIN_SUITE_IDS,
    PortfolioSplitRegistryError,
    aggregate_task_hashes,
    build_registry,
    load_contract,
    main,
    verify_registry,
)


REVISION = "a" * 40
RECEIPT_NAMES = ("revision", "license", "container", "decontamination", "verifier", "split_manifest")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def contract_copy() -> dict[str, object]:
    return copy.deepcopy(load_contract())


def ready_contract() -> dict[str, object]:
    contract = contract_copy()
    contract["status"] = "ready"
    return contract


def suite_manifest(suite_id: str, role: str, domains: list[str], index: int) -> dict[str, object]:
    hashes = [digest(f"{suite_id}:task:{index}:0"), digest(f"{suite_id}:task:{index}:1")]
    return {
        "suite_id": suite_id,
        "suite_role": role,
        "domain_tags": list(domains),
        "revision": format(index + 1, "040x"),
        "receipt_refs": {
            name: f"receipt://{name}/{suite_id}" for name in RECEIPT_NAMES
        },
        "task_hashes": list(hashes),
        "aggregate_sha256": aggregate_task_hashes(hashes),
        "counts": {role: len(hashes)},
    }


def portfolio_manifests() -> list[dict[str, object]]:
    contract = load_contract()
    registry = contract["suite_registry"]
    assert isinstance(registry, dict)
    manifests: list[dict[str, object]] = []
    index = 0
    for role, ids in (("train", EXPECTED_TRAIN_SUITE_IDS), ("primary_eval", EXPECTED_PRIMARY_EVAL_SUITE_IDS)):
        for suite_id in ids:
            domains = registry[suite_id]["domains"]
            manifests.append(suite_manifest(suite_id, role, list(domains), index))
            index += 1
    return manifests


def xlam_preflight() -> dict[str, object]:
    train_hashes = [digest("xlam:train")]
    test_hashes = [digest("xlam:test")]
    return {
        "component": "xLAM",
        "suite_id": "pavlov_xlam",
        "dataset_id": "Salesforce/xlam-function-calling-60k",
        "seed": 809,
        "revision": "b" * 40,
        "receipt_refs": {
            name: f"receipt://xlam/{name}" for name in RECEIPT_NAMES
        },
        "task_hashes": {"train": train_hashes, "test": test_hashes},
        "aggregate_hashes": {
            "train": aggregate_task_hashes(train_hashes),
            "test": aggregate_task_hashes(test_hashes),
        },
    }


class PavlovPortfolioSplitRegistryTests(unittest.TestCase):
    def test_contract_has_exact_12_train_and_14_primary_eval_ids(self) -> None:
        self.assertEqual(len(EXPECTED_TRAIN_SUITE_IDS), 12)
        self.assertEqual(len(EXPECTED_PRIMARY_EVAL_SUITE_IDS), 14)
        self.assertEqual(
            set(EXPECTED_TRAIN_SUITE_IDS)
            | set(EXPECTED_PRIMARY_EVAL_SUITE_IDS),
            {
                "openreward_train",
                "swe_gym_train",
                "browsergym_train",
                "bfcl_train",
                "scienceworld_train",
                "unix_ctf_train",
                "agentdojo_train",
                "rtlcoder_train",
                "crafter_train",
                "visual_app_train",
                "api_bank_rlvr_train",
                "openr1_math_train",
                "swe_bench_pro_eval",
                "frontier_swe_eval",
                "sdab_eval",
                "banker_toolbench_eval",
                "apex_agents_eval",
                "webbench_eval",
                "binaryaudit_eval",
                "lifescibench_eval",
                "mle_bench_eval",
                "agentharm_eval",
                "verilog_eval",
                "appbench_eval",
                "openreward_games_eval",
                "frontiermath_eval",
            },
        )

    def test_valid_registry_is_deterministic_and_xlam_is_separate(self) -> None:
        manifests = portfolio_manifests()
        contract = ready_contract()
        first = build_registry(manifests, contract=contract, xlam_preflight=xlam_preflight())
        second = build_registry(copy.deepcopy(manifests), contract=contract, xlam_preflight=xlam_preflight())
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "READY")
        self.assertEqual(first["suite_counts"], {"train": 12, "primary_eval": 14})
        self.assertEqual(first["cross_suite_overlap"]["overlap_count"], 0)
        self.assertEqual(first["held_out_proven_suite_ids"], [])
        self.assertEqual(first["xlam_component_preflight"]["status"], "READY")
        self.assertNotIn("pavlov_xlam", first["suite_ids"]["train"] + first["suite_ids"]["primary_eval"])
        rendered = json.dumps(first, sort_keys=True)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"target"', rendered)
        self.assertNotIn("train zero", rendered)

    def test_missing_extra_and_duplicate_suite_ids_are_rejected(self) -> None:
        manifests = portfolio_manifests()
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "missing suite IDs"):
            build_registry(manifests[1:])
        extra = manifests + [copy.deepcopy(manifests[0])]
        extra[-1]["suite_id"] = "not_in_contract"
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "extra suite IDs"):
            build_registry(extra)
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "duplicate suite ID"):
            build_registry(manifests + [copy.deepcopy(manifests[0])])

    def test_wrong_role_is_rejected(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["suite_role"] = "primary_eval"
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "disagrees with contract role"):
            build_registry(manifests)

    def test_mutable_revision_and_placeholder_receipts_are_rejected(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["revision"] = "main"
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "immutable 40-character"):
            build_registry(manifests)
        for receipt in ("license", "container", "decontamination", "verifier"):
            candidate = portfolio_manifests()
            candidate[0]["receipt_refs"][receipt] = "TODO"
            with self.assertRaisesRegex(PortfolioSplitRegistryError, f"placeholder {receipt} receipt"):
                build_registry(candidate)

    def test_cross_role_and_cross_suite_hash_overlap_are_hard_failures(self) -> None:
        manifests = portfolio_manifests()
        source_hash = manifests[0]["task_hashes"][0]
        assert isinstance(source_hash, str)
        manifests[-1]["task_hashes"][0] = source_hash
        hashes = manifests[-1]["task_hashes"]
        manifests[-1]["aggregate_sha256"] = aggregate_task_hashes(hashes)
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "cross_role task-hash overlap"):
            build_registry(manifests)

        manifests = portfolio_manifests()
        manifests[1]["task_hashes"][0] = manifests[0]["task_hashes"][0]
        manifests[1]["aggregate_sha256"] = aggregate_task_hashes(manifests[1]["task_hashes"])
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "cross_suite task-hash overlap"):
            build_registry(manifests)

    def test_domain_unions_and_company_required_domains_are_checked(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["domain_tags"] = []
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "train domain union missing"):
            build_registry(manifests)

        custom_contract = contract_copy()
        assert isinstance(custom_contract["companies"], list)
        custom_contract["companies"] = [{"name": "RequiredCo", "required_domains": ["alignment"]}]
        candidate = portfolio_manifests()
        for manifest in candidate:
            if manifest["suite_role"] == "train":
                manifest["domain_tags"] = ["tool_use"]
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "company RequiredCo train required-domain gap"):
            build_registry(candidate, contract=custom_contract)

    def test_primary_eval_is_not_held_out_without_a_receipt(self) -> None:
        manifests = portfolio_manifests()
        held_out_id = EXPECTED_PRIMARY_EVAL_SUITE_IDS[-1]
        held_out_manifest = next(item for item in manifests if item["suite_id"] == held_out_id)
        held_out_manifest["held_out"] = True
        registry = build_registry(manifests)
        self.assertEqual(registry["status"], "BLOCKED")  # xLAM preflight is absent
        self.assertEqual(registry["held_out_proven_suite_ids"], [])
        held_out_manifest["receipt_refs"]["held_out"] = f"receipt://held-out/{held_out_id}"
        registry = build_registry(manifests)
        self.assertEqual(registry["held_out_proven_suite_ids"], [held_out_id])
        self.assertEqual(
            next(item for item in registry["suites"] if item["suite_id"] == held_out_id)["role"],
            "primary_eval",
        )

    def test_held_out_role_is_canonicalized_only_with_proof(self) -> None:
        manifests = portfolio_manifests()
        held_out_id = EXPECTED_PRIMARY_EVAL_SUITE_IDS[0]
        held_out_manifest = next(item for item in manifests if item["suite_id"] == held_out_id)
        held_out_manifest["split_role"] = "held_out"
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "held_out role requires"):
            build_registry(manifests)
        held_out_manifest["receipt_refs"]["held_out"] = f"receipt://held-out/{held_out_id}"
        registry = build_registry(manifests)
        entry = next(item for item in registry["suites"] if item["suite_id"] == held_out_id)
        self.assertEqual(entry["role"], "primary_eval")
        self.assertIn(held_out_id, registry["held_out_proven_suite_ids"])

    def test_xlam_missing_or_invalid_receipts_blocks_but_does_not_join_26(self) -> None:
        registry = build_registry(portfolio_manifests())
        self.assertEqual(registry["status"], "BLOCKED")
        self.assertIn("xLAM component preflight missing", registry["blockers"])
        invalid = xlam_preflight()
        invalid["receipt_refs"]["verifier"] = "placeholder"
        registry = build_registry(portfolio_manifests(), xlam_preflight=invalid)
        self.assertEqual(registry["status"], "BLOCKED")
        self.assertTrue(any("placeholder verifier receipt" in item for item in registry["blockers"]))
        self.assertEqual(len(registry["suites"]), 26)

        overlapping = xlam_preflight()
        overlapping["task_hashes"]["train"][0] = portfolio_manifests()[0]["task_hashes"][0]
        overlapping["aggregate_hashes"]["train"] = aggregate_task_hashes(overlapping["task_hashes"]["train"])
        registry = build_registry(
            portfolio_manifests(),
            contract=ready_contract(),
            xlam_preflight=overlapping,
        )
        self.assertEqual(registry["status"], "BLOCKED")
        self.assertIn("xLAM cross-suite task-hash overlap with portfolio", registry["blockers"])

    def test_draft_contract_cannot_be_promoted_by_budget_authorization(self) -> None:
        contract = contract_copy()
        contract["status"] = "draft-awaiting-budget-cap"
        registry = build_registry(
            portfolio_manifests(), contract=contract, xlam_preflight=xlam_preflight()
        )
        self.assertEqual(registry["status"], "BLOCKED")
        self.assertIn("contract status is not finalized: draft-awaiting-budget-cap", registry["blockers"])
        self.assertIn("budget authorization cannot override a non-final contract status", registry["blockers"])
        self.assertFalse(registry["launch_authorized"])

    def test_raw_prompt_or_target_fields_are_rejected(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["prompt"] = "secret prompt"
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "metadata-only"):
            build_registry(manifests)

    def test_verify_registry_detects_revision_drift(self) -> None:
        contract = ready_contract()
        registry = build_registry(portfolio_manifests(), contract=contract, xlam_preflight=xlam_preflight())
        self.assertTrue(verify_registry(registry, contract=contract))
        expected = {EXPECTED_TRAIN_SUITE_IDS[0]: registry["suites"][0]["revision"]}
        # suites are sorted by role, so look up the exact suite revision for a
        # readable expected-revision assertion.
        expected[EXPECTED_TRAIN_SUITE_IDS[0]] = next(
            item["revision"] for item in registry["suites"] if item["suite_id"] == EXPECTED_TRAIN_SUITE_IDS[0]
        )
        changed = copy.deepcopy(registry)
        changed["suites"][0]["revision"] = "f" * 40
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "registry metadata drift"):
            verify_registry(changed, contract=contract)
        with self.assertRaisesRegex(PortfolioSplitRegistryError, "revision drift"):
            verify_registry(
                registry,
                contract=contract,
                expected_revisions={EXPECTED_TRAIN_SUITE_IDS[0]: "f" * 40},
            )

    def test_local_cli_generate_and_verify(self) -> None:
        manifests = portfolio_manifests()
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            manifest_dir = directory_path / "manifests"
            manifest_dir.mkdir()
            for index, manifest in enumerate(manifests):
                (manifest_dir / f"{index:02d}.json").write_text(json.dumps(manifest), encoding="utf-8")
            xlam_path = directory_path / "xlam.json"
            xlam_path.write_text(json.dumps(xlam_preflight()), encoding="utf-8")
            registry_path = directory_path / "registry.json"
            self.assertEqual(
                main(
                    [
                        "generate",
                        "--manifest-dir",
                        str(manifest_dir),
                        "--xlam-preflight",
                        str(xlam_path),
                        "--out",
                        str(registry_path),
                    ]
                ),
                0,
            )
            self.assertEqual(main(["verify", "--registry", str(registry_path)]), 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_portfolio_split_registry import build_registry
from flagship.pavlov_runnable_suite_resolver import (
    EXPECTED_PRIMARY_EVAL_SUITE_IDS,
    EXPECTED_TRAIN_SUITE_IDS,
    RunnableSuiteResolverError,
    main,
    resolve_runnable_suites,
    verify_resolution,
)


DOMAINS = (
    "alignment",
    "browser",
    "chip_design",
    "code",
    "computer_use",
    "design",
    "enterprise",
    "finance",
    "games",
    "long_horizon",
    "math",
    "ml",
    "multi_domain",
    "science",
    "security",
    "tool_use",
)
REVISIONS = {suite_id: format(index + 1, "040x") for index, suite_id in enumerate(
    list(EXPECTED_TRAIN_SUITE_IDS) + list(EXPECTED_PRIMARY_EVAL_SUITE_IDS)
)}
RECEIPT_KEYS = (
    "revision",
    "license",
    "container",
    "decontamination",
    "verifier",
    "split_manifest",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ready_contract() -> dict[str, object]:
    suite_registry: dict[str, dict[str, object]] = {}
    all_ids = list(EXPECTED_TRAIN_SUITE_IDS) + list(EXPECTED_PRIMARY_EVAL_SUITE_IDS)
    for index, suite_id in enumerate(all_ids):
        role = "train" if suite_id in EXPECTED_TRAIN_SUITE_IDS else "primary_eval"
        suite_registry[suite_id] = {
            "role": role,
            "domains": list(DOMAINS),
        }
    return {
        "schema_version": "test-contract-v1",
        "status": "ready",
        "domains": list(DOMAINS),
        "companies": [{"name": "Acme", "required_domains": ["alignment", "browser"]}],
        "suite_registry": suite_registry,
    }


def suite_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    all_ids = list(EXPECTED_TRAIN_SUITE_IDS) + list(EXPECTED_PRIMARY_EVAL_SUITE_IDS)
    contract = ready_contract()
    suite_registry = contract["suite_registry"]
    assert isinstance(suite_registry, dict)
    for index, suite_id in enumerate(all_ids):
        role = suite_registry[suite_id]["role"]
        hashes = [digest(f"{suite_id}:0"), digest(f"{suite_id}:1")]
        manifest: dict[str, object] = {
            "suite_id": suite_id,
            "suite_role": role,
            "domain_tags": suite_registry[suite_id]["domains"],
            "revision": REVISIONS[suite_id],
            "receipt_refs": {
                key: f"receipt://{key}/{suite_id}" for key in RECEIPT_KEYS
            },
            "task_hashes": hashes,
            "aggregate_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        }
        if suite_id == EXPECTED_PRIMARY_EVAL_SUITE_IDS[-1]:
            manifest["held_out"] = True
            manifest["receipt_refs"]["held_out"] = f"receipt://held-out/{suite_id}"
        manifests.append(manifest)
    return manifests


def registry_fixture() -> tuple[dict[str, object], dict[str, object]]:
    contract = ready_contract()
    registry = build_registry(suite_manifests(), contract=contract)
    return registry, contract


def runtime_receipt_for(
    suite: dict[str, object],
    *,
    descriptive_runtime: bool = False,
) -> dict[str, object]:
    refs = dict(suite["receipt_refs"])
    refs["runtime"] = "containerized runtime available" if descriptive_runtime else f"receipt://runtime/{suite['suite_id']}"
    return {
        "suite_id": suite["suite_id"],
        "suite_role": suite["role"],
        "status": "READY",
        "revision": suite["revision"],
        "receipt_refs": refs,
        "task_hashes": list(suite["task_hashes"]),
        "aggregate_sha256": suite["aggregate_sha256"],
    }


def runtime_bundle(registry: dict[str, object]) -> dict[str, object]:
    records = [runtime_receipt_for(suite) for suite in registry["suites"]]
    xlam = {
        "component": "xLAM",
        "suite_id": "pavlov_xlam",
        "status": "READY",
        "revision": "f" * 40,
        "receipt_refs": {
            key: f"receipt://xlam/{key}" for key in RECEIPT_KEYS
        },
        "receipt_refs": {
            **{key: f"receipt://xlam/{key}" for key in RECEIPT_KEYS},
            "runtime": "receipt://xlam/runtime",
        },
    }
    return {"schema_version": "runtime-receipt-bundle-v1", "runtime_receipts": records, "xlam_component": xlam}


class PavlovRunnableSuiteResolverTests(unittest.TestCase):
    def test_resolves_exact_train_and_primary_eval_ids_without_counting_xlam(self) -> None:
        registry, contract = registry_fixture()
        result = resolve_runnable_suites(registry, runtime_bundle(registry), contract=contract)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["runnable_suite_counts"], {"train": 12, "primary_eval": 14})
        self.assertEqual(result["runnable_suite_ids"]["train"], list(EXPECTED_TRAIN_SUITE_IDS))
        self.assertEqual(result["runnable_suite_ids"]["primary_eval"], list(EXPECTED_PRIMARY_EVAL_SUITE_IDS))
        self.assertEqual(result["xlam_component"]["suite_id"], "pavlov_xlam")
        self.assertTrue(result["xlam_component"]["runnable"])
        self.assertNotIn("pavlov_xlam", result["runnable_suite_ids"]["train"] + result["runnable_suite_ids"]["primary_eval"])

    def test_missing_runtime_receipt_is_blocked_and_not_selected(self) -> None:
        registry, contract = registry_fixture()
        bundle = runtime_bundle(registry)
        missing_id = EXPECTED_TRAIN_SUITE_IDS[0]
        bundle["runtime_receipts"] = [
            item for item in bundle["runtime_receipts"] if item["suite_id"] != missing_id
        ]
        result = resolve_runnable_suites(registry, bundle, contract=contract)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertNotIn(missing_id, result["runnable_suite_ids"]["train"])
        self.assertIn(missing_id, result["blocked_suite_ids"]["train"])
        blocked = next(item for item in result["suite_resolutions"] if item["suite_id"] == missing_id)
        self.assertIn("missing runtime receipt", blocked["blockers"])

    def test_mutable_placeholder_and_descriptive_receipts_fail_closed(self) -> None:
        registry, contract = registry_fixture()
        for field, value, message in (
            ("runtime", "latest", "mutable"),
            ("license", "TODO", "placeholder"),
            ("container", "containerized runtime available", "descriptive"),
        ):
            bundle = runtime_bundle(registry)
            target = EXPECTED_TRAIN_SUITE_IDS[1]
            record = next(item for item in bundle["runtime_receipts"] if item["suite_id"] == target)
            record["receipt_refs"][field] = value
            result = resolve_runnable_suites(registry, bundle, contract=contract)
            self.assertEqual(result["status"], "BLOCKED")
            resolution = next(item for item in result["suite_resolutions"] if item["suite_id"] == target)
            self.assertTrue(any(message in blocker for blocker in resolution["blockers"]))
            self.assertNotIn(target, result["runnable_suite_ids"]["train"])

    def test_revision_role_and_aggregate_mismatches_are_rejected(self) -> None:
        registry, contract = registry_fixture()
        target = EXPECTED_PRIMARY_EVAL_SUITE_IDS[0]
        for field, value, message in (
            ("revision", "e" * 40, "revision mismatch"),
            ("suite_role", "train", "role mismatch"),
            ("aggregate_sha256", "0" * 64, "aggregate hash mismatch"),
        ):
            bundle = runtime_bundle(registry)
            record = next(item for item in bundle["runtime_receipts"] if item["suite_id"] == target)
            record[field] = value
            result = resolve_runnable_suites(registry, bundle, contract=contract)
            resolution = next(item for item in result["suite_resolutions"] if item["suite_id"] == target)
            self.assertTrue(any(message in blocker for blocker in resolution["blockers"]))
            self.assertNotIn(target, result["runnable_suite_ids"]["primary_eval"])

    def test_heldout_private_requires_registry_proof_and_remains_primary_eval(self) -> None:
        registry, contract = registry_fixture()
        held_id = EXPECTED_PRIMARY_EVAL_SUITE_IDS[-1]
        bundle = runtime_bundle(registry)
        record = next(item for item in bundle["runtime_receipts"] if item["suite_id"] == held_id)
        record["split_role"] = "held_out"
        record["held_out"] = True
        record["receipt_refs"]["held_out"] = f"receipt://held-out/{held_id}"
        result = resolve_runnable_suites(registry, bundle, contract=contract)
        self.assertIn(held_id, result["runnable_suite_ids"]["primary_eval"])
        self.assertIn(held_id, result["held_out_private_suite_ids"])
        self.assertNotIn(held_id, result["primary_eval_only_suite_ids"])
        resolution = next(item for item in result["suite_resolutions"] if item["suite_id"] == held_id)
        self.assertEqual(resolution["role"], "primary_eval")
        self.assertTrue(resolution["held_out_private"])

        unproven_registry_manifests = suite_manifests()
        unproven = next(item for item in unproven_registry_manifests if item["suite_id"] == held_id)
        unproven["receipt_refs"].pop("held_out")
        unproven_registry = build_registry(unproven_registry_manifests, contract=contract)
        unproven_bundle = runtime_bundle(unproven_registry)
        unproven_record = next(item for item in unproven_bundle["runtime_receipts"] if item["suite_id"] == held_id)
        unproven_record["split_role"] = "private"
        unproven_record["private"] = True
        unproven_record["receipt_refs"]["private"] = f"receipt://private/{held_id}"
        result = resolve_runnable_suites(unproven_registry, unproven_bundle, contract=contract)
        self.assertNotIn(held_id, result["runnable_suite_ids"]["primary_eval"])
        blocked = next(item for item in result["suite_resolutions"] if item["suite_id"] == held_id)
        self.assertTrue(any("registry-proven" in blocker for blocker in blocked["blockers"]))

    def test_registry_blocker_is_global_but_xlam_component_blocker_is_not(self) -> None:
        registry, contract = registry_fixture()
        no_xlam_bundle = runtime_bundle(registry)
        no_xlam_bundle.pop("xlam_component")
        result = resolve_runnable_suites(registry, no_xlam_bundle, contract=contract)
        self.assertEqual(result["status"], "READY")
        self.assertFalse(result["xlam_component"]["runnable"])
        blocked_registry = copy.deepcopy(registry)
        blocked_registry["contract_gate"] = {"blockers": ["contract status is draft"]}
        blocked_registry["blockers"] = ["contract status is draft"]
        blocked_registry["status"] = "BLOCKED"
        blocked_registry["registry_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in blocked_registry.items() if key != "registry_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(RunnableSuiteResolverError, "invalid portfolio split registry"):
            resolve_runnable_suites(blocked_registry, no_xlam_bundle, contract=contract)

    def test_metadata_only_and_deterministic_resolution(self) -> None:
        registry, contract = registry_fixture()
        bundle = runtime_bundle(registry)
        first = resolve_runnable_suites(registry, bundle, contract=contract)
        second = resolve_runnable_suites(copy.deepcopy(registry), copy.deepcopy(bundle), contract=contract)
        self.assertEqual(first, second)
        rendered = json.dumps(first, sort_keys=True)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"target"', rendered)
        self.assertNotIn("secret", rendered)
        self.assertFalse(first["launch_authorized"])
        self.assertFalse(first["launches_any_job"])

    def test_raw_content_and_unknown_runtime_id_are_rejected(self) -> None:
        registry, contract = registry_fixture()
        bundle = runtime_bundle(registry)
        bundle["runtime_receipts"][0]["prompt"] = "secret"
        with self.assertRaisesRegex(RunnableSuiteResolverError, "metadata-only"):
            resolve_runnable_suites(registry, bundle, contract=contract)
        bundle = runtime_bundle(registry)
        extra = copy.deepcopy(bundle["runtime_receipts"][0])
        extra["suite_id"] = "unknown_suite"
        bundle["runtime_receipts"].append(extra)
        result = resolve_runnable_suites(registry, bundle, contract=contract)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("extra runtime receipt" in item for item in result["global_blockers"]))
        self.assertEqual(result["runnable_suite_counts"], {"train": 0, "primary_eval": 0})

    def test_existing_resolution_and_cli_are_local(self) -> None:
        registry, contract = registry_fixture()
        bundle = runtime_bundle(registry)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            bundle_path = root / "runtime.json"
            resolution_path = root / "resolution.json"
            contract_path = root / "contract.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "resolve",
                        "--registry",
                        str(registry_path),
                        "--runtime-receipts",
                        str(bundle_path),
                        "--contract",
                        str(contract_path),
                        "--out",
                        str(resolution_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verify",
                        "--resolution",
                        str(resolution_path),
                        "--registry",
                        str(registry_path),
                        "--runtime-receipts",
                        str(bundle_path),
                        "--contract",
                        str(contract_path),
                    ]
                ),
                0,
            )
            self.assertTrue(verify_resolution(resolution_path, registry_path, bundle_path, contract=contract_path))


if __name__ == "__main__":
    unittest.main()

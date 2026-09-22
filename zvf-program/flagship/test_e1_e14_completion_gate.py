from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flagship.e1_e14_completion_gate import (
    DEFAULT_FOLLOWUP,
    DEFAULT_STATUS,
    EVIDENCE_CLASSES,
    LANE_SPECS,
    PACKAGE_SCHEMA_VERSION,
    PROVIDER_GRANT_SCHEMA_VERSION,
    PackageValidationError,
    TRUST_ROOTS_SCHEMA_VERSION,
    _canonical_sha256,
    audit,
    classify_evidence,
    main,
    validate_package,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CompletionGateTests(unittest.TestCase):
    def make_package(self, root: Path, lane: str = "E3") -> Path:
        contents = {
            "assets/tasks.json": b'{"task":"one"}\n',
            "LICENSE.txt": b"Provider evaluation license\n",
            "runtime.lock": b"image@sha256:runtime\n",
            "grader.py": b"# native grader\n",
            "deploy.json": b'{"deployment":"provider"}\n',
        }
        for name, value in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        task_count = LANE_SPECS[lane].get("expected_task_count", 1)
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "lane": lane,
            "suite_id": LANE_SPECS[lane]["suite_id"],
            "provider_grant": {},
            "assets": [
                {"path": "assets/tasks.json", "sha256": digest(contents["assets/tasks.json"])}
            ],
            "license": {
                "path": "LICENSE.txt",
                "sha256": digest(contents["LICENSE.txt"]),
                "text_sha256": digest(contents["LICENSE.txt"]),
            },
            "runtime": {"path": "runtime.lock", "sha256": digest(contents["runtime.lock"])},
            "native_grader": {"path": "grader.py", "sha256": digest(contents["grader.py"])},
            "tasks": [
                {"task_id": f"private-{index:03d}", "split_id": "held-out"}
                for index in range(1, task_count + 1)
            ],
            "budget_usd": 17.25,
            "deployment_receipt": {
                "path": "deploy.json",
                "sha256": digest(contents["deploy.json"]),
            },
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def sign_manifest(
        self,
        root: Path,
        manifest_path: Path,
        *,
        private_key: Ed25519PrivateKey | None = None,
        root_private_key: Ed25519PrivateKey | None = None,
        grant_id: str = "grant-2026-08-30-123",
        issued_at: datetime | None = None,
        expires_at: datetime | None = None,
        key_id: str = "provider-key-2026-08",
    ) -> Path:
        signing_key = private_key or Ed25519PrivateKey.generate()
        trusted_key = root_private_key or signing_key
        manifest = self.load_manifest(manifest_path)
        assets = manifest["assets"]
        tasks = manifest["tasks"]
        bindings = {
            "assets_sha256": _canonical_sha256(sorted(assets, key=lambda item: item["path"])),
            "license_sha256": manifest["license"]["sha256"],
            "license_text_sha256": manifest["license"]["text_sha256"],
            "runtime_sha256": manifest["runtime"]["sha256"],
            "native_grader_sha256": manifest["native_grader"]["sha256"],
            "deployment_receipt_sha256": manifest["deployment_receipt"]["sha256"],
            "task_split_sha256": _canonical_sha256(sorted(tasks, key=lambda item: item["task_id"])),
            "task_count_sha256": _canonical_sha256(
                {
                    "expected_task_count": LANE_SPECS[manifest["lane"]].get("expected_task_count"),
                    "task_count": len(tasks),
                }
            ),
        }
        now = datetime.now(timezone.utc)
        issued = issued_at or (now - timedelta(minutes=1))
        expires = expires_at or (now + timedelta(days=1))
        public_bytes = trusted_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        fingerprint = digest(public_bytes)
        roots = {
            "schema_version": TRUST_ROOTS_SCHEMA_VERSION,
            "trust_roots": [
                {
                    "lane": manifest["lane"],
                    "suite_id": manifest["suite_id"],
                    "provider": "Example Benchmark Provider",
                    "key_id": key_id,
                    "public_key_b64": base64.b64encode(public_bytes).decode("ascii"),
                    "public_key_sha256": fingerprint,
                    "accepted_grant_ids": [grant_id],
                }
            ],
        }
        grant = {
            "schema_version": PROVIDER_GRANT_SCHEMA_VERSION,
            "lane": manifest["lane"],
            "suite_id": manifest["suite_id"],
            "provider": "Example Benchmark Provider",
            "grant_id": grant_id,
            "issued_at": issued.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "expires_at": expires.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "key_id": key_id,
            "public_key_sha256": fingerprint,
            "trust_root_sha256": _canonical_sha256(roots),
            "bindings": bindings,
        }
        signature = signing_key.sign(
            json.dumps(grant, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        )
        grant["signature_b64"] = base64.b64encode(signature).decode("ascii")
        manifest["provider_grant"] = grant
        self.save_manifest(manifest_path, manifest)
        trust_path = root / "trust-roots.json"
        trust_path.write_text(json.dumps(roots), encoding="utf-8")
        return trust_path

    def make_signed_package(self, root: Path, lane: str = "E3") -> tuple[Path, Path]:
        manifest = self.make_package(root, lane)
        return manifest, self.sign_manifest(root, manifest)

    def load_manifest(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_manifest(self, path: Path, manifest: dict[str, object]) -> None:
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_all_unfinished_lanes_are_enumerated_without_e11(self) -> None:
        receipt = audit(DEFAULT_STATUS, DEFAULT_FOLLOWUP)
        self.assertEqual([row["lane"] for row in receipt["lanes"]], list(LANE_SPECS))
        self.assertEqual(receipt["excluded_complete_lanes"], ["E11"])
        self.assertIn("E1 is also complete exact", receipt["claim_boundary"])
        self.assertIn("outside this gate's scope", receipt["claim_boundary"])
        self.assertTrue(all(row["score"] is None for row in receipt["lanes"]))
        self.assertTrue(all(row["evidence_class"] in EVIDENCE_CLASSES for row in receipt["lanes"]))
        self.assertEqual(receipt["mode"], "READ_ONLY_AUDIT_NO_NETWORK_NO_SPEND_NO_LAUNCH")
        for row in receipt["lanes"]:
            self.assertEqual(
                {surface["kind"] for surface in row["local_surfaces"]},
                {"adapter", "runner", "test"},
            )
            self.assertTrue(row["required_inputs"])

    def test_all_evidence_classes_are_known_but_unknown_status_fails_closed(self) -> None:
        self.assertEqual(
            set(
                classify_evidence(status)
                for status in (
                    "SCORED_EXACT_SUITE",
                    "PARTIAL_EXACT",
                    "PARTIAL_RECOVERY",
                    "BLOCKED_EXTERNAL",
                    "LOCAL_SETUP_READY_PROVIDER_INPUT_REQUIRED",
                )
            ),
            EVIDENCE_CLASSES,
        )
        with self.assertRaises(PackageValidationError):
            classify_evidence("PUBLIC_SAMPLE")

    def test_audit_cli_writes_receipt_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "receipt.json"
            self.assertEqual(main(["audit", "--output", str(output)]), 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "e1-e14-completion-gate-v1")
            self.assertFalse(list(output.parent.glob(".receipt.json.*")))

    def test_valid_package_is_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, trust_roots = self.make_signed_package(root)
            result = validate_package("E3", manifest, root, trust_roots)
            self.assertEqual(result["task_count"], 80)
            self.assertEqual(result["expected_task_count"], 80)
            self.assertEqual(result["split_ids"], ["held-out"])
            self.assertEqual(
                result["validation"], "VALIDATED_INPUTS_ONLY_NOT_A_SCORE_OR_LAUNCH_AUTHORIZATION"
            )
            self.assertNotIn("score", result)
            self.assertEqual(
                result["trust_root_sha256"], result["provider_grant"]["trust_root_sha256"]
            )

    def test_rejects_partial_package_for_fixed_full_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_package(root, "E3")
            data = self.load_manifest(manifest)
            data["tasks"].pop()
            self.save_manifest(manifest, data)
            trust_roots = self.sign_manifest(root, manifest)
            with self.assertRaisesRegex(
                PackageValidationError, "exact E3 full suite of 80 tasks; got 79"
            ):
                validate_package("E3", manifest, root, trust_roots)

    def test_validate_package_cli_can_write_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, trust_roots = self.make_signed_package(root, "E14")
            output = root / "validated.json"
            self.assertEqual(
                main(
                    [
                        "validate-package",
                        "--lane",
                        "E14",
                        "--manifest",
                        str(manifest),
                        "--package-root",
                        str(root),
                        "--trust-roots",
                        str(trust_roots),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(output.read_text())["lane"], "E14")

    def test_rejects_manifest_schema_adversaries(self) -> None:
        cases = {
            "unknown": lambda data: data.__setitem__("unexpected", "x"),
            "suite_mismatch": lambda data: data.__setitem__("suite_id", "wrong-suite"),
            "grant_unknown": lambda data: data["provider_grant"].__setitem__("extra", "x"),
            "bool_budget": lambda data: data.__setitem__("budget_usd", True),
            "nonfinite_budget": lambda data: data.__setitem__("budget_usd", math.inf),
            "missing_hash": lambda data: data["runtime"].__setitem__("sha256", "not-a-hash"),
            "duplicate_task": lambda data: data.__setitem__(
                "tasks",
                [{"task_id": "same", "split_id": "a"}, {"task_id": "same", "split_id": "b"}],
            ),
            "path_traversal": lambda data: data["native_grader"].__setitem__(
                "path", "../grader.py"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, trust_roots = self.make_signed_package(root)
                manifest = self.load_manifest(manifest_path)
                mutate(manifest)
                self.save_manifest(manifest_path, manifest)
                with self.assertRaises(PackageValidationError):
                    validate_package("E3", manifest_path, root, trust_roots)

    def test_rejects_file_hash_and_license_text_hash_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trust_roots = self.make_signed_package(root)
            manifest = self.load_manifest(manifest_path)
            manifest["license"]["text_sha256"] = "0" * 64
            self.save_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(PackageValidationError, "license.text_sha256"):
                validate_package("E3", manifest_path, root, trust_roots)
            manifest = self.load_manifest(manifest_path)
            manifest["license"]["text_sha256"] = digest((root / "LICENSE.txt").read_bytes())
            (root / "assets/tasks.json").write_bytes(b"tampered")
            self.save_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(PackageValidationError, "SHA-256 mismatch"):
                validate_package("E3", manifest_path, root, trust_roots)

    def test_rejects_symlinks_and_root_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            root.mkdir()
            manifest_path, trust_roots = self.make_signed_package(root)
            target = root / "outside.txt"
            target.write_text("external", encoding="utf-8")
            (root / "assets/tasks.json").unlink()
            os.symlink(target, root / "assets/tasks.json")
            with self.assertRaisesRegex(PackageValidationError, "symlinks"):
                validate_package("E3", manifest_path, root, trust_roots)
            link_root = Path(directory) / "package-link"
            os.symlink(root, link_root)
            with self.assertRaisesRegex(PackageValidationError, "non-symlink"):
                validate_package("E3", manifest_path, link_root, trust_roots)

    def test_package_does_not_accept_a_complete_lane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, trust_roots = self.make_signed_package(root)
            with self.assertRaisesRegex(PackageValidationError, "unknown or complete"):
                validate_package("E11", manifest, root, trust_roots)

    def test_requires_pinned_trust_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_package(root)
            with self.assertRaisesRegex(PackageValidationError, "trust-roots is required"):
                validate_package("E3", manifest, root)

    def test_rejects_attacker_key_modified_payload_wrong_key_and_invalid_times(self) -> None:
        cases = (
            "attacker_key",
            "modified_payload",
            "wrong_lane",
            "wrong_key",
            "expired",
            "future",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path = self.make_package(root)
                if case == "attacker_key":
                    trusted = Ed25519PrivateKey.generate()
                    attacker = Ed25519PrivateKey.generate()
                    trust_roots = self.sign_manifest(
                        root, manifest_path, private_key=attacker, root_private_key=trusted
                    )
                elif case == "expired":
                    now = datetime.now(timezone.utc)
                    trust_roots = self.sign_manifest(
                        root,
                        manifest_path,
                        issued_at=now - timedelta(days=2),
                        expires_at=now - timedelta(days=1),
                    )
                elif case == "future":
                    now = datetime.now(timezone.utc)
                    trust_roots = self.sign_manifest(
                        root,
                        manifest_path,
                        issued_at=now + timedelta(days=1),
                        expires_at=now + timedelta(days=2),
                    )
                else:
                    trust_roots = self.sign_manifest(root, manifest_path)
                    manifest = self.load_manifest(manifest_path)
                    if case == "modified_payload":
                        manifest["provider_grant"]["provider"] = "Changed Provider"
                    elif case == "wrong_lane":
                        manifest["provider_grant"]["lane"] = "E14"
                    else:
                        manifest["provider_grant"]["key_id"] = "wrong-key-id"
                    self.save_manifest(manifest_path, manifest)
                with self.assertRaises(PackageValidationError):
                    validate_package("E3", manifest_path, root, trust_roots)

    def test_rejects_trust_root_document_metadata_change_with_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trust_roots = self.make_signed_package(root)
            document = json.loads(trust_roots.read_text(encoding="utf-8"))
            document["trust_roots"][0]["accepted_grant_ids"].append("different-valid-grant")
            trust_roots.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(PackageValidationError, "trust_root_sha256"):
                validate_package("E3", manifest_path, root, trust_roots)

    def test_rejects_mixed_lane_trust_root_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trust_roots = self.make_signed_package(root)
            document = json.loads(trust_roots.read_text(encoding="utf-8"))
            document["trust_roots"][0]["lane"] = "E14"
            document["trust_roots"][0]["suite_id"] = LANE_SPECS["E14"]["suite_id"]
            trust_roots.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(PackageValidationError, "must be scoped to E3"):
                validate_package("E3", manifest_path, root, trust_roots)

    def test_audit_rejects_invalid_exact_claim_sources(self) -> None:
        source_status = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
        cases = {
            "missing_e1": lambda lanes: lanes.__setitem__(
                slice(None), [row for row in lanes if row.get("lane") != "E1"]
            ),
            "duplicate_e11": lambda lanes: lanes.append(
                copy.deepcopy(next(row for row in lanes if row.get("lane") == "E11"))
            ),
            "mislabelled_e11": lambda lanes: next(
                row for row in lanes if row.get("lane") == "E11"
            ).__setitem__("benchmark_status", "PARTIAL_EXACT"),
            "null_e1_score": lambda lanes: next(
                row for row in lanes if row.get("lane") == "E1"
            ).__setitem__("score", None),
            "bool_e11_score": lambda lanes: next(
                row for row in lanes if row.get("lane") == "E11"
            ).__setitem__("score", True),
            "nonfinite_e1_score": lambda lanes: next(
                row for row in lanes if row.get("lane") == "E1"
            ).__setitem__("score", math.inf),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                status = copy.deepcopy(source_status)
                mutate(status["lanes"])
                status_path = Path(directory) / "status.json"
                status_path.write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaises(PackageValidationError):
                    audit(status_path, DEFAULT_FOLLOWUP)


if __name__ == "__main__":
    unittest.main()

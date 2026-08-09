from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_webbench_eval_adapter import (
    WEBBENCH_CATEGORIES,
    WEBBENCH_DATASET_SHA256,
    WEBBENCH_DOMAINS,
    WEBBENCH_PUBLIC_TASK_COUNT,
    WEBBENCH_RECEIPT_FIELDS,
    WEBBENCH_ROLE,
    WEBBENCH_SOURCE_URL,
    WEBBENCH_SUITE_ID,
    WebBenchDatasetError,
    aggregate_task_hashes,
    build_split_artifacts,
    build_split_manifest,
    characterize_task_set,
    derive_task_records,
    format_task_id,
    load_training_task_manifest,
    prove_split_disjointness,
    read_webbench_csv,
    receipt_digest,
    registrable_host,
    sha256_hex,
    split_manifest_hash,
    task_digest,
    task_ids_hash,
    validate_webbench_manifest,
    write_split_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET = REPO_ROOT / "outputs" / "e6_webbench" / "webbenchfinal.csv"
REAL_TRAINING_MANIFEST = REPO_ROOT / "outputs" / "e6_webbench" / "eval_only_training_task_manifest.json"

# Recomputed by webbench_eval.py's independent hashing scheme over the same
# pinned CSV.  If either implementation drifts, these stop matching.
RUNNER_TASK_ID_HASH = "22afbdd3cc47e6dba1e3c57ddbe5f762b54be5d2af6ac76bbd206c19eb83b12e"
RUNNER_MANIFEST_HASH = "66da44a04ec48fe356b3b0d1c420c40679faa1a7ac650728e254b625bb674a07"

CSV_HEADER = "ID,Starting URL,Category,Task\n"


def _csv(rows: list[tuple[int, str, str, str]], header: str = CSV_HEADER) -> str:
    lines = [header.rstrip("\n")]
    for csv_id, url, category, task in rows:
        escaped = task.replace('"', '""')
        lines.append(f'{csv_id},{url},{category},"{escaped}"')
    return "\n".join(lines) + "\n"


FIXTURE_ROWS = [
    (7, "https://www.example-shop.test", "READ", "List the price.\nOnly use the site."),
    (3, "https://docs.example-wiki.test", "CREATE", "Create a page.\nOnly use the site."),
    (1200, "http://example-jobs.test", "FILE_MANIPULATION", "Upload a resume.\nOnly use the site."),
]


class PavlovWebBenchBoundaryTests(unittest.TestCase):
    def _receipt(self, binding: dict[str, str], number: int, payload_extra: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "identity": f"{number:040x}"[-40:],
            "artifact_digest": f"{number:064x}"[-64:],
        }
        if payload_extra:
            payload.update(payload_extra)
        return {
            "receipt_id": f"{number + 1000:040x}"[-40:],
            "digest": receipt_digest(binding, payload),
            "authenticated": True,
            "cryptographically_bound": True,
            "binding": binding,
            "payload": payload,
        }

    def _manifest(self) -> dict[str, object]:
        task_ids = ["webbench-task-0001", "webbench-task-0002", "webbench-task-0003"]
        split = {
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "split": "evaluation",
            "task_id_hash": task_ids_hash(task_ids),
            "manifest_version": "webbench-eval-v1",
        }
        environment = {
            "container_digest": "1" * 64,
            "runtime_digest": "2" * 64,
            "native_environment": {
                "entrypoint": "webbench_native_environment",
                "state_model": "browser_state_and_artifacts",
            },
            "artifact_contract": {
                "required_artifacts": ["browser_trace", "task_artifact"],
                "state_integrity_checks": ["native_state_hash", "artifact_manifest"],
                "side_effect_policy": "record_and_verify_declared_side_effects",
            },
            "verifier_contract": {
                "verifier_id": "webbench-native-verifier",
                "verifier_revision": "3" * 40,
                "checks": ["native_state", "artifact_integrity", "task_success"],
                "native_state_inspection": True,
            },
        }
        manifest: dict[str, object] = {
            "schema_version": "webbench-boundary-fixture-v1",
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "domains": list(WEBBENCH_DOMAINS),
            "heldout_status": "pending_receipts",
            "source": {
                "suite_id": WEBBENCH_SUITE_ID,
                "name": "Halluminate/WebBench",
                "url": WEBBENCH_SOURCE_URL,
                "revision": "a" * 40,
                "license": "Apache-2.0",
                "revision_receipt": self._receipt(
                    {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "revision"},
                    1,
                    {"revision": "a" * 40},
                ),
                "license_receipt": self._receipt(
                    {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "license"},
                    2,
                    {"license": "Apache-2.0"},
                ),
            },
            "task_ids": task_ids,
            "task_id_hash": task_ids_hash(task_ids),
            "split_manifest": split,
            "split_manifest_hash": split_manifest_hash(split),
            "task_receipt": self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "task"},
                3,
                {"task_id_hash": task_ids_hash(task_ids)},
            ),
            "split_receipt": self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "split"},
                4,
                {"split_manifest_hash": split_manifest_hash(split)},
            ),
            "environment": environment,
        }
        environment["container_receipt"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "container"},
            5,
            {"container_digest": environment["container_digest"], "runtime_digest": environment["runtime_digest"]},
        )
        environment["verifier_receipt"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "verifier"},
            6,
            {"verifier_revision": environment["verifier_contract"]["verifier_revision"]},
        )
        return manifest

    def _with_heldout_receipts(self, manifest: dict[str, object]) -> dict[str, object]:
        result = {"n": 100, "metric": 0.55, "result_id": "b" * 40}
        result_digest = sha256_hex(result)
        receipts: dict[str, object] = {}
        provenance_payloads = {
            "revision": {"revision": manifest["source"]["revision"]},
            "license": {"license": manifest["source"]["license"]},
            "split": {"split_manifest_hash": manifest["split_manifest_hash"]},
            "task": {"task_id_hash": manifest["task_id_hash"]},
            "container": {"container_digest": manifest["environment"]["container_digest"]},
            "decontamination": {"decontamination_hash": "e" * 64},
        }
        for offset, field in enumerate(WEBBENCH_RECEIPT_FIELDS, start=10):
            receipts[field] = self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": field},
                offset,
                provenance_payloads[field],
            )
        result_receipts: dict[str, object] = {}
        common = {"artifact_digest": result_digest, "result_digest": result_digest}
        result_receipts["wandb"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:wandb"},
            20,
            {**common, "run_id": "c" * 40, "run_url": "https://wandb.ai/example/webbench/runs/abc"},
        )
        result_receipts["tinker"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:tinker"},
            21,
            {**common, "provider": "Tinker", "run_id": "d" * 40, "cumulative_cost_usd": 1.25},
        )
        result_receipts["hf"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:hf"},
            22,
            {
                **common,
                "revision": f"{22:040x}"[-40:],
                "visibility": "private",
                "repo_url": "https://huggingface.co/example/webbench",
                "checkpoint_url": "https://huggingface.co/example/webbench/commit/" + f"{22:040x}"[-40:],
            },
        )
        # The repeated URL expression above is intentionally corrected to a
        # valid single URL before hashing; it also makes accidental URL edits in
        # a test fixture visible rather than silently accepted.
        hf_payload = result_receipts["hf"]["payload"]
        hf_payload["checkpoint_url"] = "https://huggingface.co/example/webbench/commit/" + f"{22:040x}"[-40:]
        result_receipts["hf"]["digest"] = receipt_digest(result_receipts["hf"]["binding"], hf_payload)
        manifest["receipts"] = receipts
        manifest["result"] = result
        manifest["result_receipts"] = result_receipts
        manifest["heldout_status"] = "receipt_proven_heldout"
        return manifest

    def test_primary_eval_boundary_is_valid_but_not_heldout_without_receipts(self) -> None:
        report = validate_webbench_manifest(self._manifest())

        self.assertTrue(report["boundary_valid"])
        self.assertTrue(report["primary_eval"])
        self.assertFalse(report["receipt_proven_heldout"])
        self.assertEqual(report["status"], "READY_PRIMARY_EVAL_PENDING_RECEIPTS")

    def test_complete_receipts_prove_heldout_results(self) -> None:
        report = validate_webbench_manifest(self._with_heldout_receipts(self._manifest()))

        self.assertTrue(report["boundary_valid"])
        self.assertTrue(report["receipt_proven_heldout"])
        self.assertTrue(report["heldout_claim_allowed"])
        self.assertEqual(report["status"], "READY_RECEIPT_PROVEN_HELDOUT")

    def test_heldout_role_is_not_a_substitute_for_primary_eval(self) -> None:
        manifest = self._manifest()
        manifest["role"] = "heldout"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("role_invalid", report["blocker_codes"])

    def test_self_attested_heldout_status_without_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["heldout_status"] = "receipt_proven_heldout"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("heldout_status_without_receipts", report["blocker_codes"])

    def test_authoritative_source_identity_rejects_xlam_and_related_benchmarks(self) -> None:
        manifest = self._manifest()
        manifest["source"]["url"] = "https://huggingface.co/example/xlam"
        manifest["related_benchmarks"] = ["xLAM"]

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("source_identity_mismatch", report["blocker_codes"])
        self.assertIn("related_benchmark_substitution", report["blocker_codes"])

    def test_fake_revision_license_and_receipt_strings_block(self) -> None:
        manifest = self._manifest()
        manifest["source"]["revision"] = "main"
        manifest["source"]["license"] = "pending"
        manifest["source"]["revision_receipt"] = "verified"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("revision_not_pinned", report["blocker_codes"])
        self.assertIn("license_not_pinned", report["blocker_codes"])
        self.assertIn("source_receipt_invalid", report["blocker_codes"])

    def test_task_and_split_hashes_must_be_deterministic_and_bound(self) -> None:
        manifest = self._manifest()
        manifest["task_ids"] = list(reversed(manifest["task_ids"]))
        manifest["task_id_hash"] = "f" * 64

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("task_ids_not_deterministic", report["blocker_codes"])
        self.assertIn("task_id_hash_mismatch", report["blocker_codes"])

    def test_task_receipt_cannot_attest_to_a_different_task_hash(self) -> None:
        manifest = self._manifest()
        receipt = manifest["task_receipt"]
        receipt["payload"]["task_id_hash"] = "f" * 64
        receipt["digest"] = receipt_digest(receipt["binding"], receipt["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("task_receipt_payload_mismatch", report["blocker_codes"])

    def test_native_artifact_and_verifier_contracts_are_required(self) -> None:
        manifest = self._manifest()
        del manifest["environment"]["artifact_contract"]
        manifest["environment"]["verifier_contract"] = "passed"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("artifact_contract_invalid", report["blocker_codes"])
        self.assertIn("verifier_contract_invalid", report["blocker_codes"])

    def test_result_material_without_all_wandb_tinker_hf_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["result"] = {"n": 100, "metric": 0.5, "result_id": "b" * 40}

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipts_missing", report["blocker_codes"])

    def test_fake_run_id_url_and_hf_visibility_cannot_clear_result(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["result_receipts"]["wandb"]["payload"]["run_id"] = "fake-run"
        manifest["result_receipts"]["wandb"]["digest"] = receipt_digest(
            manifest["result_receipts"]["wandb"]["binding"],
            manifest["result_receipts"]["wandb"]["payload"],
        )
        manifest["result_receipts"]["hf"]["payload"]["visibility"] = "verified"
        manifest["result_receipts"]["hf"]["digest"] = receipt_digest(
            manifest["result_receipts"]["hf"]["binding"],
            manifest["result_receipts"]["hf"]["payload"],
        )

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_tampered_result_digest_blocks_even_with_receipt_shapes(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["result_receipts"]["tinker"]["payload"]["result_digest"] = "f" * 64
        manifest["result_receipts"]["tinker"]["digest"] = receipt_digest(
            manifest["result_receipts"]["tinker"]["binding"],
            manifest["result_receipts"]["tinker"]["payload"],
        )

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_authoritative_service_urls_reject_credentials_and_query_only_revision(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        wandb = manifest["result_receipts"]["wandb"]
        wandb["payload"]["run_url"] = "https://user:secret@sub.wandb.ai/example/webbench/runs/abc"
        wandb["digest"] = receipt_digest(wandb["binding"], wandb["payload"])
        hf = manifest["result_receipts"]["hf"]
        revision = hf["payload"]["revision"]
        hf["payload"]["checkpoint_url"] = "https://huggingface.co/example/webbench/commit/latest?revision=" + revision
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_public_hf_artifact_requires_explicit_clean_safety_flags(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        hf = manifest["result_receipts"]["hf"]
        hf["payload"]["visibility"] = "public"
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_noncanonical_split_and_typed_boundary_fields_fail_closed(self) -> None:
        manifest = self._manifest()
        manifest["split_manifest"]["noncanonical"] = {"not-json"}
        manifest["domains"] = tuple(WEBBENCH_DOMAINS)
        manifest["heldout_status"] = []

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("split_manifest_not_canonicalizable", report["blocker_codes"])
        self.assertIn("domain_boundary_invalid", report["blocker_codes"])
        self.assertIn("heldout_status_invalid", report["blocker_codes"])

    def test_pending_status_with_complete_receipts_is_not_silently_promoted(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["heldout_status"] = "pending_receipts"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("heldout_status_stale", report["blocker_codes"])

    def test_cli_is_offline_and_returns_json(self) -> None:
        script = Path(__file__).with_name("pavlov_webbench_eval_adapter.py")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(self._manifest(), handle)
            handle.flush()
            process = subprocess.run(
                [sys.executable, str(script), "--manifest", handle.name],
                check=False,
                capture_output=True,
                text=True,
            )

        report = json.loads(process.stdout)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["status"], "READY_PRIMARY_EVAL_PENDING_RECEIPTS")
        self.assertFalse(report["receipt_proven_heldout"])
        self.assertNotIn("Traceback", process.stderr)


class WebBenchSplitDerivationTests(unittest.TestCase):
    """Offline derivation of task identities, hashes, and the split manifest."""

    def _write_fixture(self, directory: Path, rows: list[tuple[int, str, str, str]] | None = None) -> Path:
        path = directory / "webbenchfinal.csv"
        path.write_text(_csv(rows if rows is not None else FIXTURE_ROWS), encoding="utf-8")
        return path

    def _records(self, directory: Path, rows: list[tuple[int, str, str, str]] | None = None) -> list[dict]:
        return derive_task_records(read_webbench_csv(self._write_fixture(directory, rows), expected_sha256=None))

    def test_task_ids_are_zero_padded_so_lexicographic_equals_numeric_order(self) -> None:
        ids = [format_task_id(value) for value in (0, 9, 10, 999, 1000, 2724)]

        self.assertEqual(ids[0], "webbench-task-0000")
        self.assertEqual(ids[-1], "webbench-task-2724")
        self.assertEqual(sorted(ids), ids)
        with self.assertRaises(WebBenchDatasetError):
            format_task_id(-1)
        with self.assertRaises(WebBenchDatasetError):
            format_task_id(True)

    def test_registrable_host_only_strips_leading_www(self) -> None:
        self.assertEqual(registrable_host("www.indeed.com"), "indeed.com")
        self.assertEqual(registrable_host("open.spotify.com"), "open.spotify.com")
        self.assertEqual(registrable_host("WWW.Example.COM"), "example.com")

    def test_task_digest_is_content_addressed_and_row_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            forward = self._records(Path(tmp))
            reversed_rows = list(reversed(FIXTURE_ROWS))
            reverse = self._records(Path(tmp), reversed_rows)

        self.assertEqual([item["task_id"] for item in forward], [item["task_id"] for item in reverse])
        self.assertEqual([item["task_digest"] for item in forward], [item["task_digest"] for item in reverse])
        self.assertEqual([item["task_uid"] for item in forward], [item["task_uid"] for item in reverse])
        # The digest binds content, not the derived label.
        mutated = dict(forward[0])
        mutated["task"] = forward[0]["task"] + " extra"
        self.assertNotEqual(task_digest(mutated), forward[0]["task_digest"])

    def test_derived_records_are_sorted_unique_and_carry_stable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))

        task_ids = [item["task_id"] for item in records]
        self.assertEqual(task_ids, sorted(task_ids))
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual([item["csv_id"] for item in records], [3, 7, 1200])
        self.assertEqual(records[0]["registrable_domain"], "docs.example-wiki.test")
        self.assertEqual(records[2]["scheme"], "http")
        self.assertTrue(all(item["task_uid"].startswith("webbench-uid-") for item in records))

    def test_csv_reader_fails_closed_on_malformed_input(self) -> None:
        cases = {
            "bad_columns": ("id,url,category,task\n1,https://a.test,READ,x\n", "columns"),
            "bad_category": (_csv([(1, "https://a.test", "BROWSE", "do it")]), "category"),
            "empty_task": (_csv([(1, "https://a.test", "READ", "")]), "empty task"),
            "bad_url": (_csv([(1, "ftp://a.test", "READ", "do it")]), "Starting URL"),
            "noninteger_id": ("ID,Starting URL,Category,Task\nx,https://a.test,READ,\"do it\"\n", "non-integer"),
            "duplicate_id": (
                _csv([(1, "https://a.test", "READ", "one"), (1, "https://b.test", "READ", "two")]),
                "duplicate",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (body, fragment) in cases.items():
                path = Path(tmp) / f"{name}.csv"
                path.write_text(body, encoding="utf-8")
                with self.subTest(case=name):
                    with self.assertRaises(WebBenchDatasetError) as ctx:
                        read_webbench_csv(path, expected_sha256=None)
                    self.assertIn(fragment, str(ctx.exception))

    def test_csv_reader_enforces_the_dataset_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_fixture(Path(tmp))
            with self.assertRaises(WebBenchDatasetError) as ctx:
                read_webbench_csv(path, expected_sha256="0" * 64)
            self.assertIn("SHA-256 mismatch", str(ctx.exception))
            missing = Path(tmp) / "absent.csv"
            with self.assertRaises(WebBenchDatasetError):
                read_webbench_csv(missing, expected_sha256=None)

    def test_aggregate_hashes_are_deterministic_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
            shuffled = list(reversed(records))

        first = aggregate_task_hashes(records)
        second = aggregate_task_hashes(shuffled)
        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 3)
        self.assertEqual(first["task_id_hash"], task_ids_hash([item["task_id"] for item in records]))
        self.assertEqual(first["task_digest_hash"], sha256_hex([item["task_digest"] for item in records]))
        # A single content change must move the content hashes but not the ID hash.
        mutated = [dict(item) for item in records]
        mutated[0]["task"] = mutated[0]["task"] + "!"
        mutated[0]["task_digest"] = task_digest(mutated[0])
        third = aggregate_task_hashes(mutated)
        self.assertEqual(third["task_id_hash"], first["task_id_hash"])
        self.assertNotEqual(third["task_digest_hash"], first["task_digest_hash"])
        self.assertNotEqual(third["legacy_runner_manifest_hash"], first["legacy_runner_manifest_hash"])

    def test_built_split_manifest_satisfies_the_boundary_task_and_split_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        task_ids = [item["task_id"] for item in records]
        split = build_split_manifest(task_ids)

        self.assertEqual(split["suite_id"], WEBBENCH_SUITE_ID)
        self.assertEqual(split["role"], WEBBENCH_ROLE)
        self.assertEqual(split["split"], "evaluation")
        self.assertEqual(split["task_id_hash"], task_ids_hash(task_ids))

        manifest = PavlovWebBenchBoundaryTests()._manifest()
        manifest["task_ids"] = task_ids
        manifest["task_id_hash"] = task_ids_hash(task_ids)
        manifest["split_manifest"] = split
        manifest["split_manifest_hash"] = split_manifest_hash(split)
        manifest["task_receipt"]["payload"]["task_id_hash"] = task_ids_hash(task_ids)
        manifest["task_receipt"]["digest"] = receipt_digest(
            manifest["task_receipt"]["binding"], manifest["task_receipt"]["payload"]
        )
        manifest["split_receipt"]["payload"]["split_manifest_hash"] = split_manifest_hash(split)
        manifest["split_receipt"]["digest"] = receipt_digest(
            manifest["split_receipt"]["binding"], manifest["split_receipt"]["payload"]
        )

        report = validate_webbench_manifest(manifest)
        self.assertTrue(report["checks"]["tasks_and_split"], report["blocker_codes"])
        self.assertNotIn("task_id_hash_mismatch", report["blocker_codes"])
        self.assertNotIn("split_manifest_hash_mismatch", report["blocker_codes"])

        with self.assertRaises(WebBenchDatasetError):
            build_split_manifest(task_ids + task_ids[:1])

    def test_training_manifest_accepts_int_string_and_object_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            as_list = root / "list.json"
            as_list.write_text(json.dumps([7, 3]), encoding="utf-8")
            self.assertEqual(
                load_training_task_manifest(as_list)["task_ids"],
                ["webbench-task-0003", "webbench-task-0007"],
            )

            as_strings = root / "strings.json"
            as_strings.write_text(json.dumps({"task_ids": ["webbench-task-0007"]}), encoding="utf-8")
            loaded = load_training_task_manifest(as_strings)
            self.assertEqual(loaded["task_ids"], ["webbench-task-0007"])
            self.assertIsNone(loaded["legacy_runner_task_id_hash"])

            duplicated = root / "dupes.json"
            duplicated.write_text(json.dumps([5, 5]), encoding="utf-8")
            with self.assertRaises(WebBenchDatasetError):
                load_training_task_manifest(duplicated)

            malformed = root / "malformed.json"
            malformed.write_text("{not json", encoding="utf-8")
            with self.assertRaises(WebBenchDatasetError):
                load_training_task_manifest(malformed)

            with self.assertRaises(WebBenchDatasetError):
                load_training_task_manifest(root / "absent.json")

    def test_disjointness_proof_detects_overlap_and_bad_declared_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_ids = ["webbench-task-0003", "webbench-task-0007"]

            empty = root / "empty.json"
            # sha256 of the empty string: the runner's hash of an empty ID list.
            empty.write_text(
                json.dumps(
                    {
                        "task_ids": [],
                        "task_id_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    }
                ),
                encoding="utf-8",
            )
            clean = prove_split_disjointness(eval_ids, load_training_task_manifest(empty))
            self.assertTrue(clean["disjoint"])
            self.assertTrue(clean["valid"])
            self.assertTrue(clean["training_declared_task_id_hash_verified"])
            self.assertEqual(clean["overlap_count"], 0)

            overlapping = root / "overlap.json"
            overlapping.write_text(json.dumps([3]), encoding="utf-8")
            dirty = prove_split_disjointness(eval_ids, load_training_task_manifest(overlapping))
            self.assertFalse(dirty["disjoint"])
            self.assertFalse(dirty["valid"])
            self.assertEqual(dirty["overlap_task_ids"], ["webbench-task-0003"])

            wrong_hash = root / "wrong.json"
            wrong_hash.write_text(json.dumps({"task_ids": [], "task_id_hash": "f" * 64}), encoding="utf-8")
            bad = prove_split_disjointness(eval_ids, load_training_task_manifest(wrong_hash))
            self.assertFalse(bad["valid"])
            self.assertFalse(bad["training_declared_task_id_hash_verified"])

    def test_characterization_reports_id_gaps_and_domain_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        profile = characterize_task_set(records)

        self.assertEqual(profile["task_count"], 3)
        self.assertEqual(profile["csv_id_min"], 3)
        self.assertEqual(profile["csv_id_max"], 1200)
        self.assertFalse(profile["csv_ids_contiguous"])
        self.assertEqual(profile["csv_id_gap_count"], 1195)
        self.assertEqual(profile["distinct_registrable_domains"], 3)
        self.assertEqual(profile["url_schemes"], {"http": 1, "https": 2})
        self.assertEqual(profile["http_only_urls"], ["http://example-jobs.test"])
        self.assertEqual(set(profile["categories"]) - set(WEBBENCH_CATEGORIES), set())
        # CREATE + FILE_MANIPULATION mutate live sites; READ does not.
        self.assertEqual(profile["write_class_task_count"], 2)
        self.assertEqual(profile["keyword_probe_counts"]["captcha"], 0)

    def test_bundle_is_labelled_a_local_derivation_not_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_fixture(root)
            training = root / "training.json"
            training.write_text(json.dumps({"task_ids": []}), encoding="utf-8")
            bundle = build_split_artifacts(
                dataset, training_manifest_path=training, expected_dataset_sha256=None
            )
            written = write_split_artifacts(bundle, root / "out")

            derivation = bundle["derivation"]
            self.assertEqual(derivation["receipt_class"], "local_derivation")
            self.assertIsNone(derivation["authenticated_receipt"])
            self.assertFalse(derivation["externally_bound"])
            self.assertFalse(derivation["network_accessed"])
            self.assertTrue(derivation["disjointness"]["valid"])
            self.assertEqual(len(Path(written["task_index"]).read_text(encoding="utf-8").splitlines()), 3)
            split_payload = json.loads(Path(written["split_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(split_payload["task_count"], 3)
            self.assertIsNone(split_payload["authenticated_receipt"])

            # Re-running is byte-identical: the derivation is deterministic.
            again = build_split_artifacts(
                dataset, training_manifest_path=training, expected_dataset_sha256=None
            )
            self.assertEqual(again["derivation"]["derivation_hash"], derivation["derivation_hash"])

    def test_derived_split_alone_never_unblocks_the_environment_boundary(self) -> None:
        """A local split manifest must not make WebBench look runnable."""

        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        task_ids = [item["task_id"] for item in records]
        split = build_split_manifest(task_ids)
        manifest = {
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "domains": list(WEBBENCH_DOMAINS),
            "heldout_status": "pending_receipts",
            "source": {
                "suite_id": WEBBENCH_SUITE_ID,
                "name": "Halluminate/WebBench",
                "url": WEBBENCH_SOURCE_URL,
                "revision": "a" * 40,
                "license": "MIT",
            },
            "task_ids": task_ids,
            "task_id_hash": task_ids_hash(task_ids),
            "split_manifest": split,
            "split_manifest_hash": split_manifest_hash(split),
        }

        report = validate_webbench_manifest(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("environment_contract_missing", report["blocker_codes"])
        self.assertIn("task_receipt_invalid", report["blocker_codes"])
        self.assertIn("split_receipt_invalid", report["blocker_codes"])
        self.assertFalse(report["receipt_proven_heldout"])


@unittest.skipUnless(REAL_DATASET.is_file(), "pinned WebBench CSV is not on disk")
class WebBenchPinnedDatasetTests(unittest.TestCase):
    """Assertions against the real, pinned, MIT-licensed public CSV."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = derive_task_records(read_webbench_csv(REAL_DATASET, expected_sha256=WEBBENCH_DATASET_SHA256))
        cls.aggregates = aggregate_task_hashes(cls.records)

    def test_public_task_count_and_identity_scheme(self) -> None:
        self.assertEqual(len(self.records), WEBBENCH_PUBLIC_TASK_COUNT)
        task_ids = [item["task_id"] for item in self.records]
        self.assertEqual(task_ids, sorted(task_ids))
        self.assertEqual(len(set(task_ids)), WEBBENCH_PUBLIC_TASK_COUNT)
        self.assertEqual(len({item["task_uid"] for item in self.records}), WEBBENCH_PUBLIC_TASK_COUNT)
        self.assertEqual(len({item["task_digest"] for item in self.records}), WEBBENCH_PUBLIC_TASK_COUNT)

    def test_aggregates_agree_with_the_independent_runner_implementation(self) -> None:
        self.assertEqual(self.aggregates["legacy_runner_task_id_hash"], RUNNER_TASK_ID_HASH)
        self.assertEqual(self.aggregates["legacy_runner_manifest_hash"], RUNNER_MANIFEST_HASH)

    def test_public_ids_are_sparse_which_is_why_ids_must_be_sorted_explicitly(self) -> None:
        profile = characterize_task_set(self.records)
        self.assertFalse(profile["csv_ids_contiguous"])
        self.assertEqual(profile["csv_id_min"], 0)
        self.assertEqual(profile["csv_id_max"], 2724)
        self.assertEqual(profile["csv_id_gap_count"], 78)
        self.assertEqual(sum(profile["categories"].values()), WEBBENCH_PUBLIC_TASK_COUNT)
        self.assertEqual(profile["categories"]["READ"], 1637)
        self.assertEqual(profile["write_class_task_count"], 1010)
        self.assertEqual(profile["distinct_registrable_domains"], 448)
        self.assertEqual(profile["keyword_probe_counts"]["credential_or_account"], 536)

    @unittest.skipUnless(REAL_TRAINING_MANIFEST.is_file(), "training manifest is not on disk")
    def test_evaluation_split_is_disjoint_from_the_training_manifest(self) -> None:
        training = load_training_task_manifest(REAL_TRAINING_MANIFEST)
        proof = prove_split_disjointness([item["task_id"] for item in self.records], training)

        self.assertTrue(proof["disjoint"], proof["overlap_task_ids"][:10])
        self.assertTrue(proof["valid"], proof["errors"])
        self.assertEqual(proof["evaluation_task_count"], WEBBENCH_PUBLIC_TASK_COUNT)
        self.assertTrue(proof["training_declared_task_id_hash_verified"])

    def test_cli_build_split_is_offline_and_writes_artifacts(self) -> None:
        script = Path(__file__).with_name("pavlov_webbench_eval_adapter.py")
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--build-split",
                    "--dataset",
                    str(REAL_DATASET),
                    "--training-task-manifest",
                    str(REAL_TRAINING_MANIFEST),
                    "--out-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            payload = json.loads(process.stdout)
            self.assertEqual(payload["derivation"]["aggregates"]["task_count"], WEBBENCH_PUBLIC_TASK_COUNT)
            self.assertFalse(payload["derivation"]["network_accessed"])
            index_lines = Path(payload["written"]["task_index"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(index_lines), WEBBENCH_PUBLIC_TASK_COUNT)


if __name__ == "__main__":
    unittest.main()

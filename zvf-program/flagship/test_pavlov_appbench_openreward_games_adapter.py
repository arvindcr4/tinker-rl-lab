from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

try:
    from . import pavlov_appbench_openreward_games_adapter as adapter
except ImportError:
    _adapter_path = Path(__file__).with_name("pavlov_appbench_openreward_games_adapter.py")
    _spec = importlib.util.spec_from_file_location("adapter", _adapter_path)
    if _spec is None or _spec.loader is None:
        raise
    adapter = importlib.util.module_from_spec(_spec)  # type: ignore[assignment]
    _spec.loader.exec_module(adapter)  # type: ignore[union-attr]

_VALID_TASK_ID_1 = hashlib.sha256(b"e12-appbench-task-id-1").hexdigest()
_VALID_TASK_ID_2 = hashlib.sha256(b"e12-appbench-task-id-2").hexdigest()
_VALID_TASK_ID_3 = hashlib.sha256(b"e13-openreward-task-id-3").hexdigest()
_VALID_TASK_ID_4 = hashlib.sha256(b"e13-openreward-task-id-4").hexdigest()
_VALID_REVISION_E12 = hashlib.sha1(b"appbench-e12-revision").hexdigest()
_VALID_REVISION_E13 = hashlib.sha1(b"openreward-e13-revision").hexdigest()

_VALID_LICENSE_SHA = hashlib.sha256(b"appbench-openreward-license").hexdigest()
_VALID_ARTIFACT_SHA = hashlib.sha256(b"appbench-openreward-artifact").hexdigest()
_VALID_CONTAINER_SHA = hashlib.sha256(b"appbench-openreward-container").hexdigest()
_VALID_DECONTAMINATION_SHA = hashlib.sha256(b"appbench-openreward-decontamination").hexdigest()
_VALID_ARTIFACT_RECEIPT_SHA = hashlib.sha256(b"appbench-openreward-artifact-verifier").hexdigest()
_VALID_STATE_RECEIPT_SHA = hashlib.sha256(b"appbench-openreward-state-verifier").hexdigest()
_VALID_TINKER_RECEIPT_SHA = hashlib.sha256(b"appbench-openreward-tinker").hexdigest()
_VALID_HF_RECEIPT_SHA = hashlib.sha256(b"appbench-openreward-hf").hexdigest()
_VALID_HF_COMMIT = hashlib.sha1(b"appbench-openreward-repo-commit").hexdigest()


def _split_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_boundary(name: str, role: str, source: str, revision: str, task_ids: list[str]) -> dict:
    return {
        "name": name,
        "authoritative_source": source,
        "revision": revision,
        "evaluation_role": role,
        "task_ids": task_ids,
        "split_hash": _split_hash(task_ids),
        "license": {
            "sha256": _VALID_LICENSE_SHA,
            "path": "licenses/reproducible-license.txt",
        },
        "native_contract": {
            "artifact_sha256": _VALID_ARTIFACT_SHA,
            "artifact_source": "registry://zvf/appbench-artifact-v1",
            "artifact_size_bytes": 123,
            "environment": {
                "container": _VALID_CONTAINER_SHA,
                "decontamination": _VALID_DECONTAMINATION_SHA,
                "container_source": "container://sha256:dummy-container-hash",
                "decontamination_source": "container://sha256:dummy-decontam-hash",
            },
            "verifier": {
                "artifact_verifier_receipt": {"sha256": _VALID_ARTIFACT_RECEIPT_SHA},
                "state_verifier_receipt": {"sha256": _VALID_STATE_RECEIPT_SHA},
                "wandb": {
                    "project": "flagship",
                    "entity": "team",
                    "run_id": "a1B2c3D4",
                    "run_url": "https://wandb.ai/team/flagship/runs/a1B2c3D4",
                },
                "tinker": {
                    "job_id": "123e4567-e89b-12d3-a456-426614174000",
                    "receipt_sha256": _VALID_TINKER_RECEIPT_SHA,
                },
                "hugging_face": {
                    "repo_id": "org/reward-evals",
                    "commit": _VALID_HF_COMMIT,
                    "receipt_sha256": _VALID_HF_RECEIPT_SHA,
                },
                "artifact": "results/artifact.json",
            },
        },
    }


def _base_contract() -> dict:
    return {
        "paid_launch_allowed": False,
        "stateful_trajectory": True,
        "boundaries": {
            "E12": _valid_boundary(
                name="appbench_eval",
                role="receipt_proven_heldout",
                source="AppBench official evaluation source",
                revision=_VALID_REVISION_E12,
                task_ids=[_VALID_TASK_ID_1, _VALID_TASK_ID_2],
            ),
            "E13": _valid_boundary(
                name="openreward_games_eval",
                role="primary_eval",
                source="OpenReward Games official benchmark spec",
                revision=_VALID_REVISION_E13,
                task_ids=[_VALID_TASK_ID_3, _VALID_TASK_ID_4],
            ),
        },
    }


class PavlovAppbenchOpenrewardGamesAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = _base_contract()

    def _validate(self, updates: dict | None = None):
        payload = copy.deepcopy(self.contract)
        if updates:
            payload.update(updates)
        return adapter.validate_pavlov_openreward_games_adapter(payload)

    def test_valid_contract_passes(self) -> None:
        result = self._validate()
        self.assertFalse(result["paid_launch_allowed"])
        self.assertTrue(result["stateful_trajectory"])
        self.assertEqual(result["boundaries"]["E12"]["name"], "appbench_eval")
        self.assertEqual(result["boundaries"]["E12"]["evaluation_role"], "receipt_proven_heldout")
        self.assertEqual(result["boundaries"]["E13"]["name"], "openreward_games_eval")
        self.assertEqual(result["boundaries"]["E13"]["evaluation_role"], "primary_eval")

    def test_paid_launch_allowed_must_be_false(self) -> None:
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError, "paid_launch_allowed must be False"
        ):
            self._validate({"paid_launch_allowed": True})

    def test_stateful_trajectory_must_be_true(self) -> None:
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "stateful_trajectory must be true",
        ):
            self._validate({"stateful_trajectory": False})

    def test_boundaries_must_be_exact_e12_e13(self) -> None:
        missing = copy.deepcopy(self.contract)
        del missing["boundaries"]["E12"]
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "boundaries must contain exactly",
        ):
            adapter.validate_pavlov_openreward_games_adapter(missing)

        extra = copy.deepcopy(self.contract)
        extra["boundaries"]["E99"] = _valid_boundary(
            name="unused_eval",
            role="primary_eval",
            source="extra official source",
            revision=_VALID_REVISION_E13,
            task_ids=["e" * 64],
        )
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "boundaries must contain exactly",
        ):
            adapter.validate_pavlov_openreward_games_adapter(extra)

    def test_roles_must_match_semantics(self) -> None:
        bad_e12 = copy.deepcopy(self.contract)
        bad_e12["boundaries"]["E12"]["evaluation_role"] = "primary_eval"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "evaluation_role must be 'receipt_proven_heldout'",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad_e12)

        bad_e13 = copy.deepcopy(self.contract)
        bad_e13["boundaries"]["E13"]["evaluation_role"] = "receipt_proven_heldout"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "evaluation_role must be 'primary_eval'",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad_e13)

    def test_revisions_must_be_40_hex(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["revision"] = "g" * 39
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            r"boundaries\[E12\]\.revision",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["revision"] = "g" * 40 + "x"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            r"boundaries\[E13\]\.revision",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_revisions_cannot_use_placeholder_hashes(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["revision"] = "0" * 40
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must not be an all-identical placeholder digest",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["revision"] = "f" * 40
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must not be an all-identical placeholder digest",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_task_ids_must_be_64_hex(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["task_ids"][0] = "z" * 64
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            r"boundaries\[E12\]\.task_ids\[0\]",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["task_ids"][0] = _VALID_TASK_ID_1
        bad["boundaries"]["E12"]["task_ids"][1] = _VALID_TASK_ID_1
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must be unique",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_task_ids_cannot_use_placeholder_hashes(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["task_ids"][0] = "f" * 64
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must not be an all-identical placeholder digest",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_non_overlapping_task_ids(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["task_ids"] = [_VALID_TASK_ID_1, _VALID_TASK_ID_4]
        bad["boundaries"]["E13"]["split_hash"] = _split_hash(bad["boundaries"]["E13"]["task_ids"])
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must not overlap",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_split_hash_is_deterministic_of_task_ids(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["split_hash"] = hashlib.sha256(b"bad split hash").hexdigest()
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "split_hash is not the deterministic hash",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_authoritative_source_markers_reject_substitution(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["authoritative_source"] = "xLAM benchmark mirror"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "references blocked source marker",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["authoritative_source"] = "related benchmark dataset"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "references blocked source marker",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_authoritative_source_must_match_expected_marker(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E12"]["authoritative_source"] = "OpenReward Games benchmark specs"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must identify the authoritative source 'appbench'",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_real_upstream_source_urls_satisfy_the_markers(self) -> None:
        """The marker check must accept the punctuation upstream actually uses.

        E12's dataset is ``AfterQuery/App-Bench`` on the Hub and ``app-bench`` in
        the leaderboard URL; neither contains the literal substring ``appbench``.
        """

        for source in (
            "https://huggingface.co/datasets/AfterQuery/App-Bench",
            "https://www.afterquery.com/leaderboard/app-bench",
            "AfterQuery App Bench official task CSV",
            "AppBench official evaluation source",
        ):
            with self.subTest(source=source):
                ok = copy.deepcopy(self.contract)
                ok["boundaries"]["E12"]["authoritative_source"] = source
                normalized = adapter.validate_pavlov_openreward_games_adapter(ok)
                self.assertEqual(normalized["boundaries"]["E12"]["authoritative_source"], source)

    def test_punctuated_substitution_markers_are_still_blocked(self) -> None:
        for source in (
            "x-LAM benchmark mirror for AppBench",
            "AppBench via related_benchmark bundle",
            "AppBench (related-benchmark stand-in)",
        ):
            with self.subTest(source=source):
                bad = copy.deepcopy(self.contract)
                bad["boundaries"]["E12"]["authoritative_source"] = source
                with self.assertRaisesRegex(
                    adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                    "references blocked source marker",
                ):
                    adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_artifact_environment_and_verifier_contracts_are_required(self) -> None:
        missing = copy.deepcopy(self.contract)
        del missing["boundaries"]["E12"]["native_contract"]
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            r"boundaries\[E12\]\.native_contract",
        ):
            adapter.validate_pavlov_openreward_games_adapter(missing)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["artifact_size_bytes"] = -1
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "artifact_size_bytes must be non-negative",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["environment"]["container_source"] = ""
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "cannot be empty",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["wandb"]["run_id"] = "12345"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "wandb.run_id must be 8 alphanumeric",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["tinker"]["job_id"] = "invalid"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "tinker.job_id must be a hyphenated UUID string",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["hugging_face"]["repo_id"] = (
            "invalid_repo"
        )
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "hugging_face.repo_id must be owner/repo",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["artifact_verifier_receipt"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "must not be an all-identical placeholder digest",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

    def test_receipt_and_hash_placeholders_rejected_everywhere(self) -> None:
        for boundary_id in ("E12", "E13"):
            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["license"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

        for boundary_id in ("E12", "E13"):
            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["environment"]["container"] = "f" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["environment"]["decontamination"] = (
                "a" * 64
            )
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"][
                "artifact_verifier_receipt"
            ]["sha256"] = "b" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["state_verifier_receipt"][
                "sha256"
            ] = "c" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["tinker"][
                "receipt_sha256"
            ] = "d" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["hugging_face"][
                "receipt_sha256"
            ] = "e" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["hugging_face"][
                "commit"
            ] = "0" * 40
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)


class AppBenchGrantTests(unittest.TestCase):
    _PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    )
    TEST_ROOT = {
        "schema_version": adapter.APPBENCH_TRUST_ROOT_SCHEMA,
        "lane": "E12",
        "suite_id": "appbench_eval",
        "provider": "AfterQuery",
        "key_id": "afterquery-test-e12",
        "public_key_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    }

    @classmethod
    def _sign(cls, payload: dict) -> dict:
        unsigned = copy.deepcopy(payload)
        unsigned["signature_key_id"] = cls.TEST_ROOT["key_id"]
        unsigned.pop("signature", None)
        unsigned["signature"] = base64.b64encode(
            cls._PRIVATE_KEY.sign(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            )
        ).decode()
        return unsigned

    def _grant(self) -> dict:
        return self._sign(
            {
                "schema_version": adapter.APPBENCH_GRANT_SCHEMA,
                "lane": "E12",
                "suite_id": "appbench_eval",
                "provider": "AfterQuery",
                "issued_at": "2025-08-30T00:00:00Z",
                "expires_at": "2030-08-30T00:00:00Z",
                "grant_id": "g1",
                "license": {"approved": True, "receipt_id": "l1", "sha256": "a" * 64},
                "deployment": {
                    "revision": "b" * 40,
                    "container_sha256": "c" * 64,
                    "reset_receipt_id": "r1",
                },
                "graders": {"count": 2, "re_adjudication": True, "protocol_sha256": "d" * 64},
                "heldout": {
                    "contamination_checked": True,
                    "manifest_sha256": "e" * 64,
                    "task_count": 6,
                    "task_artifact_sha256": "f" * 64,
                },
                "runtime": {
                    "revision": "1" * 40,
                    "container_sha256": "2" * 64,
                    "endpoint": "https://afterquery.example/run",
                },
                "verifier": {"revision": "3" * 40, "sha256": "4" * 64, "approval_id": "v1"},
                "receipt_url": "https://afterquery.example/receipt",
            }
        )

    def _result(self, grant: dict, handoff: dict) -> dict:
        model_revision = hashlib.sha1(b"appbench-signed-model-revision").hexdigest()
        return self._sign(
            {
                "schema_version": adapter.APPBENCH_RESULT_SCHEMA,
                "suite_id": "appbench_eval",
                "provider": "AfterQuery",
                "issued_at": grant["issued_at"],
                "expires_at": grant["expires_at"],
                "grant_fingerprint": adapter._canonical_hash(grant, "signature"),
                "handoff_fingerprint": hashlib.sha256(
                    json.dumps(handoff, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "heldout_manifest_sha256": grant["heldout"]["manifest_sha256"],
                "task_count": grant["heldout"]["task_count"],
                "task_artifact_sha256": grant["heldout"]["task_artifact_sha256"],
                "deployment_revision": grant["deployment"]["revision"],
                "deployment_container_sha256": grant["deployment"]["container_sha256"],
                "runtime_revision": grant["runtime"]["revision"],
                "runtime_container_sha256": grant["runtime"]["container_sha256"],
                "verifier_revision": grant["verifier"]["revision"],
                "verifier_sha256": grant["verifier"]["sha256"],
                "graders": copy.deepcopy(grant["graders"]),
                "model_revision": model_revision,
                "artifact_sha256": hashlib.sha256(b"appbench-provider-artifact").hexdigest(),
                "wandb_before_tinker": True,
                "receipts": {
                    "wandb": {
                        "project": "appbench",
                        "entity": "afterquery",
                        "run_id": "A1b2C3d4",
                        "run_url": "https://wandb.ai/afterquery/appbench/runs/A1b2C3d4",
                    },
                    "tinker": {
                        "job_id": "123e4567-e89b-12d3-a456-426614174000",
                        "receipt_sha256": hashlib.sha256(b"appbench-tinker-receipt").hexdigest(),
                    },
                    "hugging_face": {
                        "repo_id": "afterquery/appbench-model",
                        "commit": model_revision,
                        "receipt_sha256": hashlib.sha256(b"appbench-hf-receipt").hexdigest(),
                    },
                },
                "metric": "appbench_score",
                "score": 0.75,
                "completed_at": "2026-08-30T00:00:00Z",
            }
        )

    def _valid_triplet(self) -> tuple[dict, dict, dict]:
        grant = self._grant()
        handoff = adapter.build_appbench_provider_handoff(grant, trust_root=self.TEST_ROOT)
        return grant, handoff, self._result(grant, handoff)

    def test_grant_requires_two_grader_protocol(self):
        grant = self._grant()
        with self.assertRaises(adapter.PavlovAppbenchOpenrewardGamesAdapterError):
            adapter.validate_appbench_provider_grant(grant)
        self.assertFalse(
            adapter.build_appbench_provider_handoff(grant, trust_root=self.TEST_ROOT)[
                "paid_launch_allowed"
            ]
        )
        grant["graders"]["count"] = 1
        with self.assertRaises(adapter.PavlovAppbenchOpenrewardGamesAdapterError):
            adapter.validate_appbench_provider_grant(grant, trust_root=self.TEST_ROOT)

    def test_only_exact_provider_signed_result_can_complete(self):
        grant, handoff, result = self._valid_triplet()
        collected = adapter.collect_appbench_signed_result(
            grant, handoff, result, trust_root=self.TEST_ROOT
        )
        self.assertEqual(collected["status"], "COMPLETE")
        self.assertEqual(collected["score"], 0.75)
        self.assertTrue(collected["provider_signed"])

    def test_signed_non_six_task_grants_and_results_are_rejected(self):
        for task_count in (5, 7):
            with self.subTest(kind="grant", task_count=task_count):
                grant = self._grant()
                grant["heldout"]["task_count"] = task_count
                grant = self._sign(grant)
                with self.assertRaisesRegex(
                    adapter.PavlovAppbenchOpenrewardGamesAdapterError, "exact six-task"
                ):
                    adapter.validate_appbench_provider_grant(grant, trust_root=self.TEST_ROOT)

        grant, handoff, result = self._valid_triplet()
        for task_count in (5, 7):
            with self.subTest(kind="result", task_count=task_count):
                wrong_count = copy.deepcopy(result)
                wrong_count["task_count"] = task_count
                wrong_count = self._sign(wrong_count)
                with self.assertRaisesRegex(
                    adapter.PavlovAppbenchOpenrewardGamesAdapterError, "bindings"
                ):
                    adapter.collect_appbench_signed_result(
                        grant, handoff, wrong_count, trust_root=self.TEST_ROOT
                    )

    def test_tampered_signature_and_exact_bindings_are_rejected(self):
        grant, handoff, result = self._valid_triplet()
        forged = copy.deepcopy(result)
        forged["score"] = 0.9
        with self.assertRaisesRegex(adapter.PavlovAppbenchOpenrewardGamesAdapterError, "signature"):
            adapter.collect_appbench_signed_result(
                grant, handoff, forged, trust_root=self.TEST_ROOT
            )

        wrong_task_count = copy.deepcopy(result)
        wrong_task_count["task_count"] += 1
        wrong_task_count = self._sign(wrong_task_count)
        with self.assertRaisesRegex(adapter.PavlovAppbenchOpenrewardGamesAdapterError, "bindings"):
            adapter.collect_appbench_signed_result(
                grant, handoff, wrong_task_count, trust_root=self.TEST_ROOT
            )

        wrong_handoff = copy.deepcopy(result)
        wrong_handoff["handoff_fingerprint"] = hashlib.sha256(b"other handoff").hexdigest()
        wrong_handoff = self._sign(wrong_handoff)
        with self.assertRaisesRegex(adapter.PavlovAppbenchOpenrewardGamesAdapterError, "bindings"):
            adapter.collect_appbench_signed_result(
                grant, handoff, wrong_handoff, trust_root=self.TEST_ROOT
            )

    def test_signed_result_time_validity_is_required(self):
        grant, handoff, result = self._valid_triplet()
        expired = copy.deepcopy(result)
        expired["expires_at"] = "2020-08-30T00:00:00Z"
        expired = self._sign(expired)
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError, "validity window"
        ):
            adapter.collect_appbench_signed_result(
                grant, handoff, expired, trust_root=self.TEST_ROOT
            )

        future_completion = copy.deepcopy(result)
        future_completion["completed_at"] = "2029-08-30T00:00:00Z"
        future_completion = self._sign(future_completion)
        with self.assertRaisesRegex(adapter.PavlovAppbenchOpenrewardGamesAdapterError, "outside"):
            adapter.collect_appbench_signed_result(
                grant, handoff, future_completion, trust_root=self.TEST_ROOT
            )

    def test_two_graders_and_adjudication_are_bound_in_result(self):
        grant, handoff, result = self._valid_triplet()
        one_grader = copy.deepcopy(result)
        one_grader["graders"]["count"] = 1
        one_grader = self._sign(one_grader)
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError, "two-grader"
        ):
            adapter.collect_appbench_signed_result(
                grant, handoff, one_grader, trust_root=self.TEST_ROOT
            )

        no_adjudication = copy.deepcopy(result)
        no_adjudication["graders"]["re_adjudication"] = False
        no_adjudication = self._sign(no_adjudication)
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError, "two-grader"
        ):
            adapter.collect_appbench_signed_result(
                grant, handoff, no_adjudication, trust_root=self.TEST_ROOT
            )

    def test_wandb_tinker_and_hugging_face_receipts_are_required(self):
        grant, handoff, result = self._valid_triplet()
        for receipt_name in ("wandb", "tinker", "hugging_face"):
            with self.subTest(receipt_name=receipt_name):
                missing = copy.deepcopy(result)
                del missing["receipts"][receipt_name]
                missing = self._sign(missing)
                with self.assertRaisesRegex(
                    adapter.PavlovAppbenchOpenrewardGamesAdapterError, "result.receipts"
                ):
                    adapter.collect_appbench_signed_result(
                        grant, handoff, missing, trust_root=self.TEST_ROOT
                    )

    def test_collect_cli_emits_only_verified_complete_receipt(self):
        grant, handoff, result = self._valid_triplet()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grant_path, handoff_path, result_path, trust_path, out_path = (
                root / "grant.json",
                root / "handoff.json",
                root / "result.json",
                root / "trust.json",
                root / "collected.json",
            )
            for path, payload in (
                (grant_path, grant),
                (handoff_path, handoff),
                (result_path, result),
                (trust_path, self.TEST_ROOT),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                adapter.main(
                    [
                        "--mode",
                        "collect",
                        "--grant",
                        str(grant_path),
                        "--handoff",
                        str(handoff_path),
                        "--result",
                        str(result_path),
                        "--trust-root",
                        str(trust_path),
                        "--out",
                        str(out_path),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(out_path.read_text(encoding="utf-8"))["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()

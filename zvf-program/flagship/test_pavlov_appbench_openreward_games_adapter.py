from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

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
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")).hexdigest()


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
        with self.assertRaisesRegex(adapter.PavlovAppbenchOpenrewardGamesAdapterError, "paid_launch_allowed must be False"):
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
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["hugging_face"]["repo_id"] = "invalid_repo"
        with self.assertRaisesRegex(
            adapter.PavlovAppbenchOpenrewardGamesAdapterError,
            "hugging_face.repo_id must be owner/repo",
        ):
            adapter.validate_pavlov_openreward_games_adapter(bad)

        bad = copy.deepcopy(self.contract)
        bad["boundaries"]["E13"]["native_contract"]["verifier"]["artifact_verifier_receipt"]["sha256"] = "0" * 64
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
            bad["boundaries"][boundary_id]["native_contract"]["environment"]["decontamination"] = "a" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["artifact_verifier_receipt"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["state_verifier_receipt"]["sha256"] = "c" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["tinker"]["receipt_sha256"] = "d" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["hugging_face"]["receipt_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)

            bad = copy.deepcopy(self.contract)
            bad["boundaries"][boundary_id]["native_contract"]["verifier"]["hugging_face"]["commit"] = "0" * 40
            with self.assertRaisesRegex(
                adapter.PavlovAppbenchOpenrewardGamesAdapterError,
                "must not be an all-identical placeholder digest",
            ):
                adapter.validate_pavlov_openreward_games_adapter(bad)


if __name__ == "__main__":
    unittest.main()

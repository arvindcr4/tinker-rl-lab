from __future__ import annotations

import copy
import unittest

from . import pavlov_openreward_schema as schema


_VALID_HASH = "a" * 64
_VALID_REVISION = "b" * 40
_ANOTHER_REVISION = "c" * 40


def _base_contract():
    return {
        "paid_launch_allowed": False,
        "stateful_trajectory": True,
        "openreward_train": {
            "revision": _VALID_REVISION,
            "role": "train",
            "task_hashes": [_VALID_HASH, "b" * 64],
        },
        "openreward_games_eval": {
            "revision": _ANOTHER_REVISION,
            "role": "primary_eval",
            "task_hashes": ["c" * 64, "d" * 64],
        },
        "artifact_verifier_receipt": {"sha256": "e" * 64},
        "state_verifier_receipt": {"sha256": "f" * 64},
        "license": {"sha256": "1" * 64},
        "container": {"sha256": "2" * 64},
        "decontamination": {"sha256": "3" * 64},
    }


class OpenrewardSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = _base_contract()

    def _validate(self, updates: dict | None = None):
        contract = copy.deepcopy(self.contract)
        if updates:
            contract.update(updates)
        return schema.validate_openreward_contract(contract)

    def test_valid_contract_passes(self) -> None:
        result = self._validate()
        self.assertFalse(result["paid_launch_allowed"])
        self.assertTrue(result["stateful_trajectory"])
        self.assertEqual(result["openreward_train"]["revision"], _VALID_REVISION)
        self.assertEqual(result["openreward_train"]["role"], "train")
        self.assertEqual(result["openreward_games_eval"]["role"], "primary_eval")
        self.assertEqual(result["openreward_train"]["task_hashes"][0], _VALID_HASH)
        self.assertEqual(result["openreward_games_eval"]["task_hashes"][0], "c" * 64)

    def test_paid_launch_allowed_must_be_false(self) -> None:
        bad = copy.deepcopy(self.contract)
        bad["paid_launch_allowed"] = True
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "paid_launch_allowed must be False"):
            schema.validate_openreward_contract(bad)

    def test_train_and_eval_must_be_present(self) -> None:
        del self.contract["openreward_train"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "openreward_train"):
            schema.validate_openreward_contract(self.contract)

    def test_revisions_must_be_40_hex(self) -> None:
        self.contract["openreward_train"]["revision"] = "g" * 39
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "openreward_train.revision"):
            schema.validate_openreward_contract(self.contract)

    def test_roles_must_be_explicit(self) -> None:
        self.contract["openreward_train"]["role"] = "primary_eval"
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "openreward_train.role"):
            schema.validate_openreward_contract(self.contract)

    def test_task_hashes_must_be_non_overlapping(self) -> None:
        self.contract["openreward_games_eval"]["task_hashes"] = [_VALID_HASH, "e" * 64]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "non-overlapping"):
            schema.validate_openreward_contract(self.contract)

    def test_task_hashes_must_be_64_hex_and_unique(self) -> None:
        self.contract["openreward_train"]["task_hashes"] = ["g" * 64, "g" * 64]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "must contain unique"):
            schema.validate_openreward_contract(self.contract)

    def test_trajectory_must_be_stateful(self) -> None:
        self.contract["stateful_trajectory"] = False
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "stateful_trajectory must be true"):
            schema.validate_openreward_contract(self.contract)

    def test_artifact_and_state_receipts_are_required(self) -> None:
        del self.contract["artifact_verifier_receipt"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "artifact_verifier_receipt"):
            schema.validate_openreward_contract(self.contract)

        del self.contract["state_verifier_receipt"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "state_verifier_receipt"):
            schema.validate_openreward_contract(self.contract)

    def test_license_container_and_decontamination_hashes_required(self) -> None:
        del self.contract["license"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "license"):
            schema.validate_openreward_contract(self.contract)

        del self.contract["container"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "container"):
            schema.validate_openreward_contract(self.contract)

        del self.contract["decontamination"]
        with self.assertRaisesRegex(schema.OpenrewardSchemaError, "decontamination"):
            schema.validate_openreward_contract(self.contract)


if __name__ == "__main__":
    unittest.main()

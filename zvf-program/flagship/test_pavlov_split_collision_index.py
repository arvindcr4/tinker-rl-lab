from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_split_collision_index import (
    EXPECTED_PRIMARY_EVAL_SUITE_IDS,
    EXPECTED_TRAIN_SUITE_IDS,
    SplitCollisionIndexError,
    build_collision_index,
    main,
    validate_collision_index,
    verify_collision_index,
)
import pytest


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


def suite_manifest(suite_id: str, role: str, index: int) -> dict[str, object]:
    hashes = [digest(f"{suite_id}:task:{index}:0"), digest(f"{suite_id}:task:{index}:1")]
    return {
        "suite_id": suite_id,
        "suite_role": role,
        "revision": format(index + 1, "040x"),
        "receipt_refs": {name: f"receipt://{name}/{suite_id}" for name in RECEIPT_KEYS},
        "task_hashes": hashes,
        "aggregate_sha256": digest("\n".join(hashes)),
        "counts": {role: len(hashes)},
    }


def portfolio_manifests() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    index = 0
    for role, suite_ids in (
        ("train", EXPECTED_TRAIN_SUITE_IDS),
        ("primary_eval", EXPECTED_PRIMARY_EVAL_SUITE_IDS),
    ):
        for suite_id in suite_ids:
            records.append(suite_manifest(suite_id, role, index))
            index += 1
    return records


def xlam_component() -> dict[str, object]:
    train_hashes = [digest("xlam:train")]
    eval_hashes = [digest("xlam:eval")]
    return {
        "component": "xLAM",
        "suite_id": "pavlov_xlam",
        "revision": "b" * 40,
        "receipt_refs": {
            "revision": "hf://Salesforce/xlam-function-calling-60k@" + ("b" * 40),
            "license": "sha256:" + digest("xlam:license"),
            "container": "sha256:" + digest("xlam:container"),
            "decontamination": "sha256:" + digest("xlam:decontamination"),
            "split_manifest": "sha256:" + digest("xlam:split"),
        },
        "task_hashes": {"train": train_hashes, "test": eval_hashes},
        "aggregate_hashes": {
            "train": digest("\n".join(train_hashes)),
            "test": digest("\n".join(eval_hashes)),
        },
    }


class PavlovSplitCollisionIndexTests(unittest.TestCase):
    def test_valid_index_is_deterministic_and_keeps_xlam_outside_26(self) -> None:
        first = build_collision_index(portfolio_manifests(), [xlam_component()])
        second = build_collision_index(
            list(reversed(portfolio_manifests())),
            [copy.deepcopy(xlam_component())],
        )
        assert first == second
        assert first["status"] == "READY"
        assert first["portfolio_counts"] == {"train": 12, "primary_eval": 14}
        assert first["portfolio_roster"]["suite_count"] == 26
        assert first["component_suite_ids"] == ["pavlov_xlam"]
        assert first["collision_count"] == 0
        assert verify_collision_index(first)
        rendered = json.dumps(first, sort_keys=True)
        assert '"prompt"' not in rendered
        assert '"target"' not in rendered

    def test_registry_shape_discovers_component_preflight_without_counting_it(self) -> None:
        registry_shape = {
            "suites": portfolio_manifests(),
            "xlam_component_preflight": xlam_component(),
        }
        result = build_collision_index(registry_shape)
        assert result["status"] == "READY"
        assert result["portfolio_counts"] == {"train": 12, "primary_eval": 14}
        assert result["component_suite_ids"] == ["pavlov_xlam"]

    def test_within_split_duplicate_is_indexed_as_collision(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["task_hashes"][1] = manifests[0]["task_hashes"][0]
        result = build_collision_index(manifests, [xlam_component()])
        assert result["status"] == "BLOCKED"
        assert "within_split" in result["collision_kinds"]
        assert any("duplicate task hash within train split" in error for error in result["errors"])
        assert sum(item["kind"] == "within_split" for item in result["collisions"]) == 1

    def test_within_suite_cross_role_collision_is_not_hidden(self) -> None:
        manifests = portfolio_manifests()
        duplicate = copy.deepcopy(manifests[0])
        duplicate["suite_role"] = "primary_eval"
        duplicate["task_hashes"] = list(manifests[0]["task_hashes"])
        duplicate["aggregate_sha256"] = digest("\n".join(duplicate["task_hashes"]))
        manifests.append(duplicate)
        result = build_collision_index(manifests, [xlam_component()])
        assert "within_suite_cross_role" in result["collision_kinds"]
        assert any("duplicate portfolio suite ID" in error for error in result["errors"])

    def test_cross_role_cross_suite_and_component_collisions_are_all_reported(self) -> None:
        manifests = portfolio_manifests()
        shared = manifests[0]["task_hashes"][0]
        manifests[-1]["task_hashes"][0] = shared
        manifests[-1]["aggregate_sha256"] = digest("\n".join(manifests[-1]["task_hashes"]))
        component = xlam_component()
        component["task_hashes"]["train"][0] = shared
        result = build_collision_index(manifests, [component])
        assert "cross_suite_cross_role" in result["collision_kinds"]
        assert "component_portfolio" in result["collision_kinds"]
        assert result["collision_count"] >= 2

    def test_three_way_collision_emits_every_owner_pair(self) -> None:
        manifests = portfolio_manifests()
        shared = manifests[0]["task_hashes"][0]
        for candidate in manifests[1:3]:
            candidate["task_hashes"][0] = shared
            candidate["aggregate_sha256"] = digest("\n".join(candidate["task_hashes"]))
        result = build_collision_index(manifests, [xlam_component()])
        same_role = [
            collision
            for collision in result["collisions"]
            if collision["kind"] == "cross_suite_same_role"
        ]
        assert len(same_role) == 3
        assert len(result["owner_map"][shared]) == 3

    def test_frozen_roster_duplicate_extra_missing_and_role_mutations_block(self) -> None:
        manifests = portfolio_manifests()
        manifests.pop(0)
        manifests[0]["suite_role"] = "primary_eval"
        manifests.append(copy.deepcopy(manifests[0]))
        manifests[-1]["suite_id"] = "not_a_frozen_suite"
        result = build_collision_index(manifests, [xlam_component()])
        errors = "\n".join(result["errors"])
        assert "missing frozen portfolio suite IDs" in errors
        assert "extra portfolio suite IDs" in errors
        assert "role" in errors

    def test_mutable_revision_and_placeholder_receipts_fail_closed(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["revision"] = "main"
        manifests[1]["receipt_refs"]["license"] = "UNRECORDED"
        result = build_collision_index(manifests, [xlam_component()])
        errors = "\n".join(result["errors"])
        assert "immutable 40-character" in errors
        assert "placeholder or mutable license receipt" in errors

        component = xlam_component()
        component["receipt_refs"]["container"] = "container available"
        component_result = build_collision_index(portfolio_manifests(), [component])
        assert any(
            "placeholder or mutable container receipt" in error
            for error in component_result["errors"]
        )

    def test_raw_content_is_rejected_before_indexing(self) -> None:
        manifests = portfolio_manifests()
        manifests[0]["prompt"] = "secret"
        with pytest.raises(SplitCollisionIndexError, match="metadata-only"):
            build_collision_index(manifests, [xlam_component()])

    def test_index_mutation_is_detected_and_blocked_index_cannot_verify(self) -> None:
        valid = build_collision_index(portfolio_manifests(), [xlam_component()])
        mutated = copy.deepcopy(valid)
        digest_key = next(iter(mutated["owner_map"]))
        mutated["owner_map"][digest_key][0]["position"] = 99
        assert any("metadata drift" in error for error in validate_collision_index(mutated))

        collision = build_collision_index(
            portfolio_manifests()[:-1] + [copy.deepcopy(portfolio_manifests()[0])],
            [xlam_component()],
        )
        with pytest.raises(SplitCollisionIndexError, match="blocked"):
            verify_collision_index(collision)

    def test_local_cli_generate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portfolio_path = root / "portfolio.json"
            component_path = root / "xlam.json"
            index_path = root / "collision-index.json"
            portfolio_path.write_text(json.dumps(portfolio_manifests()), encoding="utf-8")
            component_path.write_text(json.dumps(xlam_component()), encoding="utf-8")
            assert (
                main(
                    [
                        "generate",
                        "--portfolio",
                        str(portfolio_path),
                        "--component",
                        str(component_path),
                        "--out",
                        str(index_path),
                    ]
                )
                == 0
            )
            assert main(["verify", "--index", str(index_path)]) == 0


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flagship import pavlov_eval_receipt_compare as compare


ROOT = Path(__file__).resolve().parents[2]
BASE_RECEIPT = ROOT / "autoresearch/orchestrator-260809-0922/base_eval_100.json"


def _hex(char: str, length: int) -> str:
    return char * length


def _receipt(*, source_kind: str, successes: set[int], adapter: str | None = None) -> dict[str, object]:
    base_revision = _hex("a", 40)
    tokenizer_revision = _hex("b", 40)
    adapter_revision = adapter
    rows = []
    for index in range(100):
        rows.append(
            {
                "example_id": f"xlam-{index:03d}",
                "index": index,
                "prompt_sha256": _hex("c", 64),
                "target_sha256": _hex("d", 64),
                "score": 1.0 if index in successes else 0.0,
                "prompt_tokens": 100,
                "sample_tokens": 20,
            }
        )
    provenance = {
        "dataset_revision": _hex("e", 40),
        "split_manifest_sha256": _hex("f", 64),
        "task_id_sha256": _hex("1", 64),
        "verifier_sha256": _hex("2", 64),
        "base_model_revision": base_revision,
        "tokenizer_revision": tokenizer_revision,
        "decontamination_sha256": _hex("3", 64),
        "decontamination_receipt": "decontamination-receipt-v1",
        "container_digest": f"sha256:{_hex('4', 64)}",
        "runtime_digest": f"sha256:{_hex('5', 64)}",
        "license_id": "spdx:Apache-2.0",
        "license_receipt": "license-receipt-v1",
        "adapter_revision": adapter_revision,
    }
    receipt: dict[str, object] = {
        "schema_version": "pavlov-xlam-eval-v1",
        "source_kind": source_kind,
        "evaluated_path": "Qwen/Qwen3.6-35B-A3B" if source_kind == "base_model" else "org/xlam-adapter",
        "tokenizer_model": "Qwen/Qwen3.6-35B-A3B",
        "dataset_id": "Salesforce/xlam-function-calling-60k",
        "dataset_split": "frozen-xlam-100",
        "seed": 809,
        "examples": 100,
        "mean_strict_reward": len(successes) / 100,
        "perfect_call_rate": len(successes) / 100,
        "rows": rows,
        "provenance": provenance,
        "uncertainty": compare.wilson_uncertainty(len(successes), 100),
        "suite_id": "xlam_component",
        "suite_role": "component",
        "domains": ["tool_use"],
        "portfolio": {
            "portfolio_id": "pavlov-primary-eval-14-suite-v1",
            "portfolio_role": "primary_eval",
            "suite_count": 14,
            "suite_ids": list(compare.PAVLOV_PRIMARY_EVAL_SUITE_IDS),
            "suites": compare._primary_eval_manifest(),
            "component_only": True,
            "component_suite_id": "xlam_component",
            "component_domains": ["tool_use"],
            "coverage_claim": "xLAM component only; no full-portfolio claim",
        },
        "wandb": {
            "entity": "entity",
            "project": "project",
            "group": "pavlov-xlam-evaluation",
            "id": "wandb-base" if source_kind == "base_model" else "wandb-trained",
            "url": "https://wandb.ai/entity/project/runs/run",
        },
        "tinker": {
            "run_id": "tinker-base" if source_kind == "base_model" else "tinker-trained",
            "model_id": "Qwen/Qwen3.6-35B-A3B",
        },
        "hf": {
            "repo_id": "Qwen/Qwen3.6-35B-A3B" if source_kind == "base_model" else "org/xlam-adapter",
            "revision": base_revision if source_kind == "base_model" else adapter,
            "base_model_revision": base_revision,
        },
    }
    return receipt


class PavlovEvalReceiptCompareTests(unittest.TestCase):
    def test_valid_paired_component_comparison_passes(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["base"]["successes"], 7)
        self.assertEqual(result["trained"]["successes"], 10)
        self.assertEqual(result["paired"]["trained_only_successes"], 3)
        self.assertEqual(result["paired"]["base_only_successes"], 0)
        self.assertEqual(result["claim_boundary"]["component_only"], True)
        self.assertEqual(result["claim_boundary"]["domains"], ["tool_use"])

    def test_frozen_live_base_is_counted_but_blocked_without_new_provenance_contract(self) -> None:
        base = json.loads(BASE_RECEIPT.read_text(encoding="utf-8"))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["base"]["examples"], 100)
        self.assertEqual(result["base"]["successes"], 7)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("base.provenance" in error for error in result["errors"]))
        self.assertTrue(any("base.uncertainty" in error for error in result["errors"]))

    def test_example_id_mismatch_blocks_pairing(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["rows"][0]["example_id"] = "different-example"
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("paired example IDs/order differ", result["errors"])

    def test_prompt_or_target_identity_mismatch_blocks_pairing(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["rows"][4]["target_sha256"] = _hex("5", 64)
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("paired prompt/target identities differ", result["errors"])

    def test_dataset_split_task_and_verifier_mismatches_block(self) -> None:
        for key in ("dataset_revision", "split_manifest_sha256", "task_id_sha256", "verifier_sha256"):
            with self.subTest(key=key):
                base = _receipt(source_kind="base_model", successes=set(range(7)))
                trained = _receipt(
                    source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
                )
                trained["provenance"][key] = _hex("6", 64 if key.endswith("sha256") else 40)
                result = compare.compare_receipts(base, trained)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn(f"paired provenance mismatch: {key}", result["errors"])

    def test_uncertainty_output_must_be_exact(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["uncertainty"]["wilson_high"] = 0.123
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("exact deterministic Wilson" in error for error in result["errors"]))

    def test_missing_wandb_tinker_or_hf_provenance_blocks(self) -> None:
        for field in ("wandb", "tinker", "hf"):
            with self.subTest(field=field):
                base = _receipt(source_kind="base_model", successes=set(range(7)))
                trained = _receipt(
                    source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
                )
                del trained[field]
                result = compare.compare_receipts(base, trained)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertTrue(any(f"trained.{field}" in error for error in result["errors"]))

    def test_placeholder_immutable_provenance_blocks(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["provenance"]["verifier_sha256"] = "placeholder"
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("trained.provenance.verifier_sha256" in error for error in result["errors"]))

    def test_wandb_project_mismatch_blocks_even_with_valid_run_ids(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["wandb"]["project"] = "different-project"
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("paired W&B provenance mismatch: project", result["errors"])

    def test_wandb_and_tinker_run_identity_must_not_be_reused(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["wandb"]["id"] = base["wandb"]["id"]
        trained["tinker"]["run_id"] = base["tinker"]["run_id"]
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("paired W&B runs must have distinct run IDs", result["errors"])
        self.assertIn("paired Tinker runs must have distinct run IDs", result["errors"])

    def test_non_finite_metric_blocks_without_hashing_failure(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["mean_strict_reward"] = float("nan")
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("mean_strict_reward must be finite" in error for error in result["errors"]))

    def test_component_boundary_rejects_broad_claims_and_unknown_domains(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["domains"] = ["tool_use", "code"]
        trained["portfolio"]["suite_ids"][0] = "unknown_suite"
        trained["portfolio"]["coverage_claim"] = "covers all Pavlov domains"
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("domains must be exactly" in error for error in result["errors"]))
        self.assertTrue(any("suite_ids must be the exact" in error for error in result["errors"]))
        self.assertTrue(any("claim exceeds" in error for error in result["errors"]))

    def test_trained_adapter_revision_must_be_distinct_from_base(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path",
            successes=set(range(10)),
            adapter=_hex("a", 40),
        )
        result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("trained adapter revision" in error for error in result["errors"]))

    def test_require_comparable_raises_on_blocked_pair(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        trained["provenance"]["verifier_sha256"] = _hex("6", 64)
        with self.assertRaises(compare.ReceiptComparisonError):
            compare.require_comparable(base, trained)

    def test_cli_is_local_and_writes_blocked_receipt(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.json"
            trained_path = root / "trained.json"
            output_path = root / "comparison.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            trained_path.write_text(json.dumps(trained), encoding="utf-8")
            self.assertEqual(
                compare.main([str(base_path), str(trained_path), "--out", str(output_path)]),
                0,
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "PASS")

    def test_comparator_does_not_open_network_sockets(self) -> None:
        base = _receipt(source_kind="base_model", successes=set(range(7)))
        trained = _receipt(
            source_kind="sampler_path", successes=set(range(10)), adapter=_hex("4", 40)
        )
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            result = compare.compare_receipts(base, trained)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

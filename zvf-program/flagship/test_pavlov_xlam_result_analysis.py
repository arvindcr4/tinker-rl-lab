from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from flagship.pavlov_xlam_result_analysis import (
    XlamReceiptValidationError,
    analyze_xlam_receipts,
    main,
    validate_xlam_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RECEIPT_PATH = REPO_ROOT / "autoresearch/orchestrator-260809-0922/base_eval_100.json"


def _rows(scores: list[float], response_prefix: str) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "prompt_sha256": f"prompt-{index}",
            "target_sha256": f"target-{index}",
            "response_sha256": f"{response_prefix}-response-{index}",
            "score": score,
        }
        for index, score in enumerate(scores)
    ]


def _receipt(role: str, scores: list[float]) -> dict[str, object]:
    is_base = role == "base"
    provenance: dict[str, object] = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "model_revision": "model-revision-1",
        "tokenizer_revision": "tokenizer-revision-1",
        "dataset_id": "Salesforce/xlam-function-calling-60k",
        "dataset_revision": "dataset-revision-1",
        "split_manifest_sha256": "split-manifest-1",
        "task_id_manifest_sha256": "task-manifest-1",
        "verifier_revision": "strict-reward-revision-1",
        "container_digest": "sha256:container-1",
        "decontamination_receipt": "decontamination-receipt-1",
        "sampling": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_prompt_tokens": 1200,
            "max_response_tokens": 128,
            "num_samples": 1,
            "sampling_seed": 809,
        },
        "wandb": {
            "run_id": f"wandb-{role}-1",
            "url": f"https://wandb.example/{role}-1",
            "mode": "online",
        },
        "hf": {
            "repo": f"org/pavlov-{role}-1",
            "commit": f"hf-commit-{role}-1",
            "visibility": "private",
        },
        "tinker": {
            "run_id": f"tinker-{role}-1",
            "status": "completed",
        },
    }
    return {
        "schema_version": "pavlov-xlam-eval-v1",
        "created_at": "2026-08-09T00:00:00+00:00",
        "source_kind": "base_model" if is_base else "sampler_path",
        "evaluated_path": "Qwen/Qwen3.6-35B-A3B" if is_base else "org/pavlov-trained-1",
        "tokenizer_model": "Qwen/Qwen3.6-35B-A3B",
        "seed": 809,
        "examples": len(scores),
        "mean_strict_reward": sum(scores) / len(scores),
        "perfect_call_rate": sum(score == 1.0 for score in scores) / len(scores),
        "rows": _rows(scores, role),
        "provenance": provenance,
        "claim_scope": "xlam_component_only",
    }


def _paired_receipts() -> tuple[dict[str, object], dict[str, object]]:
    base_scores = [1.0 if index < 7 else 0.0 for index in range(100)]
    trained_scores = base_scores[:]
    trained_scores[0] = 0.0
    trained_scores[7] = 1.0
    trained_scores[8] = 1.0
    trained_scores[9] = 1.0
    return _receipt("base", base_scores), _receipt("trained", trained_scores)


class CompleteAnalysisTests(unittest.TestCase):
    def test_reports_paired_xlam_metrics_and_records_bootstrap_metadata(self) -> None:
        base, trained = _paired_receipts()
        report = analyze_xlam_receipts(base, trained, bootstrap_resamples=257, bootstrap_seed=123)

        self.assertEqual(report["status"], "admissible_xlam_component")
        self.assertEqual(report["analysis_scope"], "xlam_component_only")
        self.assertFalse(report["portfolio_claim_permitted"])
        self.assertFalse(report["company_claim_permitted"])
        comparison = report["comparison"]
        self.assertEqual(comparison["base"]["perfect_calls"], 7)
        self.assertEqual(comparison["trained"]["perfect_calls"], 9)
        paired = comparison["paired"]
        self.assertEqual(paired["base_fail_trained_success"], 3)
        self.assertEqual(paired["base_success_trained_fail"], 1)
        self.assertAlmostEqual(paired["risk_difference_trained_minus_base"], 0.02)
        self.assertAlmostEqual(paired["perfect_call_rate_difference"], 0.02)
        self.assertAlmostEqual(paired["exact_mcnemar_two_sided_p"], 0.625)
        self.assertEqual(
            paired["mean_strict_reward_bootstrap"]["seed"],
            123,
        )
        self.assertEqual(
            paired["mean_strict_reward_bootstrap"]["resamples"],
            257,
        )
        self.assertTrue(comparison["improvement_vs_7_of_100"]["point_estimate_exceeds_base"])

    def test_pairs_by_identity_even_when_trained_rows_are_reordered(self) -> None:
        base, trained = _paired_receipts()
        trained["rows"] = list(reversed(trained["rows"]))
        report = analyze_xlam_receipts(base, trained, bootstrap_resamples=31, bootstrap_seed=9)
        self.assertEqual(report["status"], "admissible_xlam_component")
        self.assertEqual(report["comparison"]["paired"]["base_fail_trained_success"], 3)

    def test_validate_helper_returns_normalized_receipt_summary(self) -> None:
        base, _ = _paired_receipts()
        summary = validate_xlam_receipt(base, role="base")
        self.assertEqual(summary["perfect_calls"], 7)
        self.assertEqual(summary["examples"], 100)
        self.assertTrue(summary["tracking_receipts_present"])


class FailClosedTests(unittest.TestCase):
    def test_unpaired_rows_produce_no_partial_comparison(self) -> None:
        base, trained = _paired_receipts()
        trained["rows"][99]["target_sha256"] = "different-target"
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["comparison"])
        self.assertTrue(any("unpaired rows" in item for item in report["diagnostics"]))

    def test_duplicate_identity_is_rejected(self) -> None:
        base, trained = _paired_receipts()
        trained["rows"][1]["index"] = trained["rows"][0]["index"]
        trained["rows"][1]["prompt_sha256"] = trained["rows"][0]["prompt_sha256"]
        trained["rows"][1]["target_sha256"] = trained["rows"][0]["target_sha256"]
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any("duplicate paired example identity" in item for item in report["diagnostics"])
        )

    def test_revision_drift_is_rejected_before_statistics(self) -> None:
        base, trained = _paired_receipts()
        trained["provenance"]["dataset_revision"] = "dataset-revision-2"
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("dataset_revision differs" in item for item in report["diagnostics"]))
        self.assertIsNone(report["comparison"])

    def test_seed_and_sampling_drift_are_rejected(self) -> None:
        for mutation, expected in (
            (lambda receipt: receipt.update(seed=810), "seed differs"),
            (
                lambda receipt: receipt["provenance"]["sampling"].update(temperature=0.2),
                "sampling field temperature differs",
            ),
        ):
            with self.subTest(expected=expected):
                base, trained = _paired_receipts()
                mutation(trained)
                report = analyze_xlam_receipts(base, trained)
                self.assertEqual(report["status"], "blocked")
                self.assertTrue(any(expected in item for item in report["diagnostics"]))

    def test_missing_tracking_receipts_fail_closed(self) -> None:
        for missing in ("wandb", "hf", "tinker"):
            with self.subTest(missing=missing):
                base, trained = _paired_receipts()
                del trained["provenance"][missing]
                report = analyze_xlam_receipts(base, trained)
                self.assertEqual(report["status"], "blocked")
                self.assertIsNone(report["comparison"])
                expected = {
                    "wandb": "missing W&B receipt",
                    "hf": "missing Hugging Face receipt",
                    "tinker": "missing Tinker receipt",
                }[missing]
                self.assertTrue(any(expected in item for item in report["diagnostics"]))

    def test_public_trained_hf_checkpoint_is_rejected(self) -> None:
        base, trained = _paired_receipts()
        trained["provenance"]["hf"]["visibility"] = "public"
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("must be private" in item for item in report["diagnostics"]))

    def test_backfilled_wandb_run_is_not_admissible(self) -> None:
        base, trained = _paired_receipts()
        trained["provenance"]["wandb"]["mode"] = "backfilled_after_interruption"
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("online mode" in item for item in report["diagnostics"]))

    def test_heldout_and_portfolio_claim_scopes_are_rejected(self) -> None:
        for scope in ("heldout", "xlam_heldout", "portfolio"):
            with self.subTest(scope=scope):
                base, trained = _paired_receipts()
                trained["claim_scope"] = scope
                report = analyze_xlam_receipts(base, trained)
                self.assertEqual(report["status"], "blocked")
                self.assertIsNone(report["comparison"])

    def test_heldout_claim_text_is_rejected_even_when_it_mentions_xlam(self) -> None:
        base, trained = _paired_receipts()
        trained["claim"] = "xLAM held-out improvement"
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("held-out claim text" in item for item in report["diagnostics"]))

    def test_summary_mismatch_is_rejected(self) -> None:
        base, trained = _paired_receipts()
        trained["perfect_call_rate"] = 0.08
        report = analyze_xlam_receipts(base, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any("perfect_call_rate does not match rows" in item for item in report["diagnostics"])
        )

    def test_actual_observed_base_receipt_is_blocked_without_tracking_provenance(self) -> None:
        _, trained = _paired_receipts()
        report = analyze_xlam_receipts(BASE_RECEIPT_PATH, trained)
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["comparison"])
        self.assertTrue(
            any(
                "missing provenance field" in item or "could not read receipt" in item
                for item in report["diagnostics"]
            )
        )

    def test_missing_file_and_strict_validator_are_fail_closed(self) -> None:
        _, trained = _paired_receipts()
        report = analyze_xlam_receipts("/tmp/does-not-exist-pavlov-base.json", trained)
        self.assertEqual(report["status"], "blocked")
        with self.assertRaises(XlamReceiptValidationError):
            validate_xlam_receipt({}, role="base")


class OfflineCliTests(unittest.TestCase):
    def test_cli_reads_local_json_only_and_returns_success_for_complete_pair(self) -> None:
        base, trained = _paired_receipts()
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.json"
            trained_path = Path(directory) / "trained.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            trained_path.write_text(json.dumps(trained), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        str(base_path),
                        str(trained_path),
                        "--bootstrap-resamples",
                        "17",
                        "--bootstrap-seed",
                        "4",
                    ]
                )
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "admissible_xlam_component")
        self.assertEqual(
            report["comparison"]["paired"]["mean_strict_reward_bootstrap"]["seed"],
            4,
        )


if __name__ == "__main__":
    unittest.main()

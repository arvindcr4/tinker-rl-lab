from __future__ import annotations

import copy
import unittest

from pilot.analysis import AnalysisContractError, confirmatory_power_gate, screening_gate


def screening_records() -> list[dict]:
    records = []
    corpus = {("balanced_equal_length", seed): f"balanced-{seed}" for seed in (11, 23, 37)}
    corpus.update({("filtered_variable_length", seed): f"filtered-{seed}" for seed in (11, 23, 37)})
    for regime in ("balanced_equal_length", "filtered_variable_length"):
        for seed in (11, 23, 37):
            for condition in ("intended_full", "native_trl", "epsilon_only", "reduction_only"):
                if regime == "balanced_equal_length":
                    intended_native_cosine = 0.9995
                    intended_native_l2 = 0.005
                    selected_cosine = 0.9995 if condition == "epsilon_only" else 1.0
                    selected_l2 = 0.005 if condition == "epsilon_only" else 0.0
                    native_curve = [0.40, 0.44, 0.48, 0.52, 0.56, 0.60]
                    intended_curve = list(native_curve)
                else:
                    intended_native_cosine = 0.95
                    intended_native_l2 = 0.10
                    selected_cosine = 0.96 if condition == "reduction_only" else 0.9995
                    selected_l2 = 0.08 if condition == "reduction_only" else 0.005
                    native_curve = [0.40, 0.43, 0.46, 0.49, 0.52, 0.55]
                    gain = {11: 0.02, 23: 0.025, 37: 0.03}[seed]
                    intended_curve = [
                        native_curve[0],
                        native_curve[1] + gain,
                        native_curve[2] + gain,
                        native_curve[3] + gain,
                        native_curve[4],
                        native_curve[5],
                    ]
                curve = intended_curve if condition == "intended_full" else native_curve
                receipts = [
                    {
                        "step": step,
                        "gradient_relation": "nonzero",
                        "gradient_cosine": intended_native_cosine,
                        "gradient_relative_l2": intended_native_l2,
                        "selected_vs_intended_relation": "nonzero",
                        "selected_vs_intended_cosine": selected_cosine,
                        "selected_vs_intended_relative_l2": selected_l2,
                    }
                    for step in range(1, 101)
                ]
                evaluations = [
                    {"step": step, "accuracy": value}
                    for step, value in zip((0, 20, 40, 60, 80, 100), curve, strict=True)
                ]
                full = {
                    "schema_version": "flagship-pilot-unit-v1",
                    "condition": condition,
                    "regime": regime,
                    "seed": seed,
                    "corpus_fingerprint": corpus[(regime, seed)],
                    "manifest": {"gradient_receipts": receipts},
                    "evaluations": evaluations,
                    "token_flop_ledger": {"charged_generated_tokens": 1000},
                }
                records.append(
                    {
                        "schema_version": "flagship-pilot-acceptance-v1",
                        "status": "accepted",
                        "full_record": full,
                    }
                )
    return records


class AnalysisTests(unittest.TestCase):
    def test_prespecified_screening_gate_passes_complete_positive_fixture(self) -> None:
        report = screening_gate(screening_records())
        self.assertEqual(report["verdict"], "GO")
        self.assertTrue(report["mechanism_pass"])
        self.assertTrue(report["causal_attribution_pass"])
        self.assertTrue(report["learning"]["pass"])
        self.assertTrue(report["matched_compute_pass"])

    def test_gate_kills_failed_mechanism_without_pseudoreplication(self) -> None:
        records = screening_records()
        target = next(
            item["full_record"]
            for item in records
            if item["full_record"]["condition"] == "intended_full"
            and item["full_record"]["regime"] == "filtered_variable_length"
            and item["full_record"]["seed"] == 23
        )
        for receipt in target["manifest"]["gradient_receipts"]:
            receipt["gradient_cosine"] = 0.9999
            receipt["gradient_relative_l2"] = 0.001
        report = screening_gate(records)
        self.assertEqual(report["verdict"], "KILL")
        self.assertFalse(report["mechanism"]["filtered_variable_length"]["23"]["pass"])

    def test_joint_zero_is_equivalent_and_one_sided_zero_is_maximal_divergence(self) -> None:
        records = screening_records()
        balanced = next(
            item["full_record"]
            for item in records
            if item["full_record"]["condition"] == "intended_full"
            and item["full_record"]["regime"] == "balanced_equal_length"
            and item["full_record"]["seed"] == 11
        )
        for receipt in balanced["manifest"]["gradient_receipts"]:
            receipt.update(
                {
                    "gradient_relation": "joint_zero",
                    "gradient_cosine": None,
                    "gradient_relative_l2": None,
                }
            )
        filtered = next(
            item["full_record"]
            for item in records
            if item["full_record"]["condition"] == "intended_full"
            and item["full_record"]["regime"] == "filtered_variable_length"
            and item["full_record"]["seed"] == 11
        )
        for receipt in filtered["manifest"]["gradient_receipts"][:20]:
            receipt.update(
                {
                    "gradient_relation": "intended_zero",
                    "gradient_cosine": None,
                    "gradient_relative_l2": None,
                }
            )
        report = screening_gate(records)
        self.assertEqual(
            report["mechanism"]["balanced_equal_length"]["11"]["gradient_relation_counts"],
            {"joint_zero": 100},
        )
        self.assertTrue(report["mechanism"]["balanced_equal_length"]["11"]["pass"])
        self.assertEqual(
            report["mechanism"]["filtered_variable_length"]["11"]["gradient_relation_counts"][
                "intended_zero"
            ],
            20,
        )
        self.assertGreater(report["causal_attribution"]["11"]["native_relation_effect"], 0.2)

    def test_gate_rejects_incomplete_or_duplicate_matrix(self) -> None:
        records = screening_records()
        with self.assertRaisesRegex(AnalysisContractError, "matrix mismatch"):
            screening_gate(records[:-1])
        duplicate = copy.deepcopy(records)
        duplicate.append(copy.deepcopy(records[0]))
        with self.assertRaisesRegex(AnalysisContractError, "duplicate"):
            screening_gate(duplicate)

    def test_power_gate_is_deterministic_and_stops_weak_design(self) -> None:
        report = screening_gate(screening_records())
        first = confirmatory_power_gate(report, draws=5_000)
        second = confirmatory_power_gate(report, draws=5_000)
        self.assertEqual(first, second)
        weak = copy.deepcopy(report)
        weak["learning"]["auc_differences"]["filtered_variable_length"] = [
            0.0001,
            -0.0001,
            0.0002,
        ]
        weak["learning"]["auc_differences"]["balanced_equal_length"] = [0.0, 0.0, 0.0]
        stopped = confirmatory_power_gate(weak, draws=5_000)
        self.assertEqual(stopped["verdict"], "STOP_UNDERPOWERED")

    def test_power_gate_cannot_run_after_screening_kill(self) -> None:
        report = screening_gate(screening_records())
        report["verdict"] = "KILL"
        with self.assertRaisesRegex(AnalysisContractError, "after a KILL"):
            confirmatory_power_gate(report, draws=100)


if __name__ == "__main__":
    unittest.main()

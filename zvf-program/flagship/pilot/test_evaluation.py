from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pilot.evaluation import (
    EvaluationContractError,
    atomic_jsonl,
    evidence_row,
    validate_evidence,
)


class EvaluationEvidenceTests(unittest.TestCase):
    def test_gsm8k_evidence_is_contiguous_hashed_and_recomputable(self) -> None:
        questions = ["one", "two", "three"]
        answers = ["#### 1", "#### 2", "#### 3"]
        source_indices = [10, 11, 12]
        rows = [
            evidence_row(
                index=index,
                source_index=source_indices[index],
                regime="balanced_equal_length",
                question=questions[index],
                answer=answers[index],
                completion_text=f"work\n#### {index + 1}",
                generated_tokens=5 + index,
            )
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.jsonl"
            atomic_jsonl(path, rows)
            summary = validate_evidence(
                path,
                regime="balanced_equal_length",
                questions=questions,
                answers=answers,
                source_indices=source_indices,
            )
            self.assertEqual(summary["correct"], 3)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["unique_row_hashes"], 3)

    def test_math_evidence_uses_strict_boxed_lower_bound(self) -> None:
        row = evidence_row(
            index=0,
            source_index=0,
            regime="filtered_variable_length",
            question="half",
            answer=r"\\frac{1}{2}",
            completion_text=r"answer \\boxed{0.5}",
            generated_tokens=4,
        )
        self.assertEqual(row["correct"], 0)

    def test_tampered_or_noncontiguous_rows_fail_closed(self) -> None:
        questions = ["one", "two"]
        answers = ["#### 1", "#### 2"]
        source_indices = [0, 1]
        rows = [
            evidence_row(
                index=index,
                source_index=index,
                regime="balanced_equal_length",
                question=questions[index],
                answer=answers[index],
                completion_text=f"#### {index + 1}",
                generated_tokens=2,
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.jsonl"
            bad = [dict(row) for row in rows]
            bad[1]["index"] = 3
            atomic_jsonl(path, bad)
            with self.assertRaisesRegex(EvaluationContractError, "not contiguous"):
                validate_evidence(
                    path,
                    regime="balanced_equal_length",
                    questions=questions,
                    answers=answers,
                    source_indices=source_indices,
                )
            tampered = [dict(row) for row in rows]
            tampered[0]["completion_text"] = "#### 9"
            path.write_text("\n".join(json.dumps(row) for row in tampered) + "\n")
            with self.assertRaisesRegex(EvaluationContractError, "completion hash mismatch"):
                validate_evidence(
                    path,
                    regime="balanced_equal_length",
                    questions=questions,
                    answers=answers,
                    source_indices=source_indices,
                )


if __name__ == "__main__":
    unittest.main()

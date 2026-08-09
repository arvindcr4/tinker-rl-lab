#!/usr/bin/env python3
"""Unit tests for the E14 FrontierMath public-sample parser and fail-closed receipt.

Two classes of test:

1. Synthetic tests that pin the schema, parser, hashing, and the score-refusal
   guards. These run anywhere.
2. Corpus tests that assert the *observed* facts about Epoch AI's 150 public
   representative sample transcripts. They skip if the corpus is absent.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import e14_frontiermath_public_samples as mod
except ImportError:  # pragma: no cover - direct execution fallback
    import e14_frontiermath_public_samples as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "outputs/e14_frontiermath/public_samples/sample_question_transcripts"
ARCHIVE = REPO_ROOT / "outputs/e14_frontiermath/frontiermath_public_samples.zip"

PREAMBLE = "Solve it.\n\n"
STATEMENT = "Compute something hard."
PROMPT = (
    PREAMBLE
    + mod.PROBLEM_STATEMENT_MARKER
    + "\n"
    + STATEMENT
    + "\n"
    + mod.RETURN_TYPE_MARKER
    + " Python integer"
)

VALID_TRANSCRIPT = [
    {"role": "user", "content": PROMPT},
    {"role": "assistant", "content": "Let me experiment.\n```python\nprint(1)\n```"},
    {"role": "user", "content": "Results from executing code block 1:\nstdout content: 1\n"},
    {
        "role": "assistant",
        "content": (
            "Done.\n```python\nimport pickle\n"
            f"{mod.FINAL_ANSWER_MARKER}\n"
            "pickle.dump(7, open('final_answer.p','wb'))\n```"
        ),
    },
]


def write_transcript(directory: Path, filename: str, messages: list[dict]) -> Path:
    path = directory / filename
    path.write_text("\n".join(json.dumps(m) for m in messages) + "\n", encoding="utf-8")
    return path


class TestFilenameParsing(unittest.TestCase):
    def test_parses_model_problem_run(self) -> None:
        ident = mod.parse_transcript_filename("gpt-4o-2024-08-06_CWA2_run-2.jsonl")
        self.assertEqual(ident.model, "gpt-4o-2024-08-06")
        self.assertEqual(ident.problem_token, "CWA2")
        self.assertEqual(ident.run_index, 2)

    def test_handles_model_names_containing_underscores_free_hyphens(self) -> None:
        ident = mod.parse_transcript_filename("claude-3-5-sonnet-20241022_CWD31_run-5.jsonl")
        self.assertEqual(ident.model, "claude-3-5-sonnet-20241022")
        self.assertEqual(ident.problem_token, "CWD31")

    def test_rejects_unconventional_filename(self) -> None:
        with self.assertRaises(mod.TranscriptSchemaError):
            mod.parse_transcript_filename("not-a-transcript.txt")


class TestSchemaValidation(unittest.TestCase):
    def test_valid_transcript_has_no_errors(self) -> None:
        self.assertEqual(mod.validate_transcript(VALID_TRANSCRIPT), [])

    def test_empty_transcript_is_invalid(self) -> None:
        self.assertEqual(mod.validate_transcript([]), ["transcript is empty"])

    def test_extra_key_is_rejected(self) -> None:
        bad = [dict(VALID_TRANSCRIPT[0], verdict="correct"), VALID_TRANSCRIPT[1]]
        errors = mod.validate_transcript(bad)
        self.assertTrue(any("unexpected key" in e for e in errors), errors)

    def test_missing_content_is_rejected(self) -> None:
        errors = mod.validate_transcript([{"role": "user"}])
        self.assertTrue(any("missing key" in e for e in errors), errors)

    def test_non_alternating_roles_rejected(self) -> None:
        bad = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        errors = mod.validate_transcript(bad)
        self.assertTrue(any("must alternate" in e for e in errors), errors)

    def test_must_start_user_and_end_assistant(self) -> None:
        errors = mod.validate_transcript([{"role": "assistant", "content": "hi"}])
        self.assertIn("first message must be the user task prompt", errors)

    def test_bad_role_rejected(self) -> None:
        errors = mod.validate_transcript([{"role": "system", "content": "x"}])
        self.assertTrue(any("role must be one of" in e for e in errors), errors)

    def test_non_string_content_rejected(self) -> None:
        errors = mod.validate_transcript([{"role": "user", "content": [{"text": "x"}]}])
        self.assertTrue(any("content must be str" in e for e in errors), errors)


class TestHarnessTurnClassification(unittest.TestCase):
    def test_first_turn_is_task_prompt(self) -> None:
        self.assertEqual(mod.classify_harness_turn(PROMPT, is_first=True), mod.TURN_TASK_PROMPT)

    def test_code_result(self) -> None:
        self.assertEqual(
            mod.classify_harness_turn("Results from executing code block 1:\nstdout"),
            mod.TURN_CODE_RESULT,
        )

    def test_final_answer_failed_timeout(self) -> None:
        self.assertEqual(
            mod.classify_harness_turn("Final answer failed: timed out\nKeep working."),
            mod.TURN_FINAL_ANSWER_FAILED_TIMEOUT,
        )

    def test_final_answer_failed_stderr(self) -> None:
        self.assertEqual(
            mod.classify_harness_turn("Final answer failed: error in stderr: Traceback"),
            mod.TURN_FINAL_ANSWER_FAILED_STDERR,
        )

    def test_unknown_turn(self) -> None:
        self.assertEqual(
            mod.classify_harness_turn("Your answer was correct. Score: 1.0"),
            mod.TURN_UNKNOWN,
        )


class TestAnswerExtraction(unittest.TestCase):
    def test_finds_marker_block(self) -> None:
        blocks = mod.extract_final_answer_blocks(VALID_TRANSCRIPT)
        self.assertEqual(len(blocks), 1)
        self.assertIn(mod.FINAL_ANSWER_PICKLE, blocks[0])

    def test_ignores_non_marker_blocks(self) -> None:
        msgs = [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
        ]
        self.assertEqual(mod.extract_final_answer_blocks(msgs), [])

    def test_ignores_marker_in_user_turn(self) -> None:
        msgs = [
            {"role": "user", "content": f"```python\n{mod.FINAL_ANSWER_MARKER}\n```"},
        ]
        self.assertEqual(mod.extract_final_answer_blocks(msgs), [])

    def test_splits_task_prompt(self) -> None:
        preamble, statement, return_type = mod.split_task_prompt(PROMPT)
        self.assertEqual(preamble, PREAMBLE)
        self.assertEqual(statement, STATEMENT)
        self.assertEqual(return_type, "Python integer")


class TestGroundTruthDetection(unittest.TestCase):
    def test_clean_transcript_has_no_ground_truth(self) -> None:
        self.assertFalse(mod.detect_ground_truth(VALID_TRANSCRIPT))
        self.assertFalse(mod.detect_grader_verdict(VALID_TRANSCRIPT))

    def test_out_of_schema_field_fails_loud(self) -> None:
        msgs = [dict(VALID_TRANSCRIPT[0], ground_truth=42)]
        self.assertTrue(mod.detect_ground_truth(msgs))

    def test_unrecognised_harness_turn_fails_loud(self) -> None:
        msgs = list(VALID_TRANSCRIPT) + [
            {"role": "user", "content": "Grader says: incorrect, expected 12."}
        ]
        self.assertTrue(mod.detect_ground_truth(msgs))


class TestHashing(unittest.TestCase):
    def test_corpus_digest_is_order_independent(self) -> None:
        a = {"b.jsonl": "22", "a.jsonl": "11"}
        b = {"a.jsonl": "11", "b.jsonl": "22"}
        self.assertEqual(mod.corpus_digest(a), mod.corpus_digest(b))

    def test_corpus_digest_changes_with_content(self) -> None:
        a = {"a.jsonl": "11"}
        b = {"a.jsonl": "12"}
        self.assertNotEqual(mod.corpus_digest(a), mod.corpus_digest(b))

    def test_corpus_digest_changes_with_membership(self) -> None:
        a = {"a.jsonl": "11"}
        b = {"a.jsonl": "11", "b.jsonl": "22"}
        self.assertNotEqual(mod.corpus_digest(a), mod.corpus_digest(b))

    def test_sha256_file_matches_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(b"hello")
            self.assertEqual(mod.sha256_file(path), mod.sha256_bytes(b"hello"))


class TestFailClosedGuards(unittest.TestCase):
    def test_compute_score_always_raises(self) -> None:
        with self.assertRaises(mod.ScoreProhibited):
            mod.compute_frontiermath_score()
        with self.assertRaises(mod.ScoreProhibited):
            mod.compute_frontiermath_score(correct=5, total=5)

    def test_receipt_is_blocked_with_null_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_transcript(d, "m-1_ALL3_run-1.jsonl", VALID_TRANSCRIPT)
            manifest = mod.build_public_sample_manifest(d)
            receipt = mod.build_blocked_receipt(
                manifest, recorded_at="2026-08-09", checkout=str(d), commit="deadbeef"
            )
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertIsNone(receipt["score"])
        self.assertFalse(receipt["is_model_score"])
        self.assertTrue(receipt["score_refusal"]["refused"])
        self.assertEqual(receipt["public_sample_asset"]["artifact_label"], mod.ARTIFACT_LABEL)

    def test_assert_rejects_injected_score(self) -> None:
        receipt = {
            "status": "BLOCKED",
            "score": 0.42,
            "experiment": {},
            "public_sample_asset": {"artifact_label": mod.ARTIFACT_LABEL},
        }
        with self.assertRaises(mod.ScoreProhibited):
            mod.assert_receipt_emits_no_score(receipt)

    def test_assert_rejects_non_blocked_status(self) -> None:
        receipt = {
            "status": "PARTIAL",
            "score": None,
            "experiment": {},
            "public_sample_asset": {"artifact_label": mod.ARTIFACT_LABEL},
        }
        with self.assertRaises(mod.ScoreProhibited):
            mod.assert_receipt_emits_no_score(receipt)

    def test_assert_rejects_benchmark_substitution(self) -> None:
        receipt = {
            "status": "BLOCKED",
            "score": None,
            "experiment": {"measured_metrics": None, "related_benchmark_substitution": True},
            "public_sample_asset": {"artifact_label": mod.ARTIFACT_LABEL},
        }
        with self.assertRaises(mod.ScoreProhibited):
            mod.assert_receipt_emits_no_score(receipt)

    def test_assert_rejects_missing_label(self) -> None:
        receipt = {"status": "BLOCKED", "score": None, "experiment": {}, "public_sample_asset": {}}
        with self.assertRaises(mod.ScoreProhibited):
            mod.assert_receipt_emits_no_score(receipt)

    def test_manifest_carries_label_and_null_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_transcript(d, "m-1_ALL3_run-1.jsonl", VALID_TRANSCRIPT)
            manifest = mod.build_public_sample_manifest(d)
        self.assertEqual(manifest["artifact_label"], mod.ARTIFACT_LABEL)
        self.assertIsNone(manifest["score"])
        self.assertFalse(manifest["is_benchmark_split"])
        self.assertFalse(manifest["is_model_score"])
        self.assertFalse(manifest["gradability"]["gradable_locally"])

    def test_manifest_has_no_accuracy_style_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_transcript(d, "m-1_ALL3_run-1.jsonl", VALID_TRANSCRIPT)
            manifest = mod.build_public_sample_manifest(d)
        blob = json.dumps(manifest).lower()
        for banned in ('"accuracy"', '"pass_rate"', '"correct"', '"reward"', '"pass@1"'):
            self.assertNotIn(banned, blob, f"manifest leaked a score-like key: {banned}")

    def test_missing_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            mod.build_public_sample_manifest("/nonexistent/e14/samples")

    def test_empty_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mod.build_public_sample_manifest(tmp)


class TestManifestOnSynthetic(unittest.TestCase):
    def test_grid_incomplete_when_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_transcript(d, "m-1_ALL3_run-1.jsonl", VALID_TRANSCRIPT)
            manifest = mod.build_public_sample_manifest(d)
        self.assertFalse(manifest["corpus"]["grid_is_complete"])
        self.assertEqual(manifest["corpus"]["transcript_count"], 1)

    def test_invalid_transcript_surfaces_in_manifest(self) -> None:
        broken = [{"role": "user", "content": PROMPT, "verdict": "correct"}]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_transcript(d, "m-1_ALL3_run-1.jsonl", broken)
            manifest = mod.build_public_sample_manifest(d)
        self.assertFalse(manifest["schema_validation"]["all_transcripts_valid"])
        self.assertTrue(manifest["gradability"]["ground_truth_present_anywhere"])


@unittest.skipUnless(SAMPLES_DIR.is_dir(), f"public sample corpus not present at {SAMPLES_DIR}")
class TestRealPublicCorpus(unittest.TestCase):
    """Pins the observed facts about Epoch AI's 150 public sample transcripts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = mod.build_public_sample_manifest(SAMPLES_DIR)

    def test_exactly_150_transcripts(self) -> None:
        self.assertEqual(self.manifest["corpus"]["transcript_count"], 150)

    def test_grid_is_6_models_by_5_problems_by_5_runs(self) -> None:
        corpus = self.manifest["corpus"]
        self.assertTrue(corpus["grid_is_complete"])
        self.assertEqual(sorted(corpus["models"]), sorted(mod.EXPECTED_MODELS))
        self.assertEqual(sorted(corpus["problem_tokens"]), sorted(mod.EXPECTED_PROBLEM_TOKENS))
        self.assertTrue(all(v == 25 for v in corpus["models"].values()))
        self.assertTrue(all(v == 30 for v in corpus["problem_tokens"].values()))

    def test_every_transcript_matches_the_schema(self) -> None:
        self.assertTrue(
            self.manifest["schema_validation"]["all_transcripts_valid"],
            self.manifest["schema_validation"]["invalid_transcripts"],
        )

    def test_no_ground_truth_and_no_grader_verdict_anywhere(self) -> None:
        self.assertFalse(self.manifest["gradability"]["ground_truth_present_anywhere"])
        self.assertFalse(self.manifest["gradability"]["grader_verdict_present_anywhere"])
        self.assertFalse(self.manifest["gradability"]["gradable_locally"])

    def test_prompt_preamble_is_uniform_across_all_150(self) -> None:
        self.assertTrue(self.manifest["prompt_contract"]["preamble_is_uniform"])

    def test_one_canonical_statement_per_problem(self) -> None:
        for token, hashes in self.manifest["prompt_contract"]["problem_statement_hashes"].items():
            self.assertEqual(len(hashes), 1, f"{token} has {len(hashes)} distinct statements")

    def test_all_five_problems_declare_integer_return_types(self) -> None:
        for token, types in self.manifest["prompt_contract"]["declared_return_types"].items():
            self.assertEqual(len(types), 1, f"{token}: {types}")
            self.assertIn("integer", types[0].lower(), f"{token}: {types}")

    def test_harness_turn_taxonomy_is_exhaustive(self) -> None:
        totals = self.manifest["harness_turn_totals"]
        self.assertNotIn(mod.TURN_UNKNOWN, totals, f"unclassified harness turns: {totals}")
        self.assertEqual(totals[mod.TURN_TASK_PROMPT], 150)
        self.assertEqual(totals[mod.TURN_CODE_RESULT], 265)
        self.assertEqual(totals[mod.TURN_FINAL_ANSWER_FAILED_STDERR], 6)
        self.assertEqual(totals[mod.TURN_FINAL_ANSWER_FAILED_TIMEOUT], 4)

    def test_final_answer_block_coverage(self) -> None:
        gradability = self.manifest["gradability"]
        self.assertEqual(gradability["transcripts_with_final_answer_block"], 149)
        self.assertEqual(
            gradability["transcripts_without_final_answer_block"],
            ["o1-mini_TIK2_run-4.jsonl"],
        )

    def test_corpus_hash_is_stable(self) -> None:
        again = mod.build_public_sample_manifest(SAMPLES_DIR)
        self.assertEqual(
            self.manifest["hashes"]["corpus_sha256"], again["hashes"]["corpus_sha256"]
        )
        self.assertEqual(len(self.manifest["hashes"]["file_sha256"]), 150)

    @unittest.skipUnless(ARCHIVE.is_file(), "public sample archive not present")
    def test_archive_hash_matches_recorded_download(self) -> None:
        self.assertEqual(
            mod.sha256_file(ARCHIVE),
            "7bdf3231086cc7de000ea57380c36a64abdf2644f1111b044d1bab0b383b0ff8",
        )

    def test_receipt_from_real_corpus_is_blocked(self) -> None:
        receipt = mod.build_blocked_receipt(
            self.manifest, recorded_at="2026-08-09", checkout=str(REPO_ROOT), commit="test"
        )
        mod.assert_receipt_emits_no_score(receipt)
        self.assertIsNone(receipt["score"])
        self.assertEqual(receipt["status"], "BLOCKED")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

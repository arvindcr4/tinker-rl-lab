"""Synthetic contract tests and published-count arithmetic; no raw-data dependencies.

Every identity, text example and constructed completion in this file is synthetic.
The source-function digests are code provenance commitments, not evidence hashes.
"""

import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from research.public_analysis import parser_v1, parser_v2, parser_v3
from research.public_analysis import parser_review as v
from research.public_analysis import reference_grader, rescue_statistics, selection, statistics

p = parser_v2
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "research/public_analysis"


class ParserV2Tests(unittest.TestCase):
    def assert_ok(self, text, value):
        actual = p.parse_answer(text)
        self.assertEqual(actual["status"], "ok", (text, actual))
        self.assertEqual(actual["value"], value)

    def assert_bad(self, text):
        self.assertNotEqual(p.parse_answer(text)["status"], "ok", text)

    def test_all_v1_positive_examples(self):
        for text, value in {
            "Steps have 999 and 123.\n#### 42": "42",
            r"Result is \boxed{42}": "42",
            "Final answer: 42": "42",
            "The steps take 2.\n42": "42",
            "answer is -0.50": "-1/2",
            "Answer = 1,234.5": "2469/2",
            r"\boxed{\frac{1}{2}}": "1/2",
            "#### +.50": "1/2",
            "#### $1,000$": "1000",
        }.items():
            with self.subTest(text=text):
                self.assert_ok(text, value)

    def test_final_heading_then_marker(self):
        for heading in ["Final Answer:", "Answer:", "final answer :"]:
            with self.subTest(heading=heading):
                self.assert_ok("#### " + heading + "\n\n#### 75", "75")

    def test_final_heading_then_numeric(self):
        self.assert_ok("#### Final Answer:\n75", "75")

    def test_placeholder_then_numeric(self):
        self.assert_ok("#### <answer>  \n10", "10")

    def test_placeholder_then_marker(self):
        self.assert_ok("#### <answer>\n#### 10", "10")

    def test_xml_anchored_single_line(self):
        for text in [
            "#### <answer>10</answer>",
            "<answer>10</answer>",
            "#### <answer> 0.5 </answer>",
        ]:
            with self.subTest(text=text):
                self.assert_ok(text, "1/2" if "0.5" in text else "10")

    def test_bold_complete_marker(self):
        self.assert_ok("#### Final Answer:\n**#### 10**", "10")
        self.assert_ok("  **#### -1/2**  ", "-1/2")

    def test_agreeing_explicit_claims(self):
        self.assert_ok("#### 0.5\n#### <answer>\n#### 1/2", "1/2")
        self.assert_ok("Answer: 0.5\n\\boxed{1/2}\n<answer>.50</answer>", "1/2")

    def test_conflicts_fail_closed(self):
        for text in [
            "#### 2\n#### 3",
            "\\boxed{2}\n#### 3",
            "#### 2\n<answer>3</answer>",
            "#### 2\n#### <answer>\n3",
            "Answer: 2\n#### 3",
            "#### 2\n3",
            "<answer>2</answer>\n<answer>3</answer>",
        ]:
            with self.subTest(text=text):
                self.assertEqual(p.parse_answer(text)["status"], "ambiguous_final_answers")

    def test_incidental_numbers_remain_rejected(self):
        for text in [
            "I used 42 apples",
            "There were 7, then 42.",
            "2 + 2 = 4",
            "2026 year 3",
            "The answer might be 42 or 43",
            "",
            " ",
            "<think>42</think>",
        ]:
            with self.subTest(text=text):
                self.assert_bad(text)

    def test_invalid_numeric_security_inputs(self):
        for value in [
            "NaN",
            "Infinity",
            "1e100000",
            "__import__('os').system('true')",
            "1/0",
            "1,23",
            "42 units",
            "42%",
            "1+2",
            r"\frac{1}{0}",
            "9" * 101,
            "42; print(1)",
        ]:
            for wrapper in [
                "#### {}",
                "<answer>{}</answer>",
                "#### <answer>{}</answer>",
                "**#### {}**",
            ]:
                with self.subTest(value=value, wrapper=wrapper):
                    self.assert_bad(wrapper.format(value))

    def test_xml_malformed_no_fallback(self):
        for text in [
            "<answer>10",
            "<answer>\n10",
            "<answer>10</answer",
            "</answer>\n10",
            "<answer>10</wrong>\n10",
            "<answer x='y'>10</answer>\n10",
            "<answer><answer>10</answer></answer>",
            "#### <answer>10\n10",
            "#### <answer>\n10\n</answer>",
            "<ANSWER>10</ANSWER>\n10",
            "prose <answer>10</answer>\n10",
            "<answer>10</answer> prose\n10",
            "< answer>10</answer>\n10",
            "<answerish>10</answerish>\n10",
            "<answer>10</answer><answer>10</answer>\n10",
        ]:
            with self.subTest(text=text):
                self.assert_bad(text)

    def test_placeholder_exception_is_narrow(self):
        for text in [
            "#### <answer>",
            "#### <answer>\nI found 10",
            "#### <answer>.\n10",
            "#### <Answer>\n10",
            "#### <answer>\nnot numeric\n10",
            "#### <answer>\n#### <answer>\n10",
        ]:
            with self.subTest(text=text):
                self.assert_bad(text)

    def test_headings_never_hide_invalid_candidates(self):
        for text in [
            "#### maybe\n42",
            "#### maybe:\n42",
            "#### Final Answer: maybe\n42",
            "#### 10 apples\n#### 10",
            "#### Final Answer:\nprose\n42",
            "#### Final Answer:\n#### Final Answer:\n42",
            "#### Final Answer:",
        ]:
            with self.subTest(text=text):
                self.assert_bad(text)

    def test_unsupported_subsection_headings_remain_invalid(self):
        for heading in ["a. **Hourly wage:**", "**A. Employee wages:**", "For 100 mg capsules:"]:
            with self.subTest(heading=heading):
                self.assert_bad("#### " + heading + "\n#### 9080")

    def test_currency_prefix_remains_invalid(self):
        self.assert_bad("#### $9,080\n#### <answer>\n#### 9080")

    def test_malformed_bold_markers_remain_invalid(self):
        for text in [
            "**#### 10",
            "#### 10**",
            "***#### 10***",
            "**#### 10****",
            "prose **#### 10**",
            "**#### 10** trailing",
            "**#### 10 #### 10**",
        ]:
            with self.subTest(text=text):
                self.assert_bad(text)

    def test_malformed_boxes_fail_closed(self):
        self.assertEqual(p.parse_answer("\\boxed{42\n42")["status"], "malformed_box")
        self.assert_bad("\\boxed{2+2}\n#### 4")

    def test_heading_box_successor(self):
        self.assert_ok("#### Answer:\n\\boxed{42}", "42")
        self.assert_bad("#### Answer:\nprose \\boxed{42}")

    def test_exact_rational_normalization(self):
        for text, value in [
            ("0.10", "1/10"),
            ("10000000000000001", "10000000000000001"),
            ("-0", "0"),
            ("1.5/0.5", "3"),
            ("−0.5", "-1/2"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(p.normalize_number(text), value)

    def test_unknown_terminal_tokens_are_not_silently_removed(self):
        self.assert_bad("#### 42<|unknown|>")


def record(old, new, rid="synthetic_request"):
    return {
        "request": {"logical_request_id": rid, "prompt_id": "synthetic_prompt"},
        "raw_completion_text": "#### **Answer:**\n7",
        "v2": old,
        "v3": new,
    }


BAD = {"status": "invalid_numeric_answer", "source": None, "value": None}
OK = {"status": "ok", "source": "hash_marker", "value": "7"}


def review(name="synthetic_reviewer_one", cases=None):
    return {
        "reviewer_id": name,
        "packet_sha256": "synthetic_packet_digest",
        "independent_output_only_review": True,
        "completed_at_utc": "2000-01-01T00:00:00+00:00",
        "cases": cases
        if cases is not None
        else [
            {
                "case_id": "c",
                "verdict": "single_numeric",
                "value": "7",
                "rationale": "The fabricated explicit answer is 7.",
            }
        ],
    }


class ParserReviewTests(unittest.TestCase):
    def test_packet_exact_safe_fields(self):
        rows, m = v.packet_rows([record(BAD, OK), record(OK, OK, "unchanged")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"case_id", "raw_completion_text"})
        self.assertEqual(rows[0]["raw_completion_text"], "#### **Answer:**\n7")
        self.assertEqual(set(m.values()), {"synthetic_request"})
        self.assertEqual(len(rows[0]["case_id"]), 32)

    def test_packet_includes_rejected_source_reverse_value(self):
        rows, m = v.packet_rows(
            [
                record(BAD, {**BAD, "status": "missing_final_answer"}, "a"),
                record(OK, {**OK, "source": "boxed"}, "b"),
                record(OK, BAD, "c"),
                record(OK, {**OK, "value": "8"}, "d"),
            ]
        )
        self.assertEqual(len(rows), 4)

    def test_zero_changed_inconclusive(self):
        d = v.decide([record(OK, OK)], {}, [review("one", []), review("two", [])])
        self.assertEqual(d["decision"], "INCONCLUSIVE_NO_FRESH_POSITIVE_CASES")
        self.assertTrue(d["fresh_v2_objects_preserved"])

    def test_reject_only_inconclusive(self):
        reviews = [
            review(
                n,
                [
                    {
                        "case_id": "c",
                        "verdict": "no_single_numeric",
                        "value": None,
                        "rationale": "No scalar.",
                    }
                ],
            )
            for n in ("one", "two")
        ]
        d = v.decide(
            [record(BAD, {**BAD, "status": "missing_final_answer"})],
            {"c": "synthetic_request"},
            reviews,
        )
        self.assertEqual(d["decision"], "INCONCLUSIVE_NO_FRESH_POSITIVE_CASES")

    def test_narrow_pass(self):
        d = v.decide([record(BAD, OK)], {"c": "synthetic_request"}, [review("one"), review("two")])
        self.assertEqual(d["decision"], "NARROW_PASS_ON_OBSERVED_CHANGED_EXTRACTIONS")
        self.assertEqual(d["accepted_ambiguous_or_wrong"], 0)

    def test_disagreement_fails(self):
        b = review("two")
        b["cases"][0]["value"] = "8"
        d = v.decide([record(BAD, OK)], {"c": "synthetic_request"}, [review(), b])
        self.assertEqual(d["decision"], "FAIL")
        self.assertEqual(d["review_disagreements_unresolved"], 1)

    def test_agree_wrong_value_fails(self):
        a, b = review(), review("two")
        a["cases"][0]["value"] = b["cases"][0]["value"] = "8"
        d = v.decide([record(BAD, OK)], {"c": "synthetic_request"}, [a, b])
        self.assertEqual(d["accepted_ambiguous_or_wrong"], 1)

    def test_ambiguous_accept_fails(self):
        a, b = review(), review("two")
        for x in [a, b]:
            x["cases"][0].update(verdict="ambiguous", value=None)
        self.assertEqual(
            v.decide([record(BAD, OK)], {"c": "synthetic_request"}, [a, b])["decision"], "FAIL"
        )

    def test_valid_source_change_fails(self):
        d = v.decide(
            [record(OK, {**OK, "source": "boxed"})],
            {"c": "synthetic_request"},
            [review(), review("two")],
        )
        self.assertEqual(d["decision"], "FAIL")
        self.assertEqual(d["source_change_count"], 1)

    def test_reverse_fails(self):
        d = v.decide([record(OK, BAD)], {"c": "synthetic_request"}, [review(), review("two")])
        self.assertEqual(d["decision"], "FAIL")
        self.assertEqual(d["reverse_status_count"], 1)

    def test_duplicate_and_missing_review_rejected(self):
        for cases in [[], review()["cases"] * 2]:
            with self.assertRaises(ValueError):
                v.validate_review(review(cases=cases), "synthetic_packet_digest", {"c"})

    def test_packet_binding_and_independence(self):
        for key, value in [
            ("packet_sha256", "wrong"),
            ("independent_output_only_review", False),
            ("reviewer_id", ""),
        ]:
            r = review()
            r[key] = value
            with self.assertRaises(ValueError):
                v.validate_review(r, "synthetic_packet_digest", {"c"})

    def test_distinct_reviewers(self):
        with self.assertRaises(ValueError):
            v.decide([record(BAD, OK)], {"c": "synthetic_request"}, [review(), review()])

    def test_packet_mapping_and_text_verified(self):
        records = [record(BAD, OK)]
        rows, mapping = v.packet_rows(records)
        v.validate_packet(rows, mapping, records)
        bad = deepcopy(rows)
        bad[0]["raw_completion_text"] = "#### 99"
        with self.assertRaises(ValueError):
            v.validate_packet(bad, mapping, records)
        bad = deepcopy(rows)
        bad[0]["question"] = "leak"
        with self.assertRaises(ValueError):
            v.validate_packet(bad, mapping, records)
        with self.assertRaises(ValueError):
            v.validate_packet(rows + rows, mapping, records)

    def test_review_noncanonical_or_guessed_values(self):
        for value in ["7.0", "14/2", "1/0", "7 apples", "NaN", 7]:
            r = review()
            r["cases"][0]["value"] = value
            with self.assertRaises(ValueError):
                v.validate_review(r, "synthetic_packet_digest", {"c"})

    def test_utc_only(self):
        for value in ["2000-01-01T00:00:00", "2000-01-01T00:00:00+01:00"]:
            with self.assertRaises(ValueError):
                v.utc(value)

    def test_duplicate_json_keys_and_nonfinite(self):
        for value in ['{"a":1,"a":2}', '{"a":NaN}']:
            with self.assertRaises(ValueError):
                v.parse(value)


class ParserPolicyTests(unittest.TestCase):
    def test_copied_definition_digests(self):
        provenance = json.loads((PACKAGE / "PROVENANCE.json").read_text())
        copied = 0
        for filename, metadata in provenance["modules"].items():
            nodes = {}
            source = (PACKAGE / filename).read_text()
            for n in ast.parse(source).body:
                if isinstance(n, ast.FunctionDef):
                    nodes[n.name] = n
                elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                    nodes[n.targets[0].id] = n
            for name, expected in metadata["selected_definition_source_sha256"].items():
                with self.subTest(module=filename, definition=name):
                    digest = hashlib.sha256(
                        ast.get_source_segment(source, nodes[name]).encode()
                    ).hexdigest()
                    self.assertEqual(digest, expected)
                    copied += 1
        self.assertEqual(copied, 58)

    def test_source_fingerprints_ignore_version_only_ast_fields(self):
        # Python 3.11 has no FunctionDef.type_params; emulate that representation.
        # Source-span hashing must remain identical without weakening text identity.
        provenance = json.loads((PACKAGE / "PROVENANCE.json").read_text())
        checked = 0
        for filename, metadata in provenance["modules"].items():
            source = (PACKAGE / filename).read_text()
            for node in ast.parse(source).body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if hasattr(node, "type_params"):
                    self.assertEqual(node.type_params, [])
                    delattr(node, "type_params")
                expected = metadata["selected_definition_source_sha256"][node.name]
                self.assertEqual(
                    hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
                    expected,
                )
                checked += 1
        self.assertGreater(checked, 30)

    def test_shared_numeric_grammar_and_helpers(self):
        for value in [
            "0.1",
            "-0",
            "−0.5",
            "1.5/0.5",
            "1,000",
            r"\frac{1}{2}",
            "$.5$",
            r"\(1/2\)",
            "9" * 101,
            "1/0",
            "NaN",
        ]:
            self.assertEqual(parser_v1.normalize_number(value), parser_v2.normalize_number(value))
            self.assertEqual(parser_v2.normalize_number(value), parser_v3.normalize_number(value))

    def test_v1_full_objects(self):
        for text, source, value in [
            ("#### 42", "hash_marker", "42"),
            ("Final answer: 0.5", "explicit_final", "1/2"),
            ("steps\n7", "standalone_final_line", "7"),
            ("\\boxed{0.5}\n#### 1/2", "boxed+hash_marker", "1/2"),
        ]:
            self.assertEqual(
                parser_v1.parse_answer(text), {"status": "ok", "source": source, "value": value}
            )

    def test_v1_version_specific_ambiguity_is_preserved(self):
        text = "Answer: 2\n#### 3"
        self.assertEqual(parser_v1.parse_answer(text)["value"], "3")
        self.assertEqual(parser_v2.parse_answer(text)["status"], "ambiguous_final_answers")
        self.assertEqual(parser_v3.parse_answer(text)["status"], "ambiguous_final_answers")
        self.assertEqual(
            parser_v1.parse_answer("#### Answer:\n7")["status"], "invalid_numeric_answer"
        )

    def test_v3_three_narrow_extensions(self):
        examples = [
            "#### **Answer:**\n7",
            "#### **Final Answer:**\n$$\n\\boxed{7}\n$$",
            "#### Answer:\n$$\n\\boxed{7}\n$$",
            "#### <answer>\n#### 7\n#### </answer>",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertNotEqual(parser_v2.parse_answer(text)["status"], "ok")
                self.assertEqual(parser_v3.parse_answer(text)["value"], "7")
        self.assertEqual(parser_v3.PARSER_STATUS, "prospective_candidate_not_deployed")

    def test_v3_requires_physical_adjacency(self):
        examples = [
            "#### Answer:\n$$\n\n\\boxed{7}\n$$",
            "#### Answer:\n$$\n\\boxed{7}\n\n$$",
            "#### <answer>\n\n#### 7\n#### </answer>",
            "#### <answer>\n#### 7\n\n#### </answer>",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertNotEqual(parser_v3.parse_answer(text)["status"], "ok")

    def test_v3_rejects_malformed_or_conflicting_extensions(self):
        examples = [
            "#### **maybe:**\n7",
            "#### **Answer:**\n8\n#### 7",
            "#### Answer:\n$$\ntext \\boxed{7}\n$$",
            "#### Answer:\n$$\n\\boxed{7}+1\n$$",
            "#### <answer>\n#### 7 apples\n#### </answer>",
            "**#### <answer>**\n#### 7\n#### </answer>",
            "#### <answer>\n#### 7\n#### </answer>\n#### 8",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertNotEqual(parser_v3.parse_answer(text)["status"], "ok")

    def test_v2_accepted_object_preservation_on_synthetic_grid(self):
        values = ["0", "7", "-1/2", "1,000", "0.50", r"\frac{3}{4}", "NaN", "1/0", "7 units"]
        wrappers = [
            "#### {}",
            "Answer: {}",
            "Final answer = {}",
            "<answer>{}</answer>",
            "#### <answer>{}</answer>",
            "**#### {}**",
            "#### Answer:\n{}",
            "#### <answer>\n#### {}",
            "\\boxed{{{}}}",
            "{}",
        ]
        accepted = 0
        for value in values:
            for wrapper in wrappers:
                for prefix in ["", "Synthetic reasoning mentions 99.\n"]:
                    for suffix in ["", "\n", "\n#### " + value]:
                        text = prefix + wrapper.format(value) + suffix
                        old, new = parser_v2.parse_answer(text), parser_v3.parse_answer(text)
                        if old["status"] == "ok":
                            with self.subTest(text=text):
                                self.assertEqual(old, new)
                                accepted += 1
        self.assertGreater(accepted, 250)

    def test_terminal_preprocessing_is_exact_and_opt_in(self):
        self.assertEqual(
            parser_v1.strip_terminal_tokens("#### 7<|im_end|>\n<|endoftext|>\n"), "#### 7"
        )
        self.assertEqual(parser_v1.strip_terminal_tokens("x <|im_end|> y"), "x <|im_end|> y")
        self.assertEqual(parser_v1.strip_terminal_tokens("7<|unknown|>"), "7<|unknown|>")
        self.assertNotEqual(parser_v2.parse_answer("#### 7<|im_end|>")["status"], "ok")

    def test_reference_comparator_is_intentionally_permissive(self):
        text = "The answer is $7 dollars."
        self.assertEqual(reference_grader.extract_answer(text), "7")
        self.assertNotEqual(parser_v2.parse_answer(text)["status"], "ok")
        self.assertEqual(reference_grader.normalize_number("7.0"), "7")
        self.assertEqual(reference_grader.normalize_number("not numeric"), "not numeric")


class StatisticsTests(unittest.TestCase):
    def test_known_wilson_interval(self):
        lower, upper = statistics.wilson(5, 10)
        self.assertAlmostEqual(lower, 0.236593090512564, places=14)
        self.assertAlmostEqual(upper, 0.763406909487436, places=14)
        self.assertEqual(statistics.estimate(5, 10)["fraction"], 0.5)

    def test_stable_counts_fail_closed(self):
        for k, n in [(True, 2), (1, False), (1.0, 2), (-1, 2), (3, 2), (0, -1), (1, 0)]:
            with self.subTest(k=k, n=n):
                with self.assertRaises(ValueError):
                    statistics.wilson(k, n)
                with self.assertRaises(ValueError):
                    selection.interval(k, n)

    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(statistics.wilson(0, 0))
        self.assertIsNone(statistics.estimate(0, 0)["fraction"])
        self.assertEqual(selection.interval(0, 0)["status"], "UNDEFINED_NO_ELIGIBLE_QUESTIONS")
        self.assertIsNone(selection.interval(0, 0)["zero_events_exact_upper_one_sided_95"])

    def test_zero_events_exact_bound(self):
        for n in [1, 20, 64, 1000000000]:
            expected = -math.expm1(math.log(0.05) / n)
            self.assertEqual(
                statistics.estimate(0, n)["zero_events_exact_upper_one_sided_95"], expected
            )
            self.assertEqual(
                selection.interval(0, n)["zero_events_exact_upper_one_sided_95"], expected
            )
            self.assertIsNone(statistics.estimate(1, n)["zero_events_exact_upper_one_sided_95"])

    def test_interval_versions_agree_within_roundoff(self):
        for n in [1, 8, 64, 512, 4096]:
            for k in sorted({0, n // 2, n}):
                intervals = [
                    statistics.wilson(k, n),
                    selection.interval(k, n)["wilson_95"],
                    rescue_statistics.wilson(k, n),
                ]
                for interval in intervals[1:]:
                    for a, b in zip(intervals[0], interval):
                        self.assertAlmostEqual(a, b, places=14)

    def test_uniform_prediction_exact_reference(self):
        expected = float(Fraction(24, 33))
        self.assertAlmostEqual(statistics.prediction(0, 1), expected, places=14)
        self.assertAlmostEqual(statistics.prediction(8, 1), expected, places=14)
        for prior in [0.5, 1.0]:
            self.assertAlmostEqual(
                rescue_statistics.rescue_prediction(0, 8, 24, prior),
                statistics.prediction(0, prior),
                places=13,
            )
            self.assertAlmostEqual(
                rescue_statistics.rescue_prediction(8, 8, 24, prior),
                statistics.prediction(8, prior),
                places=13,
            )

    def test_prediction_rejects_nonhomogeneous_group(self):
        with self.assertRaises(ValueError):
            statistics.prediction(2, 1)
        with self.assertRaises(ValueError):
            rescue_statistics.rescue_prediction(2, 8, 24, 1)

    def test_legacy_empty_summary_behavior_is_preserved(self):
        summary = rescue_statistics.binomial_summary(0, 0)
        self.assertIsNone(summary["fraction"])
        self.assertIsNone(summary["wilson_95"])
        self.assertIsNone(summary["zero_events_exact_upper_one_sided_95"])
        # The early function does not validate k before its zero-n return.
        self.assertIsNone(rescue_statistics.wilson(1, 0))


def synthetic_completion(index, reward=0, parsed_value="1", cap=False, status="ok"):
    return {
        "sample_index": index,
        "reward": reward,
        "parsed_value": parsed_value,
        "cap_hit": cap,
        "parse_status": status,
        "text": "#### " + str(parsed_value),
        "output_token_count": 2,
        "input_token_count": 3,
    }


def synthetic_row(index, completions):
    return {
        "prompt_id": f"synthetic_question_{index}",
        "prompt_index": index,
        "original_prompt_rank": 640 + index,
        "dataset_index": index,
        "question": "Synthetic: what is one plus one?",
        "gold_answer": "#### 2",
        "gold_value": "2",
        "completions": completions,
    }


def synthetic_screen(eligible=()):
    eligible = set(eligible)
    return [
        synthetic_row(
            i,
            [
                synthetic_completion(j, int(i not in eligible), "1" if i in eligible else "2")
                for j in range(8)
            ],
        )
        for i in range(4096)
    ]


class SelectionTests(unittest.TestCase):
    def test_selects_only_first_64_eligible_in_frozen_order(self):
        rows = synthetic_screen(range(100, 170))
        result = selection.selection_from_screen(rows)
        self.assertEqual(result["eligible_screen_indices"], list(range(100, 170)))
        self.assertEqual(result["selected_screen_indices"], list(range(100, 164)))
        self.assertEqual(result["eligible_questions"], 70)
        self.assertEqual(result["selected_questions"], 64)
        self.assertEqual(selection.validate_selection(result, rows), result)

    def test_all_three_exclusions_apply_without_dropping_rows(self):
        rows = synthetic_screen([0, 1, 2, 3])
        rows[0]["completions"][0]["parse_status"] = "invalid_numeric_answer"
        rows[1]["completions"][0]["cap_hit"] = True
        rows[2]["completions"][0]["parsed_value"] = "2"
        result = selection.selection_from_screen(rows)
        self.assertEqual(result["selected_screen_indices"], [3])
        self.assertEqual(
            result["ineligibility_reason_counts"],
            {"any_parse_failure": 1, "any_cap": 1, "any_valid_correct": 4093},
        )
        self.assertEqual(len(result["per_screen_question"]), 4096)

    def test_no_eligible_questions_keeps_zero_denominator(self):
        result = selection.selection_from_screen(synthetic_screen())
        self.assertEqual(result["selected_questions"], 0)
        self.assertEqual(result["selected_screen_indices"], [])

    def test_incomplete_screen_rejected(self):
        with self.assertRaises(ValueError):
            selection.selection_from_screen(synthetic_screen()[:-1])

    def test_reordered_indices_ranks_or_duplicate_id_rejected(self):
        for field in ["prompt_index", "original_prompt_rank", "prompt_id"]:
            rows = synthetic_screen()
            rows[1][field] = rows[0][field]
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    selection.selection_from_screen(rows)

    def test_missing_screen_draw_rejected(self):
        rows = synthetic_screen()
        rows[0]["completions"].pop()
        with self.assertRaises(ValueError):
            selection.selection_from_screen(rows)

    def test_altered_selected_subset_rejected(self):
        rows = synthetic_screen([2, 4])
        selected = selection.selection_from_screen(rows)
        selected["selected_screen_indices"] = [4]
        with self.assertRaises(ValueError):
            selection.validate_selection(selected, rows)

    def test_primary_success_requires_parse_and_uncapped_correctness(self):
        correct = synthetic_completion(0, 1, "2")
        self.assertTrue(selection.primary_success(correct, "2"))
        for key, value in [
            ("parse_status", "invalid_numeric_answer"),
            ("cap_hit", True),
            ("parsed_value", "1"),
        ]:
            self.assertFalse(selection.primary_success({**correct, key: value}, "2"))

    def test_quality_gate_includes_failures_and_caps(self):
        cs = [synthetic_completion(i) for i in range(20)]
        cs[0]["parse_status"] = "invalid_numeric_answer"
        cs[1]["cap_hit"] = True
        rows = [synthetic_row(0, cs)]
        self.assertTrue(selection.quality(rows)["passes_smoke_quality_gate"])
        self.assertEqual(selection.quality(rows)["completions"], 20)
        cs[2]["cap_hit"] = True
        self.assertFalse(selection.quality(rows)["passes_smoke_quality_gate"])
        self.assertFalse(selection.quality([])["passes_smoke_quality_gate"])


class SummaryTests(unittest.TestCase):
    def test_complete_synthetic_reward_summary(self):
        patterns = [[0] * 8 + [1] + [0] * 23, [1] * 8 + [0] + [1] * 23, [0, 1] * 16, [0] * 32]
        rows = [
            synthetic_row(
                i,
                [
                    synthetic_completion(j, value, "2" if value else "1")
                    for j, value in enumerate(pattern)
                ],
            )
            for i, pattern in enumerate(patterns)
        ]
        summary, per_question, audit = statistics.analyze_validated(rows)
        self.assertEqual(summary["primary"]["events"], 2)
        self.assertEqual(summary["primary"]["denominator"], 3)
        self.assertEqual(summary["all_wrong_descriptive"]["events"], 1)
        self.assertEqual(summary["all_wrong_descriptive"]["denominator"], 2)
        self.assertEqual(summary["all_correct_descriptive"]["events"], 1)
        self.assertEqual(summary["all_correct_descriptive"]["denominator"], 1)
        self.assertEqual(summary["mixedness_all_prompts"]["8"]["events"], 1)
        self.assertEqual(summary["mixedness_all_prompts"]["32"]["events"], 3)
        self.assertEqual(summary["quality"]["completions"], 128)
        self.assertEqual(summary["generated_tokens"], 256)
        self.assertEqual(summary["input_tokens_counting_every_draw"], 384)
        self.assertEqual(len(per_question), 4)
        self.assertEqual(len(audit), 2)
        self.assertEqual(summary["manual_review_required"], 2)
        self.assertFalse(summary["manual_review_complete"])
        self.assertIsNone(summary["conservative_numeric_witnesses"]["verified_count"])

    def test_invalid_first_change_remains_primary_event(self):
        cs = [synthetic_completion(i, 1, "2") for i in range(32)]
        cs[8] = synthetic_completion(8, 0, None, status="invalid_numeric_answer")
        summary, _, _ = statistics.analyze_validated([synthetic_row(0, cs)])
        self.assertEqual(summary["primary"]["events"], 1)
        self.assertEqual(summary["primary"]["denominator"], 1)
        self.assertEqual(
            summary["event_type_sensitivity_nonexclusive"]["first_change_parser_failure"], 1
        )
        self.assertEqual(summary["parseable_uncapped_sensitivity"]["denominator"], 0)
        self.assertEqual(
            summary["conservative_numeric_witnesses"]["machine_eligible_pending_manual_review"], 0
        )

    def test_empty_summary_has_no_empirical_events(self):
        summary, per_question, audit = statistics.analyze_validated([])
        self.assertEqual(summary["primary"]["denominator"], 0)
        self.assertIsNone(summary["primary"]["fraction"])
        self.assertEqual(per_question, [])
        self.assertEqual(audit, [])


class PublicPackageTests(unittest.TestCase):
    def run_cli(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-B", "-m", "research.public_analysis", *args],
            cwd=ROOT,
            input=input,
            text=True,
            capture_output=True,
        )

    def test_parse_cli(self):
        result = self.run_cli("parse", "#### 1/2", "--policy", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["label"], "ARITHMETIC_OR_PARSER_DIAGNOSTIC_ONLY")
        self.assertEqual(data["result"]["value"], "1/2")

    def test_stdin_cli_terminal_token(self):
        result = self.run_cli("parse", "-", input="#### 7<|im_end|>")
        self.assertEqual(json.loads(result.stdout)["result"]["value"], "7")

    def test_counts_cli_and_invalid_input(self):
        result = self.run_cli("counts", "0", "20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"], selection.interval(0, 20))
        self.assertNotEqual(self.run_cli("counts", "3", "2").returncode, 0)

    def test_public_modules_have_only_stdlib_or_relative_imports(self):
        for path in PACKAGE.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], sys.stdlib_module_names, path.name)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertIn(node.module.split(".")[0], sys.stdlib_module_names, path.name)


class PublishedCountArithmeticTests(unittest.TestCase):
    """Check arithmetic of published aggregates, not raw empirical reproduction."""

    def test_published_primary_interval(self):
        result = selection.interval(11, 64)
        self.assertEqual(result["fraction"], 0.171875)
        self.assertEqual(result["wilson_95"], [0.09877742301244645, 0.282132116226582])
        self.assertEqual([round(100 * x, 2) for x in result["wilson_95"]], [9.88, 28.21])

    def test_published_eligibility_interval_and_distinct_witness_count(self):
        result = selection.interval(87, 4096)
        self.assertEqual(result["fraction"], 0.021240234375)
        self.assertEqual(result["wilson_95"], [0.017252591636140006, 0.026125051281752146])
        self.assertEqual(selection.interval(3, 64)["fraction"], 0.046875)
        self.assertNotEqual(selection.interval(3, 64), selection.interval(11, 64))


if __name__ == "__main__":
    unittest.main()

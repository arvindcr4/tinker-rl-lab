"""Mutation tests for public records, not independent scientific revalidation."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools import check_public_results as checker


ROOT = Path(__file__).resolve().parents[1]


class PublicResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in checker.INPUT_FILES:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    def edit(self, path, mutate):
        target = self.root / path
        data = json.loads(target.read_text(encoding="utf-8"))
        mutate(data)
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def bad(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            checker.check_public_results(self.root)

    def test_published_records_and_document_tables(self):
        result = checker.check_public_results(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_cases"], 64)
        self.assertEqual(result["frozen_primary_events"], 11)
        self.assertEqual(result["conservative_numeric_witnesses"], 3)
        self.assertEqual(result["eligible_screened"], 87)
        self.assertEqual(result["screened_questions"], 4096)
        self.assertEqual(result["reviewed_response_records"], 126)
        self.assertEqual(result["document_tables_checked"], 4)
        self.assertIn("no raw execution replay", result["limitation"])

    def test_cli_from_unrelated_working_directory(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/check_public_results.py"),
                "--root",
                str(self.root),
            ],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["frozen_primary_events"], 11)

    def test_cli_default_root_is_script_repository(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tools/check_public_results.py")],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_input_is_value_error_and_cli_failure(self):
        (self.root / checker.LEDGER).unlink()
        self.bad("missing public scientific record")
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/check_public_results.py"),
                "--root",
                str(self.root),
            ],
            cwd=self.temp.name,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        failure = json.loads(result.stdout)
        self.assertEqual(failure["status"], "FAIL")
        self.assertIn("missing public scientific record", failure["error"])
        self.assertEqual(failure["scope"], checker.LIMITATION)
        self.assertEqual(result.stderr, "")

    def test_duplicate_json_key(self):
        target = self.root / checker.SUMMARY
        target.write_text(
            target.read_text().replace('"events": 11', '"events": 11, "events": 11', 1)
        )
        self.bad("duplicate JSON key events")

    def test_nonfinite_json_and_overflow_exponent(self):
        target = self.root / checker.SUMMARY
        original = target.read_text()
        for value in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.subTest(value=value):
                target.write_text(
                    original.replace('"fraction": 0.171875', f'"fraction": {value}', 1)
                )
                self.bad("nonfinite JSON")

    def test_missing_case(self):
        self.edit(checker.LEDGER, lambda d: d["cases"].pop())
        self.bad("selected cohort length")

    def test_oversized_finite_integer_fails_as_value_error(self):
        self.edit(checker.SUMMARY, lambda d: d["frozen_primary"].update(fraction=10**400))
        self.bad("Malformed or missing public scientific record")

    def test_reordered_cases(self):
        self.edit(checker.LEDGER, lambda d: d["cases"].reverse())
        self.bad("ordered case index")

    def test_duplicate_case(self):
        self.edit(checker.LEDGER, lambda d: d["cases"].__setitem__(1, d["cases"][0]))
        self.bad("ordered case index")

    def test_duplicate_scientific_identity(self):
        self.edit(
            checker.LEDGER, lambda d: d["cases"][1].update(audit_id=d["cases"][0]["audit_id"])
        )
        self.bad("duplicate case audit_id")

    def test_boolean_case_index_is_not_integer(self):
        self.edit(checker.LEDGER, lambda d: d["cases"][0].update(selected_index=False))
        self.bad("incorrect value type")

    def test_integer_flag_is_not_boolean(self):
        self.edit(checker.LEDGER, lambda d: d["cases"][0].update(frozen_primary_event=1))
        self.bad("expected boolean")

    def test_changed_question_hash(self):
        self.edit(
            checker.LEDGER, lambda d: d["cases"][0].update(question="Altered scientific question")
        )
        self.bad("question hash")

    def test_changed_reference_hash(self):
        self.edit(checker.LEDGER, lambda d: d["cases"][0].update(recorded_gold_reasoning="#### 9"))
        self.bad("recorded_gold_reasoning hash")

    def test_prompt_identity_is_normalized_not_raw_question_hash(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["cases"][7].update(prompt_id=d["cases"][7]["question_sha256"]),
        )
        self.bad("prompt_id")

    def test_changed_selection_rank(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["deterministic_selection"]["selected_original_ranks"].__setitem__(0, 677),
        )
        self.bad("screen rank offset")

    def test_reordered_screen_indices(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["deterministic_selection"]["selected_screen_indices"].reverse(),
        )
        self.bad("screen indices duplicate or reordered")

    def test_boolean_screen_index(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["deterministic_selection"]["selected_screen_indices"].__setitem__(0, False),
        )
        self.bad("expected nonnegative integer")

    def test_summary_event_count_mismatch(self):
        self.edit(checker.SUMMARY, lambda d: d["frozen_primary"].update(events=12))
        self.bad("frozen primary.events")

    def test_summary_fraction_mismatch(self):
        self.edit(checker.SUMMARY, lambda d: d["frozen_primary"].update(fraction=0.5))
        self.bad("fraction: arithmetic mismatch")

    def test_summary_wilson_mismatch(self):
        self.edit(checker.SUMMARY, lambda d: d["frozen_primary"]["wilson_95"].__setitem__(0, 0.05))
        self.bad("wilson_95")

    def test_missing_wilson_is_rejected(self):
        self.edit(checker.SUMMARY, lambda d: d["frozen_primary"].pop("wilson_95"))
        self.bad("wilson_95")

    def test_original_and_refined_labels_are_not_interchangeable(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d.update(
                second_review_refined_mutually_exclusive_counts=d[
                    "second_review_original_mutually_exclusive_counts"
                ]
            ),
        )
        self.bad("refined counts")

    def test_summary_boolean_count(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["conservative_same_denominator_support"][
                "not_corroborated_event_partition"
            ].update(clear_wrong_gold_match=True),
        )
        self.bad("incorrect value type")

    def test_nested_summary_reference_mismatch(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["unit_sensitive_cases"]["cases"][0].update(audit_id="0" * 64),
        )
        self.bad("summary.unit_sensitive_cases.audit_id")

    def test_summary_reference_group_count_mismatch(self):
        self.edit(checker.SUMMARY, lambda d: d["unit_sensitive_cases"].update(count=3))
        self.bad("summary.unit_sensitive_cases.count")

    def test_missing_event(self):
        self.edit(checker.LEDGER, lambda d: d["events"].pop())
        self.bad("ordered event identities")

    def test_event_identity_mismatch(self):
        self.edit(checker.LEDGER, lambda d: d["events"][0].update(prompt_id="0" * 64))
        self.bad("event 0.prompt_id")

    def test_duplicate_fresh_success(self):
        self.edit(
            checker.LEDGER, lambda d: d["events"][0]["frozen_primary_success_indices"].append(23)
        )
        self.bad("duplicate/reordered fresh")

    def test_out_of_range_fresh_success(self):
        self.edit(
            checker.LEDGER, lambda d: d["events"][0]["frozen_primary_success_indices"].append(24)
        )
        self.bad("fresh sample index out of range")

    def test_missing_response_review(self):
        self.edit(checker.LEDGER, lambda d: d["events"][0]["response_checks"].pop(0))
        self.bad("response coverage/order")

    def test_boolean_reward(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["events"][0]["response_checks"][0].update(frozen_reward=False),
        )
        self.bad("response frozen reward")

    def test_integer_cap_flag(self):
        self.edit(checker.LEDGER, lambda d: d["events"][0]["response_checks"][0].update(cap_hit=0))
        self.bad("reviewed response cap flag")

    def test_changed_terminal_literal(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["events"][0]["response_checks"][0].update(terminal_numeric_literal="9"),
        )
        self.bad("terminal literal and parsed value disagree")

    def test_noncanonical_parsed_value_cannot_hide_exact_match(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["events"][0]["response_checks"][0].update(
                terminal_numeric_literal="8.0", parsed_value="8.0"
            ),
        )
        self.bad("not a canonical frozen-v2 numeric value")

    def test_exponent_literal_rejected_by_frozen_numeric_grammar(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["events"][0]["response_checks"][0].update(
                terminal_numeric_literal="9e0", parsed_value="9"
            ),
        )
        self.bad("terminal literal and parsed value disagree")

    def test_noncanonical_recorded_gold_rejected(self):
        self.edit(checker.LEDGER, lambda d: d["cases"][0].update(recorded_gold_value="8.0"))
        self.bad("recorded gold value: not a canonical")

    def test_changed_conservative_witness_evidence(self):
        self.edit(
            checker.LEDGER,
            lambda d: d["events"][0]["response_checks"][0].update(
                reconciled_numeric_status="correct"
            ),
        )
        self.bad("witness review rule")

    def test_response_total_mismatch(self):
        self.edit(
            checker.SUMMARY, lambda d: d["event_output_review"].update(total_raw_responses=125)
        )
        self.bad("response summary.total_raw_responses")

    def test_reasoning_caveat_hash_mismatch(self):
        self.edit(
            checker.SUMMARY,
            lambda d: d["event_output_review"][
                "numeric_correct_does_not_mean_reasoning_valid"
            ].update(text_raw_sha256="0" * 64),
        )
        self.bad("reasoning caveat content binding")

    def test_thesis_projection_drift(self):
        self.edit(
            f"{checker.REVISION}/thesis/c1_case_ledger.json",
            lambda d: d["cases"][0].update(reconciliation_note="Changed"),
        )
        self.bad("thesis ledger copy")

    def test_addendum_projection_drift(self):
        self.edit(
            f"{checker.REVISION}/addendum/case_ledger_public.json",
            lambda d: d["events"][0]["response_checks"][0].update(cap_hit=True),
        )
        self.bad("addendum.events projection")

    def test_csv_projection_drift(self):
        target = self.root / f"{checker.RESULTS}/c1_case_ledger.csv"
        target.write_text(target.read_text().replace("Gold8 is", "Changed8 is", 1))
        self.bad("CSV projection")

    def test_duplicate_csv_header(self):
        target = self.root / f"{checker.RESULTS}/c1_case_ledger.csv"
        target.write_text(
            target.read_text().replace(
                "selected_index,original_prompt_rank", "selected_index,selected_index", 1
            )
        )
        self.bad("CSV header")

    def test_addendum_csv_drift(self):
        target = self.root / f"{checker.REVISION}/addendum/case_ledger_public.csv"
        with target.open(newline="") as handle:
            rows = list(csv.reader(handle))
        rows[1][2] = "0" * 64
        with target.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        self.bad("CSV projection")

    def test_missing_method_seed_case(self):
        self.edit(checker.METHOD, lambda d: d["selected_case_seeds"].pop())
        self.bad("method seed case count")

    def test_method_ordering_seed_mismatch(self):
        self.edit(checker.METHOD, lambda d: d.update(seed=1))
        self.bad("selected deterministic order mismatch")

    def test_method_seed_binding_mismatch(self):
        self.edit(
            checker.METHOD, lambda d: d["selected_case_seeds"][0].update(question_sha256="0" * 64)
        )
        self.bad("method seed identity")

    def test_method_seed_spacing_mismatch(self):
        self.edit(
            checker.METHOD, lambda d: d["selected_case_seeds"][0]["fresh_seeds"].__setitem__(0, 1)
        )
        self.bad("affine seed spacing")

    def test_method_swapped_case_seed_arrays_rejected(self):
        def swap_seeds(data):
            first, second = data["selected_case_seeds"][:2]
            for key in ("initial_seeds", "fresh_seeds"):
                first[key], second[key] = second[key], first[key]

        self.edit(checker.METHOD, swap_seeds)
        self.bad("cross-case screen-index stride")

    def test_method_shifted_single_phase_rejected(self):
        self.edit(
            checker.METHOD,
            lambda d: d["selected_case_seeds"][0].update(
                fresh_seeds=[value + 1 for value in d["selected_case_seeds"][0]["fresh_seeds"]]
            ),
        )
        self.bad("cross-case screen-index stride")

    def test_method_slot_offset_bound_to_published_prose(self):
        self.edit(checker.METHOD, lambda d: d.update(generation_seed_slot_offset=0))
        self.bad("method prose.generation slot offset")

    def test_absolute_seed_origin_is_explicitly_not_reconstructed(self):
        # Shifting every phase origin consistently preserves relative strides.
        # The independent publication manifest catches the changed file bytes;
        # this consistency gate must not imply an omitted absolute derivation.
        def shift_origins(data):
            modulus = data["seed_scheme"]["affine_modulus"]
            for row in data["selected_case_seeds"]:
                for key in ("initial_seeds", "fresh_seeds"):
                    row[key] = [(value + 1) % modulus for value in row[key]]

        self.edit(checker.METHOD, shift_origins)
        result = checker.check_public_results(self.root)
        self.assertIn("absolute generation-seed origin reconstruction", result["limitation"])

    def test_method_parser_binding_mismatch(self):
        self.edit(checker.METHOD, lambda d: d.update(parser_sha256="0" * 64))
        self.bad("original parser source binding")

    def test_method_prompt_suffix_hash_mismatch(self):
        self.edit(checker.METHOD, lambda d: d.update(prompt_suffix="Altered prompt"))
        self.bad("prompt_suffix hash")

    def test_summary_source_binding_mismatch(self):
        self.edit(checker.SUMMARY, lambda d: d.update(source_summary_sha256="0" * 64))
        self.bad("review summary source binding")

    def test_duplicate_source_binding(self):
        self.edit(checker.BINDINGS, lambda d: d["sources"].append(d["sources"][0]))
        self.bad("duplicate source binding")

    def test_document_case_classification_drift(self):
        target = self.root / checker.DOCUMENTS[0]
        target.write_text(
            target.read_text().replace("| C00 | 676 | 8 | S | E N |", "| C00 | 676 | 8 | W | E N |")
        )
        self.bad("case table")

    def test_document_event_success_drift(self):
        target = self.root / checker.DOCUMENTS[1]
        target.write_text(
            target.read_text().replace(
                "| 12, 15, 23 | Corroborated |", "| 12, 15 | Corroborated |", 1
            )
        )
        self.bad("event table")


if __name__ == "__main__":
    unittest.main()
